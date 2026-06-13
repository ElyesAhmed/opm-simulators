// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 * \copydoc Opm::CpGridVanguard
 */
#ifndef OPM_CPGRID_VANGUARD_HPP
#define OPM_CPGRID_VANGUARD_HPP

#include <opm/common/TimingMacros.hpp>

#include <opm/models/common/multiphasebaseproperties.hh>
#include <opm/models/blackoil/blackoilproperties.hh>
#include <opm/simulators/flow/FemCpGridCompat.hpp>
#include <opm/simulators/flow/FlowBaseVanguard.hpp>
#include <opm/simulators/flow/GenericCpGridVanguard.hpp>
#include <opm/simulators/flow/Transmissibility.hpp>

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace Opm {
template <class TypeTag>
class CpGridVanguard;

namespace detail {
// Detect whether the grid supports CpGrid::setPartitionCellGroups (present
// in the LGR-refinement grid fork, absent in upstream opm-grid). Lets the
// LGR-aware partitioning below compile against both.
template <class G, class = void>
struct HasSetPartitionCellGroups : std::false_type {};
template <class G>
struct HasSetPartitionCellGroups<
    G, std::void_t<decltype(std::declval<G&>().setPartitionCellGroups(
           std::declval<std::vector<std::set<int>>>()))>> : std::true_type {};
} // namespace detail
}

namespace Opm::Properties {

namespace TTag {
struct CpGridVanguard {
    using InheritsFrom = std::tuple<FlowBaseVanguard>;
};
}

// declare the properties
template<class TypeTag>
struct Vanguard<TypeTag, TTag::CpGridVanguard> {
    using type = CpGridVanguard<TypeTag>;
};
template<class TypeTag>
struct Grid<TypeTag, TTag::CpGridVanguard> {
    using type = Dune::CpGrid;
};
template<class TypeTag>
struct EquilGrid<TypeTag, TTag::CpGridVanguard> {
    using type = GetPropType<TypeTag, Properties::Grid>;
};

} // namespace Opm::Properties

namespace Opm {

/*!
 * \ingroup BlackOilSimulator
 *
 * \brief Helper class for grid instantiation of ECL file-format using problems.
 *
 * This class uses Dune::CpGrid as the simulation grid.
 */
template <class TypeTag>
class CpGridVanguard : public FlowBaseVanguard<TypeTag>
                     , public GenericCpGridVanguard<GetPropType<TypeTag, Properties::ElementMapper>,
                                                    GetPropType<TypeTag, Properties::GridView>,
                                                    GetPropType<TypeTag, Properties::Scalar>>
{
    friend class FlowBaseVanguard<TypeTag>;
    using ParentType = FlowBaseVanguard<TypeTag>;

    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using ElementMapper = GetPropType<TypeTag, Properties::ElementMapper>;

public:
    using Grid = GetPropType<TypeTag, Properties::Grid>;
    using CartesianIndexMapper = Dune::CartesianIndexMapper<Grid>;
    using EquilGrid = GetPropType<TypeTag, Properties::EquilGrid>;
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using TransmissibilityType = Transmissibility<Grid, GridView, ElementMapper, CartesianIndexMapper, Scalar>;
    static constexpr int dimensionworld = Grid::dimensionworld;
    using Indices = GetPropType<TypeTag, Properties::Indices>;
    static constexpr bool waterEnabled = Indices::waterEnabled;
    static constexpr bool gasEnabled = Indices::gasEnabled;
    static constexpr bool oilEnabled = Indices::oilEnabled;
private:
    using Element = typename GridView::template Codim<0>::Entity;

public:
    explicit CpGridVanguard(Simulator& simulator)
        : FlowBaseVanguard<TypeTag>(simulator)
    {
        this->checkConsistency();
        this->callImplementationInit();
    }

    int compressedIndexForInteriorLGR(const std::string& lgr_tag, const Connection& conn) const override
    {
        const std::array<int,3> lgr_ijk = {conn.getI(), conn.getJ(), conn.getK()};
        const auto& lgr_level = this->grid().getLgrNameToLevel().at(lgr_tag);
        if (ParentType::lgrMappers_.has_value() == false) {
            ParentType::lgrMappers_.emplace(this->grid().mapLocalCartesianIndexSetsToLeafIndexSet());
        }
        const auto& lgr_dim = this->grid().currentData()[lgr_level]->logicalCartesianSize();
        const auto lgr_cartesian_index = (lgr_ijk[2]*lgr_dim[0]*lgr_dim[1]) + (lgr_ijk[1]*lgr_dim[0]) + (lgr_ijk[0]);
        return ParentType::lgrMappers_.value()[lgr_level].at(lgr_cartesian_index);
    }
    /*!
     * Checking consistency of simulator
     */
    void checkConsistency()
    {
        const auto& runspec = this->eclState().runspec();
        const auto& config = this->eclState().getSimulationConfig();
        const auto& phases = runspec.phases();

        // check for correct module setup
        if (config.isThermal()) {
            if (getPropValue<TypeTag, Properties::EnergyModuleType>() != EnergyModules::FullyImplicitThermal) {
                throw std::runtime_error("Input specifies energy while simulator has disabled it, try xxx_energy");
            }
        } else {
            if (getPropValue<TypeTag, Properties::EnergyModuleType>() == EnergyModules::FullyImplicitThermal) {
                throw std::runtime_error("Input specifies no energy while simulator has energy, try run without _energy");
            }
        }

        if (config.isDiffusive()) {
            if (getPropValue<TypeTag, Properties::EnableDiffusion>() == false) {
                throw std::runtime_error("Input specifies diffusion while simulator has disabled it, try xxx_diffusion");
            }
        }

        if (runspec.micp()) {
            if (getPropValue<TypeTag, Properties::EnableBioeffects>() == false) {
                throw std::runtime_error("Input specifies MICP while simulator has it disabled");
            }
        }

        if (runspec.biof()) {
            if (getPropValue<TypeTag, Properties::EnableBioeffects>() == false) {
                throw std::runtime_error("Input specifies Biofilm while simulator has it disabled");
            }
        }

        if (phases.active(Phase::BRINE)) {
            if (getPropValue<TypeTag, Properties::EnableBrine>() == false) {
                throw std::runtime_error("Input specifies Brine while simulator has it disabled");
            }
        }

        if (phases.active(Phase::POLYMER)) {
            if (getPropValue<TypeTag, Properties::EnablePolymer>() == false) {
                throw std::runtime_error("Input specifies Polymer while simulator has it disabled");
            }
        }

        // checking for correct phases is more difficult TODO!
        if (phases.active(Phase::ZFRACTION)) {
            if (getPropValue<TypeTag, Properties::EnableExtbo>() == false) {
                throw std::runtime_error("Input specifies ExBo while simulator has it disabled");
            }
        }
        if (phases.active(Phase::FOAM)) {
            if (getPropValue<TypeTag, Properties::EnableFoam>() == false) {
                throw std::runtime_error("Input specifies Foam while simulator has it disabled");
            }
        }

        if (phases.active(Phase::SOLVENT)) {
            if (getPropValue<TypeTag, Properties::EnableSolvent>() == false) {
                throw std::runtime_error("Input specifies Solvent while simulator has it disabled");
            }
        }
        if(phases.active(Phase::WATER)){
            if(waterEnabled == false){
                throw std::runtime_error("Input specifies water while simulator has it disabled");
            }
        }
        if(phases.active(Phase::GAS)){
            if(gasEnabled == false){
                throw std::runtime_error("Input specifies gas while simulator has it disabled");
            }
        }
        if(phases.active(Phase::OIL)){
            if(oilEnabled == false){
                throw std::runtime_error("Input specifies oil while simulator has it disabled");
            }
        }

    }

    /*!
     * \brief Free the memory occupied by the global transmissibility object.
     *
     * After writing the initial solution, this array should not be necessary anymore.
     */
    void releaseGlobalTransmissibilities()
    {
        globalTrans_.reset();
    }

    const TransmissibilityType& globalTransmissibility() const
    {
        assert( globalTrans_ != nullptr );
        return *globalTrans_;
    }

    /*!
     * \brief Tell the partitioner to keep each LGR region on one rank.
     *
     * Builds one cell group per connected set of CARFIN boxes (boxes that
     * touch or overlap are merged, so their shared refined boundary stays on
     * one rank) and hands the groups to the grid's partitioner. Returns true
     * if the grid supports this (the LGR-refinement fork) and groups were set.
     */
    template <class Lgrs>
    bool applyLgrPartitionCellGroups_([[maybe_unused]] const Lgrs& lgrs)
    {
        if constexpr (detail::HasSetPartitionCellGroups<Grid>::value) {
            const auto dims = this->grid_->logicalCartesianSize();
            struct Box { int i0, i1, j0, j1, k0, k1; };
            std::vector<Box> boxes;
            boxes.reserve(lgrs.size());
            for (std::size_t l = 0; l < lgrs.size(); ++l) {
                const auto c = lgrs.getLgr(static_cast<int>(l));
                boxes.push_back({c.I1(), c.I2() + 1, c.J1(), c.J2() + 1, c.K1(), c.K2() + 1});
            }
            // Union-find: merge boxes that meet in every direction (touch or
            // overlap), i.e. share a face/edge/corner.
            std::vector<int> parent(boxes.size());
            std::iota(parent.begin(), parent.end(), 0);
            std::function<int(int)> find = [&](int x) {
                while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
                return x;
            };
            const auto meet = [](int a0, int a1, int b0, int b1) { return a1 >= b0 && b1 >= a0; };
            for (std::size_t i = 0; i < boxes.size(); ++i) {
                for (std::size_t j = i + 1; j < boxes.size(); ++j) {
                    if (meet(boxes[i].i0, boxes[i].i1, boxes[j].i0, boxes[j].i1)
                        && meet(boxes[i].j0, boxes[i].j1, boxes[j].j0, boxes[j].j1)
                        && meet(boxes[i].k0, boxes[i].k1, boxes[j].k0, boxes[j].k1)) {
                        parent[find(static_cast<int>(i))] = find(static_cast<int>(j));
                    }
                }
            }
            // Each group is the box(es) PLUS a halo of `halo` cells. The halo
            // keeps the box that far inside its owning rank, so the box cells
            // never appear in another rank's overlap (which would make the
            // refinement builder reject the box as touching the overlap). The
            // halo must be at least the overlap-layer count used below (2).
            const int halo = 2;
            std::map<int, std::set<int>> groups;
            for (std::size_t b = 0; b < boxes.size(); ++b) {
                auto& cells = groups[find(static_cast<int>(b))];
                const auto& bx = boxes[b];
                const int i0 = std::max(0, bx.i0 - halo), i1 = std::min(dims[0], bx.i1 + halo);
                const int j0 = std::max(0, bx.j0 - halo), j1 = std::min(dims[1], bx.j1 + halo);
                const int k0 = std::max(0, bx.k0 - halo), k1 = std::min(dims[2], bx.k1 + halo);
                for (int k = k0; k < k1; ++k) {
                    for (int j = j0; j < j1; ++j) {
                        for (int i = i0; i < i1; ++i) {
                            cells.insert(i + dims[0] * j + dims[0] * dims[1] * k);
                        }
                    }
                }
            }
            std::vector<std::set<int>> cellGroups;
            cellGroups.reserve(groups.size());
            for (auto& [root, cells] : groups) {
                cellGroups.push_back(std::move(cells));
            }
            const auto numGroups = cellGroups.size();
            this->grid_->setPartitionCellGroups(std::move(cellGroups));
            OpmLog::info("Keeping " + std::to_string(numGroups)
                         + " LGR region(s) (with halo) together for load balancing.");
            return true;
        }
        return false;
    }

    /*!
     * \brief Distribute the simulation grid over multiple processes
     *
     * (For parallel simulation runs.)
     */
    void loadBalance()
    {
#if HAVE_MPI
        if (const auto& extPFile = this->externalPartitionFile();
            !extPFile.empty() && (extPFile != "none"))
        {
            this->setExternalLoadBalancer(details::MPIPartitionFromFile { extPFile });
        }

        // LGR-aware partitioning: keep each refinement region (and groups of
        // touching ones) on a single rank so the rank-interior refinement
        // builder never splits a box across ranks. Needs the cell-group
        // partitioner (zoltanGoG) and, for a contracted interior region,
        // overlap layer 2 (a single layer misses corner/edge neighbours).
        int overlapLayers = this->numOverlap();
        auto partMethod = this->partitionMethod();
        if (this->grid_->comm().size() > 1) {
            if (const auto& lgrs = this->eclState().getLgrs(); lgrs.size() > 0) {
                if (applyLgrPartitionCellGroups_(lgrs)) {
                    overlapLayers = std::max(overlapLayers, 2);
                    partMethod = Dune::PartitionMethod::zoltanGoG;
                }
            }
        }

        this->doLoadBalance_(this->edgeWeightsMethod(), this->ownersFirst(),
                             this->addCorners(), overlapLayers,
                             partMethod, this->serialPartitioning(),
                             this->enableDistributedWells(),
                             this->allow_splitting_inactive_wells_,
                             this->imbalanceTol(),
                             this->gridView(), this->schedule(),
                             this->eclState(), this->parallelWells_,
                             this->numJacobiBlocks(), this->enableEclOutput());
#endif

        this->updateGridView_();
        this->updateCartesianToCompressedMapping_();
        this->updateCellThickness_();

#if HAVE_MPI
        this->distributeFieldProps_(this->eclState());
#endif

        // Must be done after the field properties have been distributed, since the
        // DEPTH property is needed on all ranks to honour DEPTH in the EDIT section.
        this->updateCellDepths_();
    }

    /*!
     * \brief Add LGRs and update Leaf Grid View in the simulation grid.
     */
    void addLgrs()
    {
        // Check if input file contains Lgrs. Add them, if any.
        // In a parallel run, this adds the LGRs on the distributed simulation grid.
        if (const auto& lgrs = this->eclState().getLgrs(); lgrs.size() > 0) {
            OpmLog::info("\nAdding LGRs to the grid and updating its leaf grid view");
            this->addLgrsUpdateLeafView(lgrs, lgrs.size(), *this->grid_);

            this->updateGridView_();
            this->updateCellDepths_();
            this->updateCellThickness_();

            // The global-view refinement + id sync below is the original
            // implementation's way to make refined cell ids globally
            // consistent. The rank-interior refinement builder (the fork)
            // produces deterministic per-rank refinement and does not use it.
            if constexpr (!detail::HasSetPartitionCellGroups<Grid>::value) {
                if (this->grid_->comm().size()>1) {
                    // Add LGRs and update the leaf grid view in the global (undistributed) simulation grid.
                    // Purpose: To enable synchronization of cell ids in 'serial mode',
                    //          we rely on the "parent-to-children" cell id mapping.
                    OpmLog::info("\nAdding LGRs to the global view and updating its leaf grid view");
                    this->grid_->switchToGlobalView();
                    this->addLgrsUpdateLeafView(lgrs, lgrs.size(), *this->grid_);
                    this->grid_->switchToDistributedView();
                    this->grid_->syncDistributedGlobalCellIds();
                }
            }
        }
    }

    unsigned int gridEquilIdxToGridIdx(unsigned int elemIndex) const {
        return elemIndex;
    }

    unsigned int gridIdxToEquilGridIdx(unsigned int elemIndex) const {
        return elemIndex;
    }
    /*!
     * \brief Get function to query cell centroids for a distributed grid.
     *
     * Currently this only non-empty for a loadbalanced CpGrid.
     * It is a function return the centroid for the given element
     * index.
     */
    std::function<std::array<double,dimensionworld>(int)>
    cellCentroids() const
    {
        return this->cellCentroids_(this->cartesianIndexMapper(), true);
    }

    const std::vector<int>& globalCell()
    {
        return this->grid().globalCell();
    }

protected:
    void createGrids_()
    {
        this->doCreateGrids_(this->edgeConformal(), this->eclState());
    }

    void allocTrans() override
    {
        OPM_TIMEBLOCK(allocateTrans);
        globalTrans_.reset(new TransmissibilityType(this->eclState(),
                                                    this->gridView(),
                                                    this->cartesianIndexMapper(),
                                                    this->grid(),
                                                    this->cellCentroids(),
                                                    getPropValue<TypeTag, Properties::EnergyModuleType>() == EnergyModules::FullyImplicitThermal ||
                                                    getPropValue<TypeTag, Properties::EnergyModuleType>() == EnergyModules::SequentialImplicitThermal,
                                                    getPropValue<TypeTag, Properties::EnableDiffusion>(),
                                                    getPropValue<TypeTag, Properties::EnableDispersion>()));
        globalTrans_->update(false, TransmissibilityType::TransUpdateQuantities::Trans);
    }

    double getTransmissibility(unsigned I, unsigned J) const override
    {
       return globalTrans_->transmissibility(I,J);
    }

#if HAVE_MPI
    const std::string& zoltanParams() const override
    {
        return this->zoltanParams_;
    }

    double zoltanPhgEdgeSizeThreshold() const override
    {
        return this->zoltanPhgEdgeSizeThreshold_;
    }

    const std::string& metisParams() const override
    {
        return this->metisParams_;
    }
#endif

    // \Note: this globalTrans_ is used for domain decomposition and INIT file output.
    // It only contains trans_ due to permeability and does not contain thermalHalfTrans_,
    // diffusivity_ abd dispersivity_. The main reason is to reduce the memory usage for rank 0
    // during parallel running.
    std::unique_ptr<TransmissibilityType> globalTrans_;
};

} // namespace Opm

#endif // OPM_CPGRID_VANGUARD_HPP
