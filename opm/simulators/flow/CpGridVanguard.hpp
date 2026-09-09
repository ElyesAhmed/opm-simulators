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

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
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
        // Mirror compressedIndexForInterior: return -1 when the LGR-completed
        // connection's cell is not interior on this rank.  In a parallel run
        // the refinement is rank-interior - the box is refined only on the rank
        // that owns it - so on every other rank the level grid for this LGR is
        // empty and the cell is simply absent.  Using .at() there throws
        // std::out_of_range (seen as a setup hang at higher rank counts, where
        // more ranks do not own the box); a missing cell must yield -1 instead.
        const auto& nameToLevel = this->grid().getLgrNameToLevel();
        const auto levelIt = nameToLevel.find(lgr_tag);
        if (levelIt == nameToLevel.end()) {
            return -1;
        }
        const int lgr_level = levelIt->second;

        // refine-before-redistribute: the distributed grid is the flat refined
        // leaf (refined cells but maxLevel()==0, so the per-level grids used by
        // mapLocalCartesianIndexSetsToLeafIndexSet / currentData()[lgr_level] do
        // not exist on this rank).  Static refinement means the
        // (LGR-local Cartesian -> leaf cell) relation is fixed and known at
        // build time, so reconstruct it once from each leaf cell's parent
        // Cartesian (globalCell()) + index-in-parent + the static CARFIN box,
        // and look the connection up there.
        if (this->grid().maxLevel() == 0 && this->grid().leafHasParentCellIndices()) {
            return this->compressedIndexForInteriorLGRFlat_(lgr_tag, lgr_level, conn);
        }

        if (ParentType::lgrMappers_.has_value() == false) {
            ParentType::lgrMappers_.emplace(this->grid().mapLocalCartesianIndexSetsToLeafIndexSet());
        }
        const auto& mappers = ParentType::lgrMappers_.value();
        if (lgr_level < 0 || static_cast<std::size_t>(lgr_level) >= mappers.size()) {
            return -1;
        }

        const auto& lgr_dim = this->grid().currentData()[lgr_level]->logicalCartesianSize();
        const std::array<int,3> lgr_ijk = {conn.getI(), conn.getJ(), conn.getK()};
        const auto lgr_cartesian_index = static_cast<std::size_t>(
            (lgr_ijk[2]*lgr_dim[0]*lgr_dim[1]) + (lgr_ijk[1]*lgr_dim[0]) + lgr_ijk[0]);

        const auto& mapper = mappers[lgr_level];
        const auto it = mapper.find(lgr_cartesian_index);
        if (it == mapper.end()) {
            return -1; // cell of this LGR is not present on this rank
        }
        return static_cast<int>(it->second);
    }

    //! \brief Resolve an LGR-completed connection on the flat refine-before leaf.
    //!
    //! On the refine-before-redistribute path the distributed grid is the flat
    //! refined leaf (maxLevel()==0); there are no per-level grids to index.  The
    //! refinement is static, so we reconstruct, once and cached, the per-level
    //! (LGR-local Cartesian -> local leaf cell) map directly from each leaf
    //! cell's parent Cartesian index (globalCell(), shared by refined siblings),
    //! its index-in-parent, and the static CARFIN box (offset + refinement
    //! factors).  Single nesting level (parent of every refined cell is a
    //! level-0 coarse cell).
    int compressedIndexForInteriorLGRFlat_(const std::string& lgr_tag,
                                           int lgr_level,
                                           const Connection& conn) const
    {
        if (!flatLgrMappers_.has_value()) {
            this->buildFlatLgrMappers_();
        }
        const auto& mappers = flatLgrMappers_.value();
        if (lgr_level < 0 || static_cast<std::size_t>(lgr_level) >= mappers.size()) {
            return -1;
        }
        // LGR-local Cartesian index of the connection, using the LGR's refined
        // dimensions straight from the (static) CARFIN definition.
        const auto& carfin = this->eclState().getLgrs().getLgr(lgr_tag);
        const int nx = carfin.NX();
        const int ny = carfin.NY();
        const auto lgr_cartesian_index = static_cast<std::size_t>(
            (conn.getK()*nx*ny) + (conn.getJ()*nx) + conn.getI());

        const auto& mapper = mappers[lgr_level];
        const auto it = mapper.find(lgr_cartesian_index);
        if (it == mapper.end()) {
            return -1; // cell of this LGR is not present on this rank
        }
        return it->second;
    }

    //! \brief Build (and cache) the flat refine-before LGR-local-Cartesian maps.
    void buildFlatLgrMappers_() const
    {
        const auto& lgrs = this->eclState().getLgrs();
        const auto& nameToLevel = this->grid().getLgrNameToLevel();
        const auto& dims0 = this->grid().logicalCartesianSize(); // global level-0 dims

        struct Box { int i0,j0,k0,i1,j1,k1, rx,ry,rz, nx,ny, level; };
        std::vector<Box> boxes;
        int maxLevel = 0;
        for (const auto& [name, level] : nameToLevel) {
            if (level == 0) {
                continue;
            }
            const auto& c = lgrs.getLgr(name);
            const int ni = c.I2() - c.I1() + 1;
            const int nj = c.J2() - c.J1() + 1;
            const int nk = c.K2() - c.K1() + 1;
            boxes.push_back({c.I1(), c.J1(), c.K1(), c.I2(), c.J2(), c.K2(),
                             c.NX()/ni, c.NY()/nj, c.NZ()/nk, c.NX(), c.NY(), level});
            maxLevel = std::max(maxLevel, level);
        }

        std::vector<std::unordered_map<std::size_t,int>> mappers(maxLevel + 1);
        const auto& gc = this->grid().globalCell();
        // Interior partition only: mirrors compressedIndexForInterior - a
        // connection must resolve on the single rank that owns the cell, not on
        // ranks that merely carry it as an overlap copy (which would perforate
        // the well twice).
        for (const auto& elem : elements(this->gridView(), Dune::Partitions::interior)) {
            const int idxInParent = elem.getIdxInParentCell();
            if (idxInParent < 0) {
                continue; // coarse leaf cell - not part of any LGR
            }
            const int leafIdx = elem.index();
            const int pc = gc[leafIdx];                  // parent (level-0) Cartesian
            const int pi = pc % dims0[0];
            const int pj = (pc / dims0[0]) % dims0[1];
            const int pk = pc / (dims0[0]*dims0[1]);
            for (const auto& b : boxes) {
                if (pi >= b.i0 && pi <= b.i1 && pj >= b.j0 && pj <= b.j1 &&
                    pk >= b.k0 && pk <= b.k1) {
                    const int di = idxInParent % b.rx;
                    const int dj = (idxInParent / b.rx) % b.ry;
                    const int dk = idxInParent / (b.rx*b.ry);
                    const int li = (pi - b.i0)*b.rx + di;
                    const int lj = (pj - b.j0)*b.ry + dj;
                    const int lk = (pk - b.k0)*b.rz + dk;
                    const auto lcart = static_cast<std::size_t>(
                        (lk*b.nx*b.ny) + (lj*b.nx) + li);
                    mappers[b.level][lcart] = leafIdx;
                    break;
                }
            }
        }
        flatLgrMappers_.emplace(std::move(mappers));
    }

    //! Cached flat refine-before maps: level -> (LGR-local Cartesian -> leaf idx).
    mutable std::optional<std::vector<std::unordered_map<std::size_t,int>>> flatLgrMappers_;

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
                // A nested LGR (parent != GLOBAL) addresses its parent LGR's own
                // local Cartesian space, so its I/J/K extents are NOT global cell
                // indices and must not be turned into a global box (that yields
                // out-of-range cells and a corrupt partition group). A fully
                // contained nested LGR lives inside its parent's box and is kept
                // on the parent's rank by the parent's group, so skip it here.
                if (c.PARENT_NAME() != "GLOBAL") {
                    continue;
                }
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

            // The rank-interior model requires each group to live entirely on one
            // rank. If a single group is so large that confining it to one rank
            // would leave another rank with no cells, the constraint is
            // unsatisfiable (e.g. an LGR that covers essentially the whole grid).
            // Detect that here and fail with a clear, actionable message rather
            // than aborting deep in the partitioner (a zero-cells error or, worse,
            // a non-deterministic assertion while contracting the giant group).
            // The groups are built identically on every rank from the box extents,
            // so this Cartesian-based test is collective-safe without communication.
            const int numRanks = this->grid_->comm().size();
            std::size_t maxGroup = 0;
            for (const auto& g : cellGroups) {
                maxGroup = std::max(maxGroup, g.size());
            }
            const std::size_t totalCart = static_cast<std::size_t>(dims[0]) * dims[1] * dims[2];
            if (numRanks > 1 && maxGroup + static_cast<std::size_t>(numRanks - 1) > totalCart) {
                OPM_THROW(std::runtime_error,
                          "An LGR refinement region (with its halo) spans "
                          + std::to_string(maxGroup) + " of " + std::to_string(totalCart)
                          + " cells, too much to keep on a single rank when running on "
                          + std::to_string(numRanks) + " MPI ranks. The rank-interior "
                          "parallel LGR model needs each refinement box to fit inside one "
                          "rank's interior, so an LGR covering (nearly) the whole grid "
                          "cannot be distributed. Run this case on a single MPI rank, or "
                          "reduce the extent of the refinement box(es).");
            }

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
                if (this->refineBeforeRedistribute()) {
                    // Experimental refine-then-distribute path (opt-in via
                    // --refine-before-redistribute). Refine the global grid now,
                    // before load balancing, instead of the default
                    // rank-interior model (distribute the coarse grid, then
                    // refine each box on its owning rank in addLgrs()).
                    //
                    // WARNING: CpGrid's scatter currently distributes only
                    // level 0 and discards the refinement (the distributed-
                    // refinement machinery is stripped from this fork), so this
                    // only yields a correct result in serial / single-rank runs.
                    // It is kept as a selectable option for the day refined-grid
                    // distribution is reinstated; the default remains the
                    // working rank-interior path below.
                    OpmLog::info("\nRefine-before-redistribute: adding LGRs to "
                                 "the grid before load balancing");
                    this->addLgrsUpdateLeafView(lgrs, lgrs.size(), *this->grid_);
                    this->updateGridView_();
                    // Keep each box's coarse cells together for the level-zero
                    // partition, exactly as the rank-interior path does. The
                    // leaf partition is then derived by propagating the level-
                    // zero partition to the children (CpGrid::leafPartition-
                    // FromLevelZero), so the box's refined cells all land on
                    // one rank.
                    if (applyLgrPartitionCellGroups_(lgrs)) {
                        overlapLayers = std::max(overlapLayers, 2);
                        partMethod = Dune::PartitionMethod::zoltanGoG;
                    }
                } else if (applyLgrPartitionCellGroups_(lgrs)) {
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
        this->updateCellDepths_();
        this->updateCellThickness_();

#if HAVE_MPI
        this->distributeFieldProps_(this->eclState());
#endif
    }

    /*!
     * \brief Add LGRs and update Leaf Grid View in the simulation grid.
     */
    void addLgrs()
    {
        // Check if input file contains Lgrs. Add them, if any.
        // In a parallel run, this adds the LGRs on the distributed simulation grid.
        if (const auto& lgrs = this->eclState().getLgrs(); lgrs.size() > 0) {
            // With the experimental refine-before-redistribute option the LGRs
            // are already added in loadBalance() for a parallel run (before the
            // grid is distributed), so don't add them again here. The check is on
            // comm().size() > 1, not maxLevel(): once the refined leaf has been
            // distributed the distributed grid carries the leaf as its only level
            // (maxLevel() == 0 again), so a maxLevel() test would wrongly re-add
            // the LGRs. In a serial refine-before run loadBalance() does not add
            // the LGRs, so they are still added below.
            if (this->refineBeforeRedistribute() && this->grid_->comm().size() > 1) {
                return;
            }
            OpmLog::info("\nAdding LGRs to the grid and updating its leaf grid view");
            this->addLgrsUpdateLeafView(lgrs, lgrs.size(), *this->grid_);

            this->updateGridView_();
            // Refinement changes the leaf cell count and ordering, so the
            // Cartesian->compressed map built during load balancing (on the
            // unrefined grid) is now stale. Rebuild it before well connections
            // are resolved, otherwise coarse-grid wells map to the wrong leaf
            // cell (e.g. their source term lands on an unrelated cell).
            this->updateCartesianToCompressedMapping_();
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

            // WELTRAJ/COMPTRAJ wells had their connections computed against the
            // coarse EclipseGrid at parse time, which cannot address individual
            // refined cells. Recompute them against the refined leaf so a well
            // trajectory perforates the correct LGR leaf cells. This is a
            // post-process on the Schedule; the well-model solve is untouched.
            // Done before the canary below: it legitimately reads refined cells'
            // geometry while building the trajectory connections.
            this->recomputeWellTrajectoriesInLgr_();

            // Opt-in canary (OPM_LGR_POISON_REFINED=1): poison every refined leaf
            // cell's global Cartesian index now that the grid is built and the
            // Cartesian->compressed map has been (re)built. If anything later
            // re-derives a refined cell's properties from globalCell() (i.e.
            // re-reads the parent Cartesian instead of using the materialized
            // per-cell arrays) it will index out of bounds and fail loudly. A
            // correct run is unaffected. Default off => zero impact on the
            // achieved (rank-interior / refine-before) behaviour.
            if (std::getenv("OPM_LGR_POISON_REFINED") != nullptr) {
                this->grid_->poisonRefinedGlobalCell(-1);
                OpmLog::info("\n[canary] poisoned refined leaf cells' global Cartesian index "
                             "(OPM_LGR_POISON_REFINED): any late re-derivation will now fail loudly.");
            }
        }
    }

    /*!
     * \brief Post-process WELTRAJ/COMPTRAJ wells against the refined leaf grid.
     *
     * For each leaf cell we supply its corner geometry and, when it is a refined
     * (LGR) cell, its LGR-local (i,j,k) and owning LGR name. Schedule then
     * re-intersects each trajectory against this geometry and tags single-LGR
     * wells so the existing LGR connection path resolves them to the refined
     * leaf cells. Only runs when the grid is refined and trajectory wells exist.
     */
    void recomputeWellTrajectoriesInLgr_()
    {
        auto& grid = *this->grid_;
        if (grid.maxLevel() == 0) {
            return; // no refinement -> nothing to do
        }

        // Cheap pre-check: any trajectory wells at all?
        bool anyTraj = false;
        for (const auto& w : this->schedule().getWellsatEnd()) {
            if (w.getConnections().hasTrajectory()) { anyTraj = true; break; }
        }
        if (!anyTraj) {
            return;
        }

        const auto& gv = this->gridView();
        ElementMapper elemMapper(gv, Dune::mcmgElementLayout());
        const auto& cartMapper = this->cartesianIndexMapper();

        const std::size_t numCells = gv.size(0);

        // leaf cell index -> (level, LGR-local cartesian) for refined cells.
        const auto leafMappers = grid.mapLocalCartesianIndexSetsToLeafIndexSet();
        std::unordered_map<std::size_t, std::pair<int, std::size_t>> leafToLevelCart;
        for (std::size_t lev = 1; lev < leafMappers.size(); ++lev) {
            for (const auto& [lgrCart, leafIdx] : leafMappers[lev]) {
                leafToLevelCart[leafIdx] = { static_cast<int>(lev), lgrCart };
            }
        }
        std::unordered_map<int, std::string> levelToLgrName;
        for (const auto& [name, lev] : grid.getLgrNameToLevel()) {
            levelToLgrName[lev] = name;
        }

        // Cell properties for the connection-factor (Peaceman) computation.
        // Refined cells inherit their parent coarse cell's perm/ntg/satnum; the
        // field props live on the (coarse) input grid, indexed by parent cell.
        const auto& fp = this->eclState().fieldProps();
        const auto& permx = fp.get_double("PERMX");
        const auto& permy = fp.get_double("PERMY");
        const auto& permz = fp.get_double("PERMZ");
        const std::vector<double>* ntgPtr = fp.has_double("NTG") ? &fp.get_double("NTG") : nullptr;
        const std::vector<int>*    satPtr = fp.has_int("SATNUM") ? &fp.get_int("SATNUM") : nullptr;
        const auto& inputGrid = this->eclState().getInputGrid();

        std::vector<std::array<std::array<double, 3>, 8>> cellCorners(numCells);
        std::vector<std::optional<WellConnections::TrajectoryCell>> cellInfo(numCells);

        for (const auto& elem : elements(gv)) {
            const auto idx = elemMapper.index(elem);
            const auto geom = elem.geometry();

            double depth = 0.0;
            for (int c = 0; c < 8; ++c) {
                const auto corner = geom.corner(c);
                cellCorners[idx][c] = { corner[0], corner[1], corner[2] };
                depth += corner[2];
            }
            depth /= 8.0;

            WellConnections::TrajectoryCell tc;
            tc.depth = depth;

            // Perm/ntg/satnum from the parent coarse cell (refined cells inherit
            // them); dimensions are the actual (smaller) leaf cell extents, so
            // the Peaceman CTF/Kh are computed for the refined cell.
            const auto parentCart = static_cast<std::size_t>(cartMapper.cartesianIndex(idx));
            const auto ai = inputGrid.activeIndex(parentCart);
            tc.perm = { permx[ai], permy[ai], permz[ai] };
            tc.ntg = ntgPtr ? (*ntgPtr)[ai] : 1.0;
            tc.satnum = satPtr ? (*satPtr)[ai] : 1; // 1-based SATNUM region
            const auto edge = [&cellCorners, idx](int a, int b) {
                double s = 0.0;
                for (int d = 0; d < 3; ++d) {
                    const double q = cellCorners[idx][a][d] - cellCorners[idx][b][d];
                    s += q * q;
                }
                return std::sqrt(s);
            };
            // OPM corner order is binary i-fastest: 0=(0,0,0) 1=(1,0,0)
            // 2=(0,1,0) 4=(0,0,1) -> dx,dy,dz edge lengths.
            tc.dimensions = { edge(0, 1), edge(0, 2), edge(0, 4) };

            if (const auto it = leafToLevelCart.find(idx); it != leafToLevelCart.end()) {
                const int level = it->second.first;
                const std::size_t lgrCart = it->second.second;
                const auto& dim = grid.currentData()[level]->logicalCartesianSize();
                const std::size_t nx = dim[0];
                const std::size_t ny = dim[1];
                tc.ijk = { static_cast<int>(lgrCart % nx),
                           static_cast<int>((lgrCart / nx) % ny),
                           static_cast<int>(lgrCart / (nx * ny)) };
                const auto nameIt = levelToLgrName.find(level);
                tc.lgr_name = (nameIt != levelToLgrName.end()) ? nameIt->second : std::string{};

                // Record the connection in the *opm-common* LGR indexing (the
                // same encoding COMPDATL uses), so the ECL output places the well
                // in the refined grid and the connection's global index is
                // validated against the LGR grid (not the coarse grid).
                const auto& lgrLabels = inputGrid.get_all_lgr_labels();
                if (! tc.lgr_name.empty()
                    && std::find(lgrLabels.begin(), lgrLabels.end(),
                                 tc.lgr_name) != lgrLabels.end())
                {
                    // LGR grid number in the ScheduleGrid/COMPDATL convention:
                    // GLOBAL=0, LGRs=1,2,... get_lgr_cell_index is 0-based over
                    // the LGRs (GLOBAL excluded), so add one.
                    tc.lgr_grid = static_cast<int>(inputGrid.get_lgr_cell_index(tc.lgr_name)) + 1;
                    tc.global_index = inputGrid.getLGRCell(tc.lgr_name)
                                          .getGlobalIndex(static_cast<std::size_t>(tc.ijk[0]),
                                                          static_cast<std::size_t>(tc.ijk[1]),
                                                          static_cast<std::size_t>(tc.ijk[2]));
                }
                else if (! tc.lgr_name.empty()) {
                    // Adaptive (non-deck) LGR: the input EclipseGrid knows
                    // nothing about it, so there is no COMPDATL grid numbering
                    // to encode. Keep a non-zero grid number (the refinement
                    // level) so the connection is recognizably refined, and use
                    // the LGR-local Cartesian index as its global index. The
                    // runtime lookup (compressedIndexForInteriorLGR) only needs
                    // the LGR-local ijk + the CpGrid LGR name, both recorded.
                    tc.lgr_grid = level;
                    tc.global_index = lgrCart;
                }
                else {
                    tc.global_index = parentCart;
                }
            }
            else {
                std::array<int, 3> ijk{};
                cartMapper.cartesianCoordinate(idx, ijk);
                tc.ijk = ijk;
                tc.lgr_name.clear();
                tc.lgr_grid = 0;
                tc.global_index = parentCart;
            }

            cellInfo[idx] = tc;
        }

        auto cellInfoFn = [&cellInfo](std::size_t i)
            -> std::optional<WellConnections::TrajectoryCell>
        {
            if (i < cellInfo.size()) {
                return cellInfo[i];
            }
            return std::nullopt;
        };

        OpmLog::info("\nRecomputing well-trajectory connections against the refined grid");
        this->schedule().recomputeTrajectoryConnections(cellCorners, cellInfoFn);

        // Summarize the outcome (last report step) so refined-well placement
        // is visible in the log.
        for (const auto& w : this->schedule().getWellsatEnd()) {
            const auto& cs = w.getConnections();
            if (! cs.hasTrajectory()) {
                continue;
            }
            std::string msg = "  well " + w.name() + ": "
                + std::to_string(cs.size()) + " connection(s)";
            if (w.is_lgr_well()) {
                msg += " in LGR " + w.get_lgr_well_tag().value();
            }
            for (const auto& c : cs) {
                msg += "\n    (" + std::to_string(c.getI() + 1) + ","
                    + std::to_string(c.getJ() + 1) + ","
                    + std::to_string(c.getK() + 1) + ") lgr_grid "
                    + std::to_string(c.get_lgr_level());
            }
            OpmLog::info(msg);
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
