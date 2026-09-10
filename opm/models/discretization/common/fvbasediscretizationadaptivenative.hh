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
 * \copydoc Opm::FvBaseDiscretizationAdaptiveNative
 */
#ifndef EWOMS_FV_BASE_DISCRETIZATION_ADAPTIVE_NATIVE_HH
#define EWOMS_FV_BASE_DISCRETIZATION_ADAPTIVE_NATIVE_HH

#include <opm/models/discretization/common/fvbasediscretization.hh>

#include <opm/common/OpmLog/OpmLog.hpp>

#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/common/partitionset.hh>

#include <fmt/format.h>

#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace Opm {

/*!
 * \ingroup FiniteVolumeDiscretizations
 *
 * \brief Finite-volume discretization with NATIVE (no dune-fem) local grid
 *        adaptation.
 *
 * Keeps Flow's BlockVectorWrapper solution container and does NOT pull in
 * dune-fem. adaptGrid() (called once per accepted substep from
 * FvBaseDiscretization::advanceTimeLevel(), the Algorithm 6.1 seam):
 *   1. problem().markForGridAdaptation()  -- marks leaf fathers (grid.mark)
 *   2. problem().prepareForAdapt()        -- inventory + vanguard Cartesian snapshot
 *   3. snapshot solution(0) keyed by persistent local id
 *   4. grid.preAdapt() / adapt() / postAdapt()
 *   5. vanguard().rebuildAfterAdapt()     -- Cartesian ids for new cells, leaf view
 *   6. resize + prolong-by-injection solution(0); solution(1) := solution(0)
 *   7. finishInit() / resetLinearizer()   -- volumes, caches, matrix
 *   8. problem().gridChanged()            -- transmissibility, porosity, ...
 *   9. problem().verifyAfterAdapt()       -- component-inventory conservation
 */
template <class TypeTag>
class FvBaseDiscretizationAdaptiveNative : public FvBaseDiscretization<TypeTag>
{
    using ParentType = FvBaseDiscretization<TypeTag>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using DiscreteFunction = GetPropType<TypeTag, Properties::DiscreteFunction>;
    using PrimaryVariables = GetPropType<TypeTag, Properties::PrimaryVariables>;
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using Grid = GetPropType<TypeTag, Properties::Grid>;
    using IdType = typename Grid::LocalIdSet::IdType;

    static constexpr unsigned historySize =
        getPropValue<TypeTag, Properties::TimeDiscHistorySize>();

public:
    template<class Serializer>
    struct SerializeHelper
    {
        template<class SolutionType>
        static void serializeOp(Serializer& serializer, SolutionType& solution)
        {
            for (auto& sol : solution) {
                serializer(*sol);
            }
        }
    };

    explicit FvBaseDiscretizationAdaptiveNative(Simulator& simulator)
        : ParentType(simulator)
    {
        const std::size_t numDof = this->asImp_().numGridDof();
        for (unsigned timeIdx = 0; timeIdx < historySize; ++timeIdx) {
            this->solution_[timeIdx] =
                std::make_unique<DiscreteFunction>("solution", numDof);
        }
    }

    void adaptGrid()
    {
        if (!this->enableGridAdaptation_) {
            return;
        }

        auto& problem = this->simulator_.problem();

        const unsigned marked = problem.markForGridAdaptation();
        if (marked == 0) {
            return;
        }

        auto& grid = this->simulator_.vanguard().grid();
        const std::size_t nBefore = this->gridView_.size(/*codim=*/0);

        // --- (2) pre-adapt: inventory + Cartesian-id snapshot -----------------
        problem.prepareForAdapt();

        // --- (3) snapshot solution(0) by persistent local id -----------------
        std::unordered_map<IdType, PrimaryVariables> sol0;
        sol0.reserve(nBefore);
        {
            const auto& idSet = grid.localIdSet();
            Dune::MultipleCodimMultipleGeomTypeMapper<GridView>
                em(this->gridView_, Dune::mcmgElementLayout());
            const auto& bv = this->solution_[0]->blockVector();
            for (const auto& e : elements(this->gridView_, Dune::Partitions::interior)) {
                sol0.emplace(idSet.id(e), bv[em.index(e)]);
            }
        }

        // --- (4) adapt in place ---------------------------------------------
        const bool preOk = grid.preAdapt();
        const bool changed = grid.adapt();
        grid.postAdapt();

        // --- (5) rebuild the vanguard's Cartesian machinery -----------------
        // The vanguard leaf grid view is NOT recreated (ALU tracks it live).
        this->simulator_.vanguard().rebuildAfterAdapt();
        // Reconstruct the discretization's mappers -- NOT .update(), which
        // copy-assigns the GridView and leaves stale ALU iterator internals.
        // (MultipleCodim...Mapper::operator= is deleted -> placement-new.)
        using ElementMapper = std::decay_t<decltype(this->elementMapper_)>;
        using VertexMapper = std::decay_t<decltype(this->vertexMapper_)>;
        this->elementMapper_.~ElementMapper();
        ::new (static_cast<void*>(&this->elementMapper_))
            ElementMapper(this->gridView_, Dune::mcmgElementLayout());
        this->vertexMapper_.~VertexMapper();
        ::new (static_cast<void*>(&this->vertexMapper_))
            VertexMapper(this->gridView_, Dune::mcmgVertexLayout());

        const std::size_t nAfter = this->asImp_().numGridDof();

        // --- (6) resize + prolong-by-injection solution(0), then (1):=(0) ----
        {
            const auto& idSet = grid.localIdSet();
            Dune::MultipleCodimMultipleGeomTypeMapper<GridView>
                em(this->gridView_, Dune::mcmgElementLayout());
            for (unsigned t = 0; t < historySize; ++t) {
                this->solution_[t]->blockVector().resize(nAfter);
            }
            auto& bv0 = this->solution_[0]->blockVector();
            for (const auto& e : elements(this->gridView_, Dune::Partitions::interior)) {
                auto a = e;
                const PrimaryVariables* v = nullptr;
                while (true) {
                    auto it = sol0.find(idSet.id(a));
                    if (it != sol0.end()) { v = &it->second; break; }
                    if (a.level() == 0 || !a.hasFather()) { break; }
                    a = a.father();
                }
                if (v == nullptr) {
                    throw std::runtime_error("FvBaseDiscretizationAdaptiveNative::"
                        "adaptGrid: no pre-adapt ancestor for a leaf cell.");
                }
                bv0[em.index(e)] = *v;
            }
            // advanceTimeLevel() sets solution(1) = solution(0) right after this,
            // but keep them consistent here too.
            for (unsigned t = 1; t < historySize; ++t) {
                this->solution_[t]->blockVector() = bv0;
            }
        }


        // --- (7) rebuild discretization geometry/caches/matrix --------------
        this->resetLinearizer();
        this->finishInit();

        // --- (8) rebuild problem-side geometry-dependent data ---------------
        problem.gridChanged();

        // --- recompute intensive quantities from the transferred state ------
        this->invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/0);

        // explicit per-cell state (max sat, rock-compaction mult) that
        // beginTimeStep normally maintains -- needs valid intensive quantities.
        problem.finishAdaptExplicitQuantities();
        this->invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/0);

        for (auto& module : this->outputModules_) {
            module->allocBuffers();
        }

        // --- (9) conservation gate ----------------------------------------
        problem.verifyAfterAdapt();

        OpmLog::info(fmt::format(
            "[alu-hadapt] in-place adapt: marked={} preAdapt={} changed={}  "
            "leaf cells {} -> {}", marked, preOk, changed, nBefore, nAfter));
    }
};

} // namespace Opm

#endif // EWOMS_FV_BASE_DISCRETIZATION_ADAPTIVE_NATIVE_HH
