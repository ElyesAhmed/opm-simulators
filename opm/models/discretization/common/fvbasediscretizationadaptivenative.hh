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

#include <fmt/format.h>

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace Opm {

/*!
 * \ingroup FiniteVolumeDiscretizations
 *
 * \brief Finite-volume discretization with NATIVE (no dune-fem) local grid
 *        adaptation.
 *
 * Unlike FvBaseDiscretizationFemAdapt this keeps Flow's own BlockVectorWrapper
 * solution container and does NOT pull in dune-fem / Dune::Fem::AdaptationManager.
 * The whole mark -> adapt -> conservative-transfer -> rebuild sequence is
 * delegated to the problem (Problem::adaptGrid()), so the physical state
 * transfer stays application code with explicit conservation checks rather than
 * a dune-fem RestrictProlong default (which averages primary variables and is
 * not inventory-conservative on coarsening).
 *
 * The base FvBaseDiscretization::advanceTimeLevel() calls asImp_().adaptGrid()
 * once per accepted substep (via NonlinearSystem::prepareStep), which is the
 * Algorithm 6.1 adaptation seam.
 */
template <class TypeTag>
class FvBaseDiscretizationAdaptiveNative : public FvBaseDiscretization<TypeTag>
{
    using ParentType = FvBaseDiscretization<TypeTag>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using DiscreteFunction = GetPropType<TypeTag, Properties::DiscreteFunction>;

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
        // NOTE: FvBaseDiscretizationNoAdapt throws here when
        // enableGridAdaptation_ is set; this class exists precisely to allow it
        // with a native (non-fem) execution path, so no throw.
        const std::size_t numDof = this->asImp_().numGridDof();
        for (unsigned timeIdx = 0; timeIdx < historySize; ++timeIdx) {
            this->solution_[timeIdx] =
                std::make_unique<DiscreteFunction>("solution", numDof);
        }
    }

    /*!
     * \brief Adapt the grid in place, once per accepted substep.
     *
     * PHASE 0 (this commit): if the problem marks nothing (the default), this is
     * a pure no-op, so --enable-grid-adaptation=true runs identically to the
     * non-adaptive path. If cells ARE marked it throws, because the execution
     * path (native mesh adapt + conservative state transfer + rebuild) is not
     * yet implemented.
     *
     * PHASE 1+ will replace the throw with:
     *   grid.mark(); grid.preAdapt(); grid.adapt(); grid.postAdapt();
     *   conservative prolong/restrict of {p,Sw,Sg,Rs,Rv,T} + histories;
     *   rebuild mappers / transmissibility / wells / linearizer / buffers.
     */
    void adaptGrid()
    {
        if (!this->enableGridAdaptation_) {
            return;
        }

        // The problem marks leaf entities (grid.mark(+/-1, e)) and returns the
        // count. 0 => nothing to do.
        const unsigned marked = this->simulator_.problem().markForGridAdaptation();
        if (marked == 0) {
            return;
        }

        auto& grid = this->simulator_.vanguard().grid();
        const std::size_t nBefore = this->gridView_.size(/*codim=*/0);

        // PHASE 1: exercise the ALUGrid in-place adaptation lifecycle and
        // confirm the leaf view changes. Conservative state transfer is NOT
        // done yet, so the solution vector would be stale -- stop cleanly with
        // evidence rather than solve on garbage.
        const bool preOk = grid.preAdapt();
        const bool changed = grid.adapt();
        grid.postAdapt();

        this->gridView_ = this->simulator_.gridView();
        const std::size_t nAfter = this->gridView_.size(/*codim=*/0);

        OpmLog::info(fmt::format(
            "[alu-hadapt PHASE 1] marked={}  preAdapt={}  adapt-changed={}  "
            "leaf cells {} -> {}",
            marked, preOk, changed, nBefore, nAfter));

        throw std::runtime_error(fmt::format(
            "FvBaseDiscretizationAdaptiveNative::adaptGrid PHASE 1: grid adapted "
            "in place ({} -> {} leaf cells), conservative state transfer not yet "
            "implemented -- stopping.", nBefore, nAfter));
    }
};

} // namespace Opm

#endif // EWOMS_FV_BASE_DISCRETIZATION_ADAPTIVE_NATIVE_HH
