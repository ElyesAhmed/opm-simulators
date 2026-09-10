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

        const unsigned marked = this->simulator_.problem().markForGridAdaptation();
        if (marked == 0) {
            return;
        }

        throw std::runtime_error(
            "FvBaseDiscretizationAdaptiveNative::adaptGrid: the problem marked "
            + std::to_string(marked) + " cell(s) for adaptation, but the native "
            "execution path (mesh adapt + conservative state transfer) is not "
            "yet implemented.");
    }
};

} // namespace Opm

#endif // EWOMS_FV_BASE_DISCRETIZATION_ADAPTIVE_NATIVE_HH
