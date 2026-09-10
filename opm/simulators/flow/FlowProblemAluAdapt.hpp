// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
/*!
 * \file
 * \copydoc Opm::FlowProblemAluAdapt
 */
#ifndef OPM_FLOW_PROBLEM_ALU_ADAPT_HPP
#define OPM_FLOW_PROBLEM_ALU_ADAPT_HPP

#include <opm/simulators/flow/FlowProblemBlackoil.hpp>

namespace Opm {

/*!
 * \brief Black-oil Flow problem for native (non-dune-fem) in-place h-adaptivity
 *        on a locally-adaptive grid (dune-ALUGrid).
 *
 * Adds the adaptation hooks that FvBaseDiscretizationAdaptiveNative calls, on top
 * of FlowProblemBlackoil:
 *   - markForGridAdaptation(): the plug point for the a posteriori
 *     SpatialMarker. Until that lands (Phase 5) it returns 0, so
 *     --enable-grid-adaptation is a no-op plumbing path. It DELIBERATELY shadows
 *     MultiPhaseBaseProblem::markForGridAdaptation() (a saturation-variation
 *     heuristic that calls grid.mark() inside a per-phase loop and miscounts) --
 *     that heuristic returns later behind --adapt-indicator=saturation.
 *   - gridChanged(): rebuild problem-side geometry-dependent data after an
 *     in-place adapt. Phase 0: base behaviour only.
 */
template <class TypeTag>
class FlowProblemAluAdapt : public FlowProblemBlackoil<TypeTag>
{
    using Base = FlowProblemBlackoil<TypeTag>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;

public:
    explicit FlowProblemAluAdapt(Simulator& simulator)
        : Base(simulator)
    {}

    /*!
     * \brief Number of leaf cells marked for refinement/coarsening this step.
     *
     * PHASE 0: always 0 (adaptation execution not implemented). PHASE 5 will
     * dispatch to the estimator-driven SpatialMarker (default) or the
     * saturation-variation heuristic (--adapt-indicator=saturation).
     */
    unsigned markForGridAdaptation()
    {
        return 0;
    }

    /*!
     * \brief Rebuild problem-side geometry-dependent data after an in-place
     *        grid adaptation.
     *
     * PHASE 1+: transmissibility, well connection->cell map + WI, cell
     * depths/thickness, threshold pressure, output projection map.
     */
    void gridChanged()
    {
        Base::gridChanged();
    }
};

} // namespace Opm

#endif // OPM_FLOW_PROBLEM_ALU_ADAPT_HPP
