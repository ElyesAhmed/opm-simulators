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
#include "config.h"

#include <flow/flow_blackoil_adaptive_dynamic.hpp>

#include <opm/material/common/ResetLocale.hpp>
#include <opm/grid/CpGrid.hpp>
#include <opm/simulators/flow/SimulatorFullyImplicit.hpp>
#include <opm/simulators/flow/FlowMain.hpp>
#include <opm/simulators/flow/Main.hpp>
#include <opm/simulators/flow/AdaptiveCpGridVanguard.hpp>
#include <opm/simulators/flow/python/PyMain.hpp>

#include <opm/models/blackoil/blackoilconvectivemixingmodule.hh>
#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>

#include <cstdlib>
#include <memory>

namespace Opm::Properties {

namespace TTag {
// Same model configuration as FlowProblemAdaptive (flow_blackoil_adaptive.cpp):
// TPFA blackoil with the AdaptiveCpGridVanguard. A separate TypeTag so the two
// executables stay independent while the dynamic driver evolves.
struct FlowProblemAdaptiveDynamic
{ using InheritsFrom = std::tuple<FlowProblem>; };
}

template<class TypeTag>
struct Linearizer<TypeTag, TTag::FlowProblemAdaptiveDynamic>
{ using type = TpfaLinearizer<TypeTag>; };

template<class TypeTag>
struct LocalResidual<TypeTag, TTag::FlowProblemAdaptiveDynamic>
{ using type = BlackOilLocalResidualTPFA<TypeTag>; };

template<class TypeTag>
struct EnableDiffusion<TypeTag, TTag::FlowProblemAdaptiveDynamic>
{ static constexpr bool value = false; };

template<class TypeTag>
struct AvoidElementContext<TypeTag, TTag::FlowProblemAdaptiveDynamic>
{ static constexpr bool value = true; };

template<class TypeTag>
struct Vanguard<TypeTag, TTag::FlowProblemAdaptiveDynamic>
{ using type = AdaptiveCpGridVanguard<TypeTag>; };

} // namespace Opm::Properties

namespace Opm {

// ----------------- Main program: the report-step driver -----------------
//
// Instead of FlowMain::execute() (which owns the whole time loop), drive the
// run one report step at a time through the step API the Python bindings use:
//   initFlowBlackoil() -> executeInitStep() -> executeStep()* -> cleanup.
// PyMain is a plain-C++ Main subclass (no Python dependency) that splits
// initialization from the run loop; we reuse it as the bootstrap.
//
// The while-loop below is the adaptation seam of the dynamic-refinement plan
// (opm-gridrefined docs/DYNAMIC-REFINEMENT-FLOW-PLAN.md, phase 1 step S2):
// between two executeStep() calls the simulator can be torn down, the grid
// re-refined to a new mark set, and the remapped state injected -- everything
// this skeleton runs is already structured for that insertion.
int flowBlackoilTpfaAdaptiveDynamicMainStandalone(int argc, char** argv)
{
    using TypeTag = Properties::TTag::FlowProblemAdaptiveDynamic;

    // we always want to use the default locale, and thus spare us the trouble
    // with incorrect locale settings.
    resetLocale();

    // Main's ctor/dtor own MPI_Init/MPI_Finalize.
    auto mainObject = std::make_unique<PyMain<TypeTag>>(argc, argv);

    int exitCode = EXIT_SUCCESS;
    auto flowMain = mainObject->initFlowBlackoil(exitCode);
    if (!flowMain) {
        mainObject.reset();
        return exitCode;
    }

    int status = flowMain->executeInitStep();
    if (status == EXIT_SUCCESS) {
        // One executeStep() per report step. runStep returns a continue flag
        // (false = schedule EXIT), not an exit status; errors throw.
        bool continueLooping = true;
        while (continueLooping && !flowMain->getSimTimer()->done()) {
            continueLooping = (flowMain->executeStep() != 0);
        }
        status = flowMain->executeStepsCleanup();
    }

    flowMain.reset();
    mainObject.reset();   // destructor calls MPI_Finalize
    return status;
}

} // namespace Opm
