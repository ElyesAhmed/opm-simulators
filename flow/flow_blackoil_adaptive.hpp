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
#ifndef FLOW_BLACKOIL_ADAPTIVE_HPP
#define FLOW_BLACKOIL_ADAPTIVE_HPP

#include <opm/simulators/flow/TTagFlowProblemTPFA.hpp>

#include <memory>

namespace Opm {

namespace Properties { namespace TTag { struct FlowProblemAdaptive; } }

//! \brief Main function used in the flow_blackoil_adaptive binary.
int flowBlackoilTpfaAdaptiveMain(int argc, char** argv, bool outputCout, bool outputFiles);

template<class TypeTag> class FlowMain;

//! \brief Initialization function used in the flow_blackoil_adaptive binary.
std::unique_ptr<FlowMain<Properties::TTag::FlowProblemAdaptive>>
flowBlackoilTpfaAdaptiveMainInit(int argc, char** argv, bool outputCout, bool outputFiles);

//! \brief Standalone main used in the flow_blackoil_adaptive binary.
int flowBlackoilTpfaAdaptiveMainStandalone(int argc, char** argv);

}

#endif // FLOW_BLACKOIL_ADAPTIVE_HPP
