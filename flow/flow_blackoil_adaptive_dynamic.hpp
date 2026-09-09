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
#ifndef FLOW_BLACKOIL_ADAPTIVE_DYNAMIC_HPP
#define FLOW_BLACKOIL_ADAPTIVE_DYNAMIC_HPP

namespace Opm {

//! \brief Standalone main of the flow_blackoil_adaptive_dynamic binary.
//!
//! Drives the simulation one report step at a time through the FlowMain
//! step API (executeInitStep / executeStep / executeStepsCleanup) instead of
//! the monolithic execute(). The step loop is the seam where dynamic grid
//! adaptation (docs/DYNAMIC-REFINEMENT-FLOW-PLAN.md, phase 1) will rebuild
//! the simulator between report steps. Without adaptation the run is
//! equivalent to flow_blackoil_adaptive.
int flowBlackoilTpfaAdaptiveDynamicMainStandalone(int argc, char** argv);

}

#endif // FLOW_BLACKOIL_ADAPTIVE_DYNAMIC_HPP
