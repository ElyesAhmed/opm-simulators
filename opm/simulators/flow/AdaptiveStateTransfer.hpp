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
 * \brief State extract / remap / inject across a mid-run simulator rebuild
 *        (dynamic grid refinement, phase 1 -- steps S3-S5 of
 *        opm-gridrefined docs/DYNAMIC-REFINEMENT-FLOW-PLAN.md).
 *
 * The per-cell state is keyed by CpGridData::stableCellId(): the plain
 * Cartesian index for coarse cells and the packed (parent Cartesian, child
 * lattice index) for refined cells -- invariant across grid rebuilds, so the
 * same map works for an unchanged grid (identity), for refinement (child
 * looks up its parent's id: constant prolongation) and for coarsening
 * (parent reduces over its children's entries).
 *
 * Injection reuses the ECL restart machinery end-to-end: fill the output
 * module's restart buffers via setRestart() from an in-memory data::Solution,
 * then let FlowProblemBlackoil::readSolutionFromOutputModule() convert the
 * per-cell field arrays into primary variables (including the
 * switching-variable logic) exactly as a file restart would.
 */
#ifndef OPM_ADAPTIVE_STATE_TRANSFER_HPP
#define OPM_ADAPTIVE_STATE_TRANSFER_HPP

#include <opm/output/data/Cells.hpp>
#include <opm/input/eclipse/Units/UnitSystem.hpp>

#include <opm/material/common/MathToolbox.hpp>
#include <opm/material/fluidstates/BlackOilFluidState.hpp>

#include <opm/models/utils/propertysystem.hh>
#include <opm/models/utils/basicproperties.hh>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace Opm {

//! Per-cell transferable state (the ECL restart field set, SI units).
struct AdaptiveCellState
{
    double pressure{};   //!< oil-phase (reference) pressure
    double swat{};
    double sgas{};
    double rs{};
    double rv{};
    double temperature{};
};

using AdaptiveStateMap = std::unordered_map<std::int64_t, AdaptiveCellState>;

//! Extract the transferable state of every leaf cell, keyed by stableCellId.
//! Call with an explicit TypeTag: extractAdaptiveState<TypeTag>(sim).
template <class TypeTag>
AdaptiveStateMap
extractAdaptiveState(GetPropType<TypeTag, Properties::Simulator>& simulator)
{
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;

    const auto& grid = simulator.vanguard().grid();
    const auto stableIds = grid.currentData().back()->stableCellId();

    AdaptiveStateMap state;
    state.reserve(stableIds.size());

    ElementContext elemCtx(simulator);
    const auto& gridView = simulator.gridView();
    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
        elemCtx.updatePrimaryStencil(elem);
        elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);
        const unsigned elemIdx = elemCtx.globalSpaceIndex(0, /*timeIdx=*/0);
        const auto& fs = elemCtx.intensiveQuantities(0, /*timeIdx=*/0).fluidState();

        AdaptiveCellState cs;
        if (FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)) {
            cs.pressure = getValue(fs.pressure(FluidSystem::oilPhaseIdx));
        } else if (FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)) {
            cs.pressure = getValue(fs.pressure(FluidSystem::gasPhaseIdx));
        } else {
            cs.pressure = getValue(fs.pressure(FluidSystem::waterPhaseIdx));
        }
        if (FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
            cs.swat = getValue(fs.saturation(FluidSystem::waterPhaseIdx));
        }
        if (FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)) {
            cs.sgas = getValue(fs.saturation(FluidSystem::gasPhaseIdx));
        }
        if (FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)
            && FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)) {
            cs.rs = getValue(fs.Rs());
            cs.rv = getValue(fs.Rv());
        }
        cs.temperature = getValue(fs.temperature(0));

        state.emplace(stableIds[elemIdx], cs);
    }
    return state;
}

//! Remap an extracted state onto the (new) grid of @p simulator, leaf-ordered.
//! Phase 1: exact-id copy only (identity on an unchanged grid); refinement /
//! coarsening reductions are the next increment (plan S4). Throws if a cell
//! has no source entry.
template <class TypeTag>
data::Solution
remapAdaptiveState(const AdaptiveStateMap& state,
                   GetPropType<TypeTag, Properties::Simulator>& simulator)
{
    const auto& grid = simulator.vanguard().grid();
    const auto stableIds = grid.currentData().back()->stableCellId();
    const std::size_t n = stableIds.size();

    std::vector<double> pressure(n), swat(n), sgas(n), rs(n), rv(n), temp(n);
    for (std::size_t c = 0; c < n; ++c) {
        const auto it = state.find(stableIds[c]);
        if (it == state.end()) {
            throw std::logic_error("adaptive state transfer: no source state for cell "
                                   + std::to_string(c) + " (stable id "
                                   + std::to_string(stableIds[c]) + ")");
        }
        const auto& cs = it->second;
        pressure[c] = cs.pressure;
        swat[c] = cs.swat;
        sgas[c] = cs.sgas;
        rs[c] = cs.rs;
        rv[c] = cs.rv;
        temp[c] = cs.temperature;
    }

    // The extracted values are SI (straight from the fluid state).
    data::Solution sol(/*si=*/true);
    using M = UnitSystem::measure;
    using T = data::TargetType;
    sol.insert("PRESSURE", M::pressure,          std::move(pressure), T::RESTART_SOLUTION);
    sol.insert("SWAT",     M::identity,          std::move(swat),     T::RESTART_SOLUTION);
    sol.insert("SGAS",     M::identity,          std::move(sgas),     T::RESTART_SOLUTION);
    sol.insert("RS",       M::gas_oil_ratio,     std::move(rs),       T::RESTART_SOLUTION);
    sol.insert("RV",       M::oil_gas_ratio,     std::move(rv),       T::RESTART_SOLUTION);
    sol.insert("TEMP",     M::temperature,       std::move(temp),     T::RESTART_SOLUTION);
    return sol;
}

//! Inject a leaf-ordered solution into a freshly initialized simulator at
//! report step @p step: position clock/episode, fill the restart buffers, run
//! the restart array->primary-variable assignment, and refresh the model
//! caches that a completed init has already built.
template <class TypeTag>
void injectAdaptiveState(GetPropType<TypeTag, Properties::Simulator>& simulator,
                         const data::Solution& sol, const int step)
{
    auto& problem = simulator.problem();
    const auto& schedule = simulator.vanguard().schedule();

    // Mirror readEclRestartSolution_'s clock/episode positioning.
    simulator.setTime(schedule.seconds(step));
    simulator.startNextEpisode(simulator.startTime() + simulator.time(),
                               schedule.stepLength(step));
    simulator.setEpisodeIndex(step);

    // Restart buffers sized for this leaf, then per-cell injection. The
    // data::Solution is leaf-ordered, so local index == lookup index.
    auto& outputModule = problem.eclWriter().mutableOutputModule();
    const auto numElements = simulator.model().numGridDof();
    outputModule.allocBuffers(numElements, step,
                              /*isSubStep=*/false, /*log=*/false, /*isRestart=*/true);
    for (std::size_t elemIdx = 0; elemIdx < numElements; ++elemIdx) {
        outputModule.setRestart(sol, elemIdx, elemIdx);
    }

    // Arrays -> initialFluidStates_ -> PrimaryVariables (switching logic
    // included), written into model().solution(0).
    problem.readSolutionFromOutputModule(step, false);

    // A completed init also has history and caches; refresh them.
    simulator.model().solution(/*timeIdx=*/1) = simulator.model().solution(/*timeIdx=*/0);
    simulator.model().invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/0);
}

} // namespace Opm

#endif // OPM_ADAPTIVE_STATE_TRANSFER_HPP
