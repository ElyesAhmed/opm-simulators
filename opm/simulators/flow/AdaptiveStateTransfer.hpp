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

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/material/common/MathToolbox.hpp>
#include <opm/material/fluidstates/BlackOilFluidState.hpp>

#include <opm/models/utils/propertysystem.hh>
#include <opm/models/utils/basicproperties.hh>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

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

    // stableCellId packing (CpGridData::stableCellId, design D3): refined cell
    // = refinedTag | parentCart<<childBits | childIdx; coarse cell = plain
    // Cartesian index.
    constexpr int childBits = 20;
    constexpr std::int64_t refinedTag = std::int64_t(1) << 62;

    // Restriction buckets: plain average of the old refined children per
    // parent Cartesian index (phase 1; pv-weighted avg and max/min ops are the
    // next increment).
    std::unordered_map<std::int64_t, std::pair<AdaptiveCellState, int>> parentAvg;
    for (const auto& [id, cs] : state) {
        if (id & refinedTag) {
            auto& [acc, cnt] = parentAvg[(id & ~refinedTag) >> childBits];
            acc.pressure += cs.pressure; acc.swat += cs.swat; acc.sgas += cs.sgas;
            acc.rs += cs.rs; acc.rv += cs.rv; acc.temperature += cs.temperature;
            ++cnt;
        }
    }

    std::vector<double> pressure(n), swat(n), sgas(n), rs(n), rv(n), temp(n);
    for (std::size_t c = 0; c < n; ++c) {
        const std::int64_t id = stableIds[c];
        AdaptiveCellState cs;
        if (const auto it = state.find(id); it != state.end()) {
            cs = it->second;                                  // exact match
        } else if ((id & refinedTag)
                   && state.count((id & ~refinedTag) >> childBits)) {
            cs = state.at((id & ~refinedTag) >> childBits);   // prolong: parent value
        } else if (const auto pa = parentAvg.find(id); pa != parentAvg.end()) {
            cs = pa->second.first;                            // restrict: child average
            const double inv = 1.0 / pa->second.second;
            cs.pressure *= inv; cs.swat *= inv; cs.sgas *= inv;
            cs.rs *= inv; cs.rv *= inv; cs.temperature *= inv;
        } else {
            throw std::logic_error("adaptive state transfer: no source state for cell "
                                   + std::to_string(c) + " (stable id "
                                   + std::to_string(id) + ")");
        }
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

//! {water, oil, gas} total component inventory in SURFACE volume over the
//! interior leaf cells -- the black-oil accumulation term
//!   sum_K PV_K * [ S_w b_w,  S_o b_o + R_v S_g b_g,  S_g b_g + R_s S_o b_o ]
//! i.e. exactly the quantity the scheme conserves. Constant intensive
//! prolongation to child cells preserves this iff the child pore volumes
//! partition the parent's, so comparing it before/after a rebuild is the
//! direct conservativeness check (plan step: strict per-component inventory).
template <class TypeTag>
std::array<double, 3>
blackOilComponentInventory(GetPropType<TypeTag, Properties::Simulator>& simulator)
{
    using FluidSystem    = GetPropType<TypeTag, Properties::FluidSystem>;
    using ElementContext  = GetPropType<TypeTag, Properties::ElementContext>;

    constexpr int waterPhaseIdx = FluidSystem::waterPhaseIdx;
    constexpr int oilPhaseIdx   = FluidSystem::oilPhaseIdx;
    constexpr int gasPhaseIdx   = FluidSystem::gasPhaseIdx;
    const bool oilActive = FluidSystem::phaseIsActive(oilPhaseIdx);
    const bool gasActive = FluidSystem::phaseIsActive(gasPhaseIdx);
    const bool watActive = FluidSystem::phaseIsActive(waterPhaseIdx);

    std::array<double, 3> inv{0.0, 0.0, 0.0};   // 0 = water, 1 = oil, 2 = gas

    ElementContext elemCtx(simulator);
    for (const auto& elem : elements(simulator.gridView(), Dune::Partitions::interior)) {
        elemCtx.updatePrimaryStencil(elem);
        elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);
        const auto& iq  = elemCtx.intensiveQuantities(0, /*timeIdx=*/0);
        const unsigned idx = elemCtx.globalSpaceIndex(0, /*timeIdx=*/0);
        const auto& fs = iq.fluidState();

        const double pv = simulator.model().dofTotalVolume(idx)
                        * getValue(iq.porosity());

        double svW = 0.0, svO = 0.0, svG = 0.0;
        if (watActive)
            svW = getValue(fs.saturation(waterPhaseIdx)) * getValue(fs.invB(waterPhaseIdx)) * pv;
        if (oilActive)
            svO = getValue(fs.saturation(oilPhaseIdx))   * getValue(fs.invB(oilPhaseIdx))   * pv;
        if (gasActive)
            svG = getValue(fs.saturation(gasPhaseIdx))   * getValue(fs.invB(gasPhaseIdx))   * pv;

        inv[0] += svW;
        inv[1] += svO;
        inv[2] += svG;
        if (oilActive && gasActive) {
            inv[2] += getValue(fs.Rs()) * svO;   // dissolved gas in oil
            inv[1] += getValue(fs.Rv()) * svG;   // vaporized oil in gas
        }
    }

    const auto& comm = simulator.gridView().comm();
    for (auto& v : inv) v = comm.sum(v);
    return inv;
}

//! Per-ORIGINAL-PARENT black-oil inventory: parentCartesian -> {PV, water,
//! oil, gas} (surface volume). Each leaf cell is bucketed by its stableCellId
//! -- a coarse cell into its own Cartesian index, a refined child into its
//! parent's -- so comparing the old coarse map against the new refined map
//! detects opposite local errors that a single global sum would hide
//! (review 2026-09-10). Serial only.
template <class TypeTag>
std::unordered_map<std::int64_t, std::array<double, 4>>
blackOilInventoryByParent(GetPropType<TypeTag, Properties::Simulator>& simulator)
{
    using FluidSystem   = GetPropType<TypeTag, Properties::FluidSystem>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
    constexpr int wP = FluidSystem::waterPhaseIdx;
    constexpr int oP = FluidSystem::oilPhaseIdx;
    constexpr int gP = FluidSystem::gasPhaseIdx;
    const bool wA = FluidSystem::phaseIsActive(wP);
    const bool oA = FluidSystem::phaseIsActive(oP);
    const bool gA = FluidSystem::phaseIsActive(gP);

    constexpr int childBits = 20;
    constexpr std::int64_t refinedTag = std::int64_t(1) << 62;
    const auto& grid = simulator.vanguard().grid();
    const auto stableIds = grid.currentData().back()->stableCellId();

    // Level-zero Cartesian extent: a coarse cell's stableCellId is its plain
    // Cartesian index, and a refined child's decoded parent must land in the
    // same range. Anything outside means the id packing is not what we assume
    // -- fail rather than bucket into a bogus parent (review 2026-09-10:
    // the old "idx >= size() -> use compressed index" fallback silently
    // fabricated parent keys).
    const auto& cartDims =
        simulator.vanguard().cartesianIndexMapper().cartesianDimensions();
    const std::int64_t nCart = static_cast<std::int64_t>(cartDims[0])
                             * cartDims[1] * cartDims[2];

    std::unordered_map<std::int64_t, std::array<double, 4>> out;
    ElementContext elemCtx(simulator);
    for (const auto& elem : elements(simulator.gridView(), Dune::Partitions::interior)) {
        elemCtx.updatePrimaryStencil(elem);
        elemCtx.updatePrimaryIntensiveQuantities(0);
        const auto& iq  = elemCtx.intensiveQuantities(0, 0);
        const unsigned idx = elemCtx.globalSpaceIndex(0, 0);
        const auto& fs = iq.fluidState();

        if (idx >= stableIds.size()) {
            throw std::runtime_error(fmt::format(
                "blackOilInventoryByParent: leaf cell {} has no stableCellId "
                "(id array holds {}). Per-parent conservation cannot be checked.",
                idx, stableIds.size()));
        }
        const std::int64_t id = stableIds[idx];
        std::int64_t key;
        if (id & refinedTag) {
            key = (id & ~refinedTag) >> childBits;   // decoded parent Cartesian
        } else {
            key = id;                                // coarse: own Cartesian
        }
        if (key < 0 || key >= nCart) {
            throw std::runtime_error(fmt::format(
                "blackOilInventoryByParent: leaf cell {} stableCellId {} decodes "
                "to parent Cartesian {} outside [0,{}). Unexpected id packing.",
                idx, id, key, nCart));
        }

        const double pv = simulator.model().dofTotalVolume(idx) * getValue(iq.porosity());
        double svW = 0, svO = 0, svG = 0;
        if (wA) svW = getValue(fs.saturation(wP)) * getValue(fs.invB(wP)) * pv;
        if (oA) svO = getValue(fs.saturation(oP)) * getValue(fs.invB(oP)) * pv;
        if (gA) svG = getValue(fs.saturation(gP)) * getValue(fs.invB(gP)) * pv;
        auto& a = out[key];
        a[0] += pv;
        a[1] += svW;
        a[2] += svO + (oA && gA ? getValue(fs.Rv()) * svG : 0.0);
        a[3] += svG + (oA && gA ? getValue(fs.Rs()) * svO : 0.0);
    }
    return out;
}

//! Compare per-parent inventories. Only parents present in BOTH maps are
//! checked (a refined parent must conserve PV and every component). Logs the
//! worst offenders; returns true iff all within \p tol (relative).
inline bool
verifyPerParentConservation(
    const std::unordered_map<std::int64_t, std::array<double, 4>>& before,
    const std::unordered_map<std::int64_t, std::array<double, 4>>& after,
    double tol, bool throwOnFail = false)
{
    // Refinement-only path: the parent-key SETS must be identical (every
    // original coarse cell is a parent; a child's parent must appear).
    // Non-finite / negative inventory, a non-finite / non-positive tolerance,
    // a missing OR an extra parent all FAIL -- the earlier version skipped a
    // missing parent, ignored extras, and let NaN through (reviews 2026-09-10).
    static constexpr const char* nm[4] = {"PV", "water", "oil", "gas"};
    bool ok = std::isfinite(tol) && tol > 0.0;
    if (!ok)
        OpmLog::error("per-parent conservation: non-finite / non-positive tolerance");

    int nMissing = 0, nExtra = 0;
    for (const auto& [key, b] : before)
        if (!after.count(key)) ++nMissing;
    for (const auto& [key, a] : after) {
        static_cast<void>(a);
        if (!before.count(key)) ++nExtra;
    }
    if (nMissing || nExtra) ok = false;

    int nBad = 0, nChecked = 0, nNonFinite = 0;
    std::vector<std::string> invalidDetails;
    std::array<double, 4> worstRel{0, 0, 0, 0}, worstAbs{0, 0, 0, 0};
    std::array<double, 4> sumSigned{0, 0, 0, 0}, sumAbs{0, 0, 0, 0};
    std::array<std::int64_t, 4> worstRelKey{-1, -1, -1, -1};
    std::array<double, 4> worstRelScale{0, 0, 0, 0};
    for (const auto& [key, b] : before) {
        const auto it = after.find(key);
        if (it == after.end()) continue;
        ++nChecked;
        const auto& a = it->second;
        bool bad = false;
        for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(a[c]) || !std::isfinite(b[c])
                || a[c] < -1e-12 || b[c] < -1e-12) {
                ++nNonFinite;
                bad = true;
                if (invalidDetails.size() < 8)
                    invalidDetails.push_back(fmt::format(
                        "  invalid {} @cart {}: before={:.17g}, after={:.17g}",
                        nm[c], key, b[c], a[c]));
                continue;
            }
            const double d = a[c] - b[c];
            const double s = std::max(std::abs(a[c]), std::abs(b[c]));
            const double rel = s > 0.0 ? std::abs(d) / s : 0.0;
            sumSigned[c] += d;
            sumAbs[c]    += std::abs(d);
            if (std::abs(d) > worstAbs[c]) worstAbs[c] = std::abs(d);
            if (rel > worstRel[c]) { worstRel[c] = rel; worstRelKey[c] = key; worstRelScale[c] = s; }
            if (rel > tol) bad = true;
        }
        if (bad) ++nBad;
    }

    std::string msg = fmt::format(
        "per-parent conservation: {} parents checked (before {}, after {}), "
        "{} missing, {} extra, {} non-finite, {} outside tol {:.1e}",
        nChecked, before.size(), after.size(), nMissing, nExtra, nNonFinite, nBad, tol);
    for (int c = 0; c < 4; ++c)
        msg += fmt::format(
            "\n  {:<5} worst-rel {:.2e} @cart {} (scale {:.3e})  worst-abs {:.3e}  "
            "signed-sum {:+.3e}  abs-sum {:.3e}",
            nm[c], worstRel[c], worstRelKey[c], worstRelScale[c],
            worstAbs[c], sumSigned[c], sumAbs[c]);
    for (const auto& detail : invalidDetails)
        msg += "\n" + detail;

    ok = ok && (nBad == 0) && (nNonFinite == 0);
    if (ok) OpmLog::info(msg);
    else {
        OpmLog::error(msg + "\n  -> per-parent conservation FAILED");
        if (throwOnFail)
            throw std::runtime_error("dynamic refinement: per-parent conservation failed");
    }
    return ok;
}

//! Compare two component inventories component-wise. Logs the table; on a
//! mismatch above \p tol logs an error and (default) throws, so a
//! non-conservative rebuild aborts the run rather than silently continuing.
//! A component that appears from nothing (before==0, after>0), a non-finite
//! entry, or a non-finite \p tol all FAIL -- the earlier `before[i]>0` guard
//! let those through (review 2026-09-10).
inline bool
verifyComponentInventory(const std::array<double, 3>& before,
                         const std::array<double, 3>& after,
                         double tol,
                         bool throwOnFail = true)
{
    static constexpr const char* nm[3] = {"water", "oil", "gas"};
    bool ok = std::isfinite(tol) && tol > 0.0;
    std::string msg = "adaptive state transfer -- black-oil component inventory "
                      "(surface volume):";
    for (int i = 0; i < 3; ++i) {
        const double diff  = std::abs(after[i] - before[i]);
        const double scale = std::max(std::abs(before[i]), std::abs(after[i]));
        // absolute-plus-relative: exact match (scale 0) passes; a component
        // that appears from zero has scale>0 and diff==scale -> rel 1 -> fails.
        const double rel = (scale > 0.0) ? diff / scale : 0.0;
        msg += fmt::format("\n  {:<5} before = {:.12e}  after = {:.12e}  rel.err = {:.3e}",
                           nm[i], before[i], after[i], rel);
        if (!std::isfinite(before[i]) || !std::isfinite(after[i]) || rel > tol)
            ok = false;
    }
    if (ok) {
        OpmLog::info(msg + fmt::format("\n  -> conserved to < {:.1e}", tol));
    } else {
        OpmLog::error(msg + fmt::format("\n  -> NOT conserved (tolerance {:.1e})", tol));
        if (throwOnFail)
            throw std::runtime_error("adaptive state transfer: black-oil component "
                                     "inventory not conserved across the grid rebuild");
    }
    return ok;
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
    // included), written into model().solution(0). processSaturations=false:
    // the source is a converged double-precision state, so a phase below 1e-6
    // is real -- the single-precision-RESTART "drop + renormalise" cleanup
    // would break the per-parent component balance (review 2026-09-10).
    problem.readSolutionFromOutputModule(step, false, /*processSaturations=*/false);

    // A completed init also has history and caches; refresh them.
    simulator.model().solution(/*timeIdx=*/1) = simulator.model().solution(/*timeIdx=*/0);
    simulator.model().invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/0);
}

} // namespace Opm

#endif // OPM_ADAPTIVE_STATE_TRANSFER_HPP
