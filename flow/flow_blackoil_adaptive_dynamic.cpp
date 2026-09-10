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
#include <opm/simulators/flow/AdaptiveLgr.hpp>
#include <opm/simulators/flow/AdaptiveRefinementProtection.hpp>
#include <opm/simulators/flow/AdaptiveStateTransfer.hpp>
#include <opm/simulators/flow/WellFingerprint.hpp>
#include <opm/simulators/flow/python/PyMain.hpp>

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/input/eclipse/EclipseState/EclipseState.hpp>
#include <opm/input/eclipse/EclipseState/Phase.hpp>
#include <opm/input/eclipse/EclipseState/Runspec.hpp>
#include <opm/input/eclipse/EclipseState/SimulationConfig/RockConfig.hpp>
#include <opm/input/eclipse/EclipseState/SimulationConfig/SimulationConfig.hpp>
#include <opm/input/eclipse/EclipseState/TracerConfig.hpp>
#include <opm/input/eclipse/Schedule/Action/State.hpp>
#include <opm/input/eclipse/Schedule/OilVaporizationProperties.hpp>
#include <opm/input/eclipse/Schedule/UDQ/UDQState.hpp>
#include <opm/input/eclipse/Schedule/Well/WellTestState.hpp>

#include <opm/models/blackoil/blackoilconvectivemixingmodule.hh>
#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>

#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/common/partitionset.hh>
#include <dune/grid/common/rangegenerators.hh>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Opm::Parameters {

//! \brief Report step at which the simulator world is torn down and rebuilt
//! (grid re-refined from the current mark set, state remapped). -1 = never.
//! Phase-1 exercise of the adaptation seam; the mark set is unchanged for now,
//! so the rebuilt run must reproduce the continuous run.
struct AdaptiveRebuildStep { static constexpr int value = -1; };

//! \brief Refinement spec (same CARFIN-style syntax as --adaptive-lgr) that
//! replaces the mark set at the rebuild step. Empty = keep the initial spec.
struct AdaptiveRebuildLgr { static constexpr auto value = ""; };

//! \brief Algorithm 6.1 h-adaptivity: let the a posteriori spatial estimator
//! drive the refinement. After every report step the estimator's marked-REFINE
//! bounding box (apost_refine_request.txt) is read; when it changes, the
//! simulator world is rebuilt on that box. Requires the --enable-aposteriori-*
//! estimator to be active. -1 disables; N = act only from report step N on.
struct AdaptiveEstimatorDriven { static constexpr int value = -1; };

} // namespace Opm::Parameters

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

} // namespace Opm::Properties

namespace Opm {

namespace {

//! Parse an environment variable that must be a plain positive integer.
//! Returns std::nullopt when the variable is unset. Throws std::invalid_argument
//! when it is set but malformed (trailing junk, sign, overflow, <= 0) so a
//! typo silently disabling a safety limit is turned into a hard failure
//! (review 2026-09-10: std::atoll("12x")==12, std::atoll("bad")==0).
std::optional<long long> envPositiveInt(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return std::nullopt;
    }
    std::string_view sv{raw};
    const auto b = sv.find_first_not_of(" \t");
    const auto e = sv.find_last_not_of(" \t");
    if (b == std::string_view::npos) {
        throw std::invalid_argument(fmt::format("{} is set but empty.", name));
    }
    sv = sv.substr(b, e - b + 1);

    long long value = 0;
    const auto* first = sv.data();
    const auto* last = sv.data() + sv.size();
    const auto res = std::from_chars(first, last, value);
    if (res.ec != std::errc{} || res.ptr != last) {
        throw std::invalid_argument(
            fmt::format("{}='{}' is not a plain integer.", name, sv));
    }
    if (value <= 0) {
        throw std::invalid_argument(
            fmt::format("{}={} must be strictly positive.", name, value));
    }
    return value;
}

} // anonymous namespace

//! \brief The adaptive vanguard plus the dynamic driver's parameters
//! (registration must happen in the registration phase, so it lives here).
template <class TypeTag>
class AdaptiveDynamicVanguard : public AdaptiveCpGridVanguard<TypeTag>
{
    using Base = AdaptiveCpGridVanguard<TypeTag>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
public:
    explicit AdaptiveDynamicVanguard(Simulator& simulator)
        : Base(simulator)
    {}

    static void registerParameters()
    {
        Base::registerParameters();
        Parameters::Register<Parameters::AdaptiveRebuildStep>(
            "Report step at which the dynamic driver tears down and rebuilds "
            "the simulator (dynamic-refinement phase 1); -1 disables.");
        Parameters::Register<Parameters::AdaptiveRebuildLgr>(
            "Refinement spec applied at the rebuild step (CARFIN-style, see "
            "--adaptive-lgr); empty keeps the initial spec.");
        Parameters::Register<Parameters::AdaptiveEstimatorDriven>(
            "Algorithm 6.1 h-adaptivity: from this report step on, let the a "
            "posteriori spatial estimator's marked-REFINE box drive grid "
            "rebuilds (requires --enable-aposteriori-*); -1 disables.");
    }

    //! Mark-set override for the rebuild (process-wide: the driver sets it
    //! before constructing the second simulator world).
    static inline std::string rebuildSpecOverride{};

    std::string adaptiveLgrSpec() const override
    {
        if (rebuildSpecOverride == "none") {   // un-mark everything: coarsen
            return {};
        }
        return rebuildSpecOverride.empty()
            ? Base::adaptiveLgrSpec() : rebuildSpecOverride;
    }
};

} // namespace Opm

namespace Opm::Properties {

template<class TypeTag>
struct Vanguard<TypeTag, TTag::FlowProblemAdaptiveDynamic>
{ using type = AdaptiveDynamicVanguard<TypeTag>; };

} // namespace Opm::Properties

namespace Opm {

//! \brief PyMain with a second-init entry: rebuild the FlowMain/simulator
//! world from the already-parsed model description (no re-parse, no second
//! MPI init). One DynamicMain object lives for the whole process.
template <class TypeTag>
class DynamicMain : public PyMain<TypeTag>
{
public:
    using FlowMainType = FlowMain<TypeTag>;
    using PyMain<TypeTag>::PyMain;

    //! Re-populate the vanguard's (moved-from) static model parameters and
    //! construct a fresh FlowMain. The shared_ptrs come from the snapshot the
    //! driver took before the first vanguard construction; the evolving
    //! schedule-state objects are carried over from the previous simulator.
    std::unique_ptr<FlowMainType>
    rebuildFlowBlackoil(const FlowGenericVanguard::SimulationModelParams& snapshot,
                        const Action::State& actionState,
                        const UDQState& udqState)
    {
        FlowGenericVanguard::modelParams_.setupTime_ = snapshot.setupTime_;
        FlowGenericVanguard::modelParams_.eclState_ = snapshot.eclState_;
        FlowGenericVanguard::modelParams_.eclSchedule_ = snapshot.eclSchedule_;
        FlowGenericVanguard::modelParams_.eclSummaryConfig_ = snapshot.eclSummaryConfig_;
        FlowGenericVanguard::modelParams_.actionState_ =
            std::make_unique<Action::State>(actionState);
        FlowGenericVanguard::modelParams_.udqState_ =
            std::make_unique<UDQState>(udqState);
        // Phase-1 limitation: WTEST history restarts empty (the vanguard's
        // WellTestState was moved into the previous well model).
        FlowGenericVanguard::modelParams_.wtestState_ =
            std::make_unique<WellTestState>();

        return flowMainInit<TypeTag>(this->argc_, this->argv_,
                                     this->outputCout_, this->outputFiles_);
    }
};

// ----------------- Main program: the report-step driver -----------------
//
// Instead of FlowMain::execute() (which owns the whole time loop), drive the
// run one report step at a time through the step API the Python bindings use:
//   initFlowBlackoil() -> executeInitStep() -> executeStep()* -> cleanup.
// PyMain is a plain-C++ Main subclass (no Python dependency) that splits
// initialization from the run loop; we reuse it as the bootstrap.
//
// The step loop is the adaptation seam of the dynamic-refinement plan
// (opm-gridrefined docs/DYNAMIC-REFINEMENT-FLOW-PLAN.md, phase 1 steps
// S2-S5): at --adaptive-rebuild-step=N the simulator world is torn down,
// rebuilt from the already-parsed model description (the vanguard re-refines
// during construction), and the extracted state is remapped by stable cell id
// and injected through the restart machinery. With an unchanged mark set the
// rebuilt run must reproduce the continuous run (plan test S8a).
int flowBlackoilTpfaAdaptiveDynamicMainStandalone(int argc, char** argv)
{
    using TypeTag = Properties::TTag::FlowProblemAdaptiveDynamic;

    // we always want to use the default locale, and thus spare us the trouble
    // with incorrect locale settings.
    resetLocale();

    // Main's ctor/dtor own MPI_Init/MPI_Finalize.
    auto mainObject = std::make_unique<DynamicMain<TypeTag>>(argc, argv);

    int exitCode = EXIT_SUCCESS;
    auto flowMain = mainObject->initFlowBlackoil(exitCode);
    if (!flowMain) {
        mainObject.reset();
        return exitCode;
    }

    // The vanguard's static model parameters are populated now and will be
    // moved out during the first vanguard construction (inside
    // executeInitStep). Snapshot the shared_ptrs first so a rebuild can
    // re-populate them without re-parsing the deck.
    FlowGenericVanguard::SimulationModelParams snapshot;
    snapshot.setupTime_ = FlowGenericVanguard::modelParams_.setupTime_;
    snapshot.eclState_ = FlowGenericVanguard::modelParams_.eclState_;
    snapshot.eclSchedule_ = FlowGenericVanguard::modelParams_.eclSchedule_;
    snapshot.eclSummaryConfig_ = FlowGenericVanguard::modelParams_.eclSummaryConfig_;

    int status = flowMain->executeInitStep();
    // NOSIM / --enable-dry-run initializes the grid but intentionally does
    // not create a step driver. Do not enter stepping or finalize a null one.
    if (status == EXIT_SUCCESS && flowMain->getStepDriverPtr()) {
        const int rebuildStep = Parameters::Get<Parameters::AdaptiveRebuildStep>();
        const int estDrivenFrom = Parameters::Get<Parameters::AdaptiveEstimatorDriven>();

        // Algorithm 6.1 h-adaptivity: the estimator publishes its marked-REFINE
        // box to this file on every converged report step; we read it back at
        // the next step boundary.
        const std::string reqPath = "apost_refine_request.txt";
        if (estDrivenFrom >= 0) {
            ::setenv("OPM_APOST_REFINE_REQUEST", reqPath.c_str(), /*overwrite=*/1);
            std::remove(reqPath.c_str());   // drop any stale request from a prior run
        }
        auto readRefineRequest = [&reqPath]() -> std::string {
            std::ifstream is(reqPath);
            std::string line;
            if (is && std::getline(is, line)) {
                const auto b = line.find_first_not_of(" \t\r\n");
                if (b == std::string::npos) return {};
                const auto e = line.find_last_not_of(" \t\r\n");
                return line.substr(b, e - b + 1);
            }
            return {};
        };
        // The spec the current grid was built on (empty = unrefined base run).
        std::string appliedSpec = Parameters::Get<Parameters::AdaptiveLgr>();

        // ---- Preflight: validate a refinement spec against the LIVE coarse
        // grid + well set BEFORE anything destructive happens. Catches a
        // malformed / well-straddling / non-conforming spec while the coarse
        // simulator is still intact, so a bad request fails cleanly instead of
        // tearing the world down first (review 2026-09-10: not a full
        // transaction, but "validate the checkable part before destroy").
        auto preflightSpec = [&](const std::string& spec) -> bool {
            auto* sim = flowMain->getSimulatorPtr();
            const auto& gdims = sim->vanguard().cartesianIndexMapper().cartesianDimensions();
            const long NXg = gdims[0], NYg = gdims[1], NZg = gdims[2];

            // (1) grammar + subdivision -- the SAME parser the grid builder uses
            // (AdaptiveLgr.hpp::parseAdaptiveLgrSpec), not a second copy. Typed
            // boxes: startIJK 0-based inclusive, endIJK 0-based EXCLUSIVE,
            // cellsPerDim = per-parent factor.
            std::vector<Opm::AdaptiveLgrBox> boxes;
            try {
                boxes = Opm::parseAdaptiveLgrSpec(spec);
            } catch (const std::exception& e) {
                OpmLog::error(std::string("preflight: ") + e.what());
                return false;
            }
            if (boxes.empty()) {
                OpmLog::error("preflight: empty refinement spec.");
                return false;
            }

            // (2) shared protected set (AdaptiveRefinementProtection.hpp)
            int phalo = 0;
            if (const char* hs = std::getenv("OPM_APOST_PROTECT_HALO"))
                phalo = std::max(0, std::atoi(hs));
            const auto protVec = buildProtectedRefinementCells(
                sim->vanguard().schedule(),
                {static_cast<int>(NXg), static_cast<int>(NYg), static_cast<int>(NZg)}, phalo);
            const auto isProt = [&](long i, long j, long k) {
                return std::binary_search(protVec.begin(), protVec.end(),
                    static_cast<int>((k * NYg + j) * NXg + i));
            };

            // (4a) safety limits -- parsed once, strictly (a malformed value is
            // a hard failure, not a silently-disabled guard).
            std::optional<long long> maxLeaf;
            long long maxSubdiv = 64;   // per-axis per-parent subdivision ceiling
            try {
                maxLeaf = envPositiveInt("OPM_APOST_MAX_LEAF_CELLS");
                if (const auto s = envPositiveInt("OPM_APOST_MAX_SUBDIV_PER_AXIS")) {
                    maxSubdiv = *s;
                }
            } catch (const std::exception& e) {
                OpmLog::error(std::string("preflight: ") + e.what());
                return false;
            }
            const long long kOverflowGuard =
                std::numeric_limits<long long>::max() / 4;

            long long coarseRefined = 0, refinedLeaf = 0;
            for (std::size_t bi = 0; bi < boxes.size(); ++bi) {
                const auto& b = boxes[bi];
                const long i1 = b.startIJK[0], j1 = b.startIJK[1], k1 = b.startIJK[2];
                const long i2 = b.endIJK[0] - 1, j2 = b.endIJK[1] - 1, k2 = b.endIJK[2] - 1;

                if (i1 < 0 || j1 < 0 || k1 < 0 || i2 >= NXg || j2 >= NYg || k2 >= NZg
                    || i2 < i1 || j2 < j1 || k2 < k1) {
                    OpmLog::error(fmt::format("preflight: box {} outside the coarse grid.", b.name));
                    return false;
                }
                for (int c = 0; c < 3; ++c) {
                    const long long f = b.cellsPerDim[c];
                    if (f < 1 || f > maxSubdiv) {
                        OpmLog::error(fmt::format(
                            "preflight: box {} subdivision factor {} in direction {} "
                            "outside [1, {}] (OPM_APOST_MAX_SUBDIV_PER_AXIS).",
                            b.name, f, c, maxSubdiv));
                        return false;
                    }
                }
                for (long k = k1; k <= k2; ++k)
                for (long j = j1; j <= j2; ++j)
                for (long i = i1; i <= i2; ++i)
                    if (isProt(i, j, k)) {
                        OpmLog::error(fmt::format("preflight: box {} enters a protected cell "
                            "(well completion / same-well k-span / SOURCE / halo) at "
                            "({},{},{}). Refusing.", b.name, i + 1, j + 1, k + 1));
                        return false;
                    }
                // (3) box-pair rules -- exactly the ConformingBlockBuilder logic:
                //  * volumetric overlap (all 3 dims overlap) is rejected;
                //  * boxes that interact (touch-or-overlap in EVERY dim) must
                //    have equal cellsPerDim in each dim where they OVERLAP
                //    (touch-only dims are unconstrained).
                for (std::size_t bj = 0; bj < bi; ++bj) {
                    const auto& a = boxes[bj];
                    bool interacts = true, volOverlap = true;
                    bool ovDim[3];
                    for (int c = 0; c < 3; ++c) {
                        const bool gap = a.endIJK[c] <= b.startIJK[c]
                                      || b.endIJK[c] <= a.startIJK[c];
                        // endIJK exclusive: "touch" is endIJK[c] == startIJK[c].
                        const bool touchOrOverlap =
                            a.endIJK[c] >= b.startIJK[c] && b.endIJK[c] >= a.startIJK[c];
                        ovDim[c] = a.startIJK[c] < b.endIJK[c]
                                && b.startIJK[c] < a.endIJK[c];
                        static_cast<void>(gap);
                        interacts = interacts && touchOrOverlap;
                        volOverlap = volOverlap && ovDim[c];
                    }
                    if (volOverlap) {
                        OpmLog::error(fmt::format("preflight: boxes {} and {} overlap.",
                                                  a.name, b.name));
                        return false;
                    }
                    if (interacts)
                        for (int c = 0; c < 3; ++c)
                            if (ovDim[c] && a.cellsPerDim[c] != b.cellsPerDim[c]) {
                                OpmLog::error(fmt::format(
                                    "preflight: boxes {} and {} meet with different "
                                    "subdivision factors ({} vs {}) in direction {} "
                                    "-- non-conforming shared interface.",
                                    a.name, b.name, a.cellsPerDim[c], b.cellsPerDim[c], c));
                                return false;
                            }
                }
                // Checked multiplication: box parents, then box leaves. Each
                // factor is already bounded (extent <= grid dim, subdivision
                // <= maxSubdiv), but bound the products explicitly so a
                // pathological spec cannot wrap a signed 64-bit accumulator.
                const long long ci = i2 - i1 + 1, cj = j2 - j1 + 1, ck = k2 - k1 + 1;
                const long long boxParents = ci * cj * ck;                 // <= base, safe
                const long long fProd = static_cast<long long>(b.cellsPerDim[0])
                                      * b.cellsPerDim[1] * b.cellsPerDim[2]; // <= maxSubdiv^3
                if (fProd != 0 && boxParents > kOverflowGuard / fProd) {
                    OpmLog::error(fmt::format(
                        "preflight: box {} projected child count overflows "
                        "({} parents x {} per parent).", b.name, boxParents, fProd));
                    return false;
                }
                const long long boxLeaves = boxParents * fProd;
                if (coarseRefined > kOverflowGuard - boxParents
                    || refinedLeaf > kOverflowGuard - boxLeaves) {
                    OpmLog::error(fmt::format(
                        "preflight: cumulative refined-cell count overflows at box {}.",
                        b.name));
                    return false;
                }
                coarseRefined += boxParents;
                refinedLeaf   += boxLeaves;
            }

            // (4) projected-leaf budget -- ENFORCED here, not just logged.
            const long long base = NXg * NYg * NZg;
            const long long totalLeaf = base - coarseRefined + refinedLeaf;
            if (maxLeaf && totalLeaf > *maxLeaf) {
                OpmLog::error(fmt::format("preflight: projected leaf grid {} exceeds "
                    "OPM_APOST_MAX_LEAF_CELLS={}.", totalLeaf, *maxLeaf));
                return false;
            }
            OpmLog::info(fmt::format(
                "preflight OK  |  boxes {}  base active {}  marked parents {}  "
                "child cells {}  -> final leaf ~{}  (subject to ACTNUM).  "
                "No box enters a protected cell; interfaces conforming.",
                boxes.size(), base, coarseRefined, refinedLeaf, totalLeaf));
            return true;
        };

        // Tear down world #1, rebuild it refined on `spec`, remap state.
        // Returns false on ANY refusal/failure; every such path sets `status`
        // to a non-zero exit code so the run is not reported as a success.
        auto rebuildOn = [&](const std::string& spec, int step) -> bool {
            auto* sim = flowMain->getSimulatorPtr();

            // Path-dependent / non-black-oil state is NOT carried across the
            // rebuild: the transfer only moves {p, Sw, Sg, Rs, Rv, T}. Guard
            // every deck feature whose per-cell history would be silently lost
            // and refuse BEFORE the old simulator is touched (Copilot/ChatGPT
            // review 2026-09-10: an unsupported deck must fail, not shorten).
            const char* unsupported = nullptr;
            {
                const auto& es = sim->vanguard().eclState();
                const auto& rspec = es.runspec();
                const auto& ph = rspec.phases();
                const auto& rock = es.getSimulationConfig().rock_config();
                const auto epi = std::max(0, sim->episodeIndex());
                const auto& ovp = sim->vanguard().schedule()[epi].oilvap();
                using P = Phase;

                if (sim->problem().materialLawManager()->hysteresisConfig().enableHysteresis())
                    unsupported = "saturation-function hysteresis";
                else if (es.aquifer().active())
                    unsupported = "analytic/numeric aquifers";
                else if (ph.active(P::SOLVENT))    unsupported = "the solvent model";
                else if (ph.active(P::POLYMER))    unsupported = "the polymer model";
                else if (ph.active(P::POLYMW))     unsupported = "polymer molecular weight";
                else if (ph.active(P::FOAM))       unsupported = "the foam model";
                else if (ph.active(P::BRINE))      unsupported = "the brine model";
                else if (ph.active(P::ZFRACTION))  unsupported = "the z-fraction (GASWAT) model";
                else if (ph.active(P::ENERGY))     unsupported = "the thermal/energy model";
                else if (rspec.micp())             unsupported = "the MICP model";
                else if (rspec.co2Storage())       unsupported = "CO2STORE";
                else if (rspec.h2Storage())        unsupported = "H2STORE";
                else if (rspec.co2Sol())           unsupported = "dissolved CO2 (CO2SOL)";
                else if (rspec.h2Sol())            unsupported = "dissolved H2 (H2SOL)";
                else if (!es.tracer().empty())     unsupported = "passive tracers";
                else if (rock.active() &&
                         (rock.hysteresis_mode() == RockConfig::Hysteresis::IRREVERS ||
                          rock.water_compaction()))
                    unsupported = "irreversible / water-induced rock compaction";
                else if (ovp.drsdtConvective())    unsupported = "convective dissolution (DRSDTCON)";
            }
            if (unsupported) {
                OpmLog::error(std::string("flow_blackoil_adaptive_dynamic: the deck uses ")
                    + unsupported + " -- its path-dependent state is not transferred "
                    "across a grid rebuild. Refusing the rebuild (run fails).");
                status = EXIT_FAILURE;
                return false;
            }

            // Validate the spec against the still-intact coarse world first.
            if (!spec.empty() && spec != "none" && !preflightSpec(spec)) {
                status = EXIT_FAILURE;
                return false;
            }

            AdaptiveStateMap state;
            std::array<double, 3> invBefore{};
            std::unordered_map<std::int64_t, std::array<double, 4>> invParentBefore;
            try {
                state = extractAdaptiveState<TypeTag>(*sim);
                invBefore = blackOilComponentInventory<TypeTag>(*sim);
                invParentBefore = blackOilInventoryByParent<TypeTag>(*sim);
            } catch (const std::exception& e) {
                OpmLog::error(std::string("flow_blackoil_adaptive_dynamic: pre-rebuild "
                    "state capture failed (coarse world intact, run fails): ") + e.what());
                status = EXIT_FAILURE;
                return false;
            }
            const auto wellFpBefore = captureWellFingerprint(
                sim->vanguard().schedule(), sim->episodeIndex());
            // schedule-wide protected SOURCE cells, to confirm they stay coarse
            std::vector<long> srcCells;
            {
                const auto& sc = sim->vanguard().cartesianIndexMapper().cartesianDimensions();
                const long nxs = sc[0], nys = sc[1];
                const auto& sch = sim->vanguard().schedule();
                for (std::size_t rs = 0; rs < sch.size(); ++rs)
                    for (const auto& [ijk, sd] : sch[rs].source()) {
                        static_cast<void>(sd);
                        srcCells.push_back((static_cast<long>(ijk[2]) * nys + ijk[1]) * nxs + ijk[0]);
                    }
                std::sort(srcCells.begin(), srcCells.end());
                srcCells.erase(std::unique(srcCells.begin(), srcCells.end()), srcCells.end());
            }
            const Action::State actionState = sim->vanguard().actionState();
            const UDQState udqState = sim->vanguard().udqState();
            // Carry the adaptive time stepper's rhythm across the rebuild: a
            // fresh init restarts the substep ramp from TSINIT, which changes
            // the post-rebuild time discretisation and its temporal error (an
            // identity rebuild otherwise differs from the continuous run by
            // ~10% in the well rates purely from this).
            const double nextDt = flowMain->getStepDriverPtr()
                ? flowMain->getStepDriverPtr()->suggestedNextStep() : -1.0;

            AdaptiveDynamicVanguard<TypeTag>::rebuildSpecOverride =
                spec.empty() ? std::string{"none"} : spec;

            // NOTE (review 2026-09-10): this is NOT yet a transaction -- the old
            // simulator is destroyed here, before the candidate is validated.
            // A failed candidate init / inventory check below cannot roll back
            // to the coarse run; it can only fail the whole run cleanly.
            flowMain.reset();
            try {
                flowMain = mainObject->rebuildFlowBlackoil(snapshot, actionState, udqState);
                status = flowMain->executeInitStep();
            } catch (const std::exception& e) {
                OpmLog::error(std::string("flow_blackoil_adaptive_dynamic: candidate "
                    "simulator construction failed: ") + e.what());
                status = EXIT_FAILURE;
                return false;
            }
            if (status != EXIT_SUCCESS) {
                return false;
            }
            auto* sim2 = flowMain->getSimulatorPtr();

            bool ok = false;
            try {
                const auto sol = remapAdaptiveState<TypeTag>(state, *sim2);
                injectAdaptiveState<TypeTag>(*sim2, sol, step);

                // Strict conservativeness gate: constant intensive prolongation
                // must preserve every black-oil component inventory (child pore
                // volumes partition the parent's). tol must be finite & > 0.
                double invTol = 1e-10;
                if (const char* s = std::getenv("OPM_APOST_INVENTORY_TOL")) {
                    const double v = std::atof(s);
                    if (std::isfinite(v) && v > 0.0) invTol = v;
                }
                const auto invAfter = blackOilComponentInventory<TypeTag>(*sim2);
                ok = verifyComponentInventory(invBefore, invAfter, invTol,
                                              /*throwOnFail=*/false);

                // per-parent conservation (a global sum hides opposite local errors)
                const auto invParentAfter = blackOilInventoryByParent<TypeTag>(*sim2);
                const bool ppOk = verifyPerParentConservation(
                    invParentBefore, invParentAfter, std::max(invTol, 1e-9));

                // well fingerprint: protected wells must come back identical.
                // OPM_APOST_VALIDATE=warn downgrades a fingerprint / per-parent
                // failure to a warning (research runs); default = hard gate.
                const bool fpOk = verifyWellFingerprint(
                        wellFpBefore,
                        captureWellFingerprint(sim2->vanguard().schedule(),
                                               sim2->episodeIndex()));
                const char* vmode = std::getenv("OPM_APOST_VALIDATE");
                const bool warnOnly = vmode && std::string(vmode) == "warn";
                if (!warnOnly) ok = ok && ppOk && fpOk;
                else if (!ppOk || !fpOk)
                    OpmLog::warning("OPM_APOST_VALIDATE=warn: continuing despite a "
                                    "per-parent / well-fingerprint failure.");

                // Every SOURCE coordinate must resolve to EXACTLY ONE interior
                // leaf cell after the rebuild -- it is protected, so it must
                // stay coarse; more than one leaf means the source was refined
                // and FlowProblemBlackoil would apply the deck rate per child
                // (reviews 2026-09-10: a single compressedIndexForInterior()
                // lookup proves existence, not uniqueness).
                if (!srcCells.empty()) {
                    std::unordered_map<long, int> leafPerSource;
                    const auto& cim2 = sim2->vanguard().cartesianIndexMapper();
                    const auto& gv2 = sim2->vanguard().gridView();
                    auto mapper2 = Dune::MultipleCodimMultipleGeomTypeMapper<
                        std::decay_t<decltype(gv2)>>(gv2, Dune::mcmgElementLayout());
                    for (const auto& e : elements(gv2, Dune::Partitions::interior)) {
                        const long cart = static_cast<long>(
                            cim2.cartesianIndex(mapper2.index(e)));
                        if (std::binary_search(srcCells.begin(), srcCells.end(), cart))
                            ++leafPerSource[cart];
                    }
                    for (long g : srcCells) {
                        const int n = leafPerSource.count(g) ? leafPerSource[g] : 0;
                        if (n != 1) {
                            OpmLog::error(fmt::format(
                                "SOURCE cell (global {}) maps to {} interior leaf "
                                "cell(s) after rebuild (must be exactly 1).", g, n));
                            ok = false;
                        }
                    }
                }
            } catch (const std::exception& e) {
                OpmLog::error(std::string("flow_blackoil_adaptive_dynamic: state "
                    "remap/inject failed: ") + e.what());
                ok = false;
            }
            if (!ok) {
                OpmLog::error("flow_blackoil_adaptive_dynamic: rebuild rejected "
                              "(state transfer not validated). Run fails.");
                status = EXIT_FAILURE;
                return false;
            }

            flowMain->getSimTimer()->setCurrentStepNum(step);
            if (flowMain->getStepDriverPtr()) {
                flowMain->getStepDriverPtr()->setSuggestedNextStep(nextDt);
            }
            appliedSpec = spec;
            return true;
        };

        // One executeStep() per report step. runStep returns a continue flag
        // (false = schedule EXIT), not an exit status; errors throw.
        //
        // Phase-1 limitation: the state-transfer machinery snapshots the parsed
        // model once, but Schedule::synthesizeWellTrajectories() mutates the
        // shared schedule in place during the first refined build, so a SECOND
        // teardown re-resolves already-synthetic trajectories against yet
        // another leaf and wells fall off the grid. Until the snapshot also
        // deep-copies the schedule, cap estimator-driven adaptation at one
        // rebuild -- enough to demonstrate Algorithm 6.1's spatial marking
        // driving a real grid change.
        // ONE guard for every refined-grid construction, whatever triggered it
        // (--adaptive-rebuild-step, or the estimator). The state-transfer
        // snapshot is taken once and Schedule::synthesizeWellTrajectories()
        // mutates the shared schedule in place on the first refined build, so a
        // second teardown re-resolves already-synthetic trajectories against
        // another leaf and wells fall off the grid. Serial only: the marking is
        // rank-local and there is no global Dörfler / gather / canonical spec.
        if ((estDrivenFrom >= 0 || rebuildStep >= 0)
            && flowMain->getSimulatorPtr()->gridView().comm().size() > 1) {
            OpmLog::error("flow_blackoil_adaptive_dynamic: mid-run grid adaptation "
                          "(--adaptive-rebuild-step / --adaptive-estimator-driven) "
                          "is serial-only. Run with one MPI rank.");
            return EXIT_FAILURE;
        }
        bool worldRebuilt = false;
        bool continueLooping = true;
        while (continueLooping && !flowMain->getSimTimer()->done()) {
            const int step = flowMain->getSimTimer()->currentStepNum();

            if (!worldRebuilt && step == rebuildStep) {
                // ---- fixed-schedule adaptation event (S2-S5) ----
                const std::string newSpec =
                    Parameters::Get<Parameters::AdaptiveRebuildLgr>();
                if (!rebuildOn(newSpec.empty() ? appliedSpec : newSpec, step)) {
                    break;
                }
                worldRebuilt = true;
            }
            else if (!worldRebuilt && estDrivenFrom >= 0 && step >= estDrivenFrom) {
                // ---- Algorithm 6.1 estimator-driven adaptation ----
                const std::string want = readRefineRequest();
                if (!want.empty() && want != appliedSpec) {
                    OpmLog::info("\n[Algorithm 6.1] estimator-driven refinement at "
                                 "report step " + std::to_string(step)
                                 + ": rebuilding grid on box '" + want + "'");
                    if (!rebuildOn(want, step)) {
                        break;
                    }
                    worldRebuilt = true;
                }
            }
            continueLooping = (flowMain->executeStep() != 0);
        }
        if (status == EXIT_SUCCESS) {
            status = flowMain->executeStepsCleanup();
        }
    }

    flowMain.reset();
    mainObject.reset();   // destructor calls MPI_Finalize
    return status;
}

} // namespace Opm
