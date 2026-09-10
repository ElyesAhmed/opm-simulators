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
#include <opm/simulators/flow/AdaptiveStateTransfer.hpp>
#include <opm/simulators/flow/python/PyMain.hpp>

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/input/eclipse/Schedule/Action/State.hpp>
#include <opm/input/eclipse/Schedule/UDQ/UDQState.hpp>
#include <opm/input/eclipse/Schedule/Well/WellTestState.hpp>

#include <opm/models/blackoil/blackoilconvectivemixingmodule.hh>
#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
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
    if (status == EXIT_SUCCESS) {
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
            const auto& mapper = sim->vanguard().cartesianIndexMapper();
            const auto& gdims = mapper.cartesianDimensions();
            const long NXg = gdims[0], NYg = gdims[1], NZg = gdims[2];

            // schedule-wide protected completion cells (global Cartesian)
            std::vector<long> prot;
            const auto& sched = sim->vanguard().schedule();
            for (const auto& w : sched.getWellsatEnd())
                for (const auto& c : w.getConnections())
                    prot.push_back(static_cast<long>(c.global_index()));
            for (const auto& [wn, cells] : sched.getPossibleFutureConnections()) {
                static_cast<void>(wn);
                for (auto gi : cells) prot.push_back(static_cast<long>(gi));
            }
            for (std::size_t rs = 0; rs < sched.size(); ++rs)
                for (const auto& [ijk, sc] : sched[rs].source()) {
                    static_cast<void>(sc);
                    prot.push_back((static_cast<long>(ijk[2]) * NYg + ijk[1]) * NXg + ijk[0]);
                }
            // OPM_APOST_PROTECT_HALO: same near-singularity ring the estimator
            // keeps coarse -- the preflight must not reject a box that only
            // touches the halo, and must reject one that enters it.
            int phalo = 0;
            if (const char* hs = std::getenv("OPM_APOST_PROTECT_HALO"))
                phalo = std::max(0, std::atoi(hs));
            for (int pass = 0; pass < phalo; ++pass) {
                std::vector<long> grown = prot;
                for (long g : prot) {
                    const long i = g % NXg, j = (g / NXg) % NYg, k = g / (NXg * NYg);
                    for (auto [di, dj, dk] : {std::array<long,3>{1,0,0},{-1,0,0},
                                              {0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}) {
                        const long ni = i+di, nj = j+dj, nk = k+dk;
                        if (ni<0||nj<0||nk<0||ni>=NXg||nj>=NYg||nk>=NZg) continue;
                        grown.push_back((nk*NYg+nj)*NXg+ni);
                    }
                }
                prot.swap(grown);
            }
            std::sort(prot.begin(), prot.end());
            prot.erase(std::unique(prot.begin(), prot.end()), prot.end());
            const auto isProt = [&](long i, long j, long k) {
                return std::binary_search(prot.begin(), prot.end(),
                                          (k * NYg + j) * NXg + i);
            };

            std::vector<std::array<long, 6>> boxes;   // i1 i2 j1 j2 k1 k2 (0-based incl.)
            long long refinedLeaf = 0, coarseRefined = 0;
            std::size_t p = 0;
            auto nextTok = [&](long& out) -> bool {
                while (p < spec.size() && (spec[p] == ' ' || spec[p] == ';')) ++p;
                if (p >= spec.size()) return false;
                std::size_t q = p;
                while (q < spec.size() && spec[q] != ' ' && spec[q] != ';') ++q;
                try { out = std::stol(spec.substr(p, q - p)); }
                catch (...) { return false; }
                p = q;
                return true;
            };
            long v[9];
            while (nextTok(v[0])) {
                bool okrow = true;
                for (int t = 1; t < 9 && okrow; ++t) okrow = nextTok(v[t]);
                if (!okrow) {
                    OpmLog::error("preflight: malformed refinement spec (need 9 ints/box).");
                    return false;
                }
                const long i1 = v[0] - 1, i2 = v[1] - 1, j1 = v[2] - 1, j2 = v[3] - 1;
                const long k1 = v[4] - 1, k2 = v[5] - 1;
                const long nx = v[6], ny = v[7], nz = v[8];
                if (i1 < 0 || j1 < 0 || k1 < 0 || i2 >= NXg || j2 >= NYg || k2 >= NZg
                    || i2 < i1 || j2 < j1 || k2 < k1) {
                    OpmLog::error("preflight: refinement box outside the coarse grid.");
                    return false;
                }
                const long ci = i2 - i1 + 1, cj = j2 - j1 + 1, ck = k2 - k1 + 1;
                if (nx <= 0 || ny <= 0 || nz <= 0
                    || nx % ci != 0 || ny % cj != 0 || nz % ck != 0) {
                    OpmLog::error("preflight: box subdivision is not a positive integer "
                                  "multiple of the parent cell count (non-conforming).");
                    return false;
                }
                for (long k = k1; k <= k2; ++k)
                for (long j = j1; j <= j2; ++j)
                for (long i = i1; i <= i2; ++i)
                    if (isProt(i, j, k)) {
                        OpmLog::error("preflight: refinement box contains a protected "
                                      "well-completion cell (would split a well "
                                      "GLOBAL/LGR). Refusing.");
                        return false;
                    }
                for (const auto& b : boxes) {
                    const bool ov = i1 <= b[1] && b[0] <= i2 && j1 <= b[3] && b[2] <= j2
                                 && k1 <= b[5] && b[4] <= k2;
                    if (ov) {
                        OpmLog::error("preflight: refinement boxes overlap.");
                        return false;
                    }
                }
                boxes.push_back({i1, i2, j1, j2, k1, k2});
                coarseRefined += ci * cj * ck;
                refinedLeaf += nx * ny * nz;
            }
            const long long totalLeaf =
                NXg * NYg * NZg - coarseRefined + refinedLeaf;
            OpmLog::info("preflight OK: " + std::to_string(boxes.size()) + " box(es), "
                + std::to_string(coarseRefined) + " coarse cell(s) -> "
                + std::to_string(refinedLeaf) + " child cell(s); refined leaf grid ~"
                + std::to_string(totalLeaf) + " cells (coarse " + std::to_string(NXg*NYg*NZg)
                + "). No box straddles a well; all subdivisions conforming.");
            return true;
        };

        // Tear down world #1, rebuild it refined on `spec`, remap state.
        // Returns false on ANY refusal/failure; every such path sets `status`
        // to a non-zero exit code so the run is not reported as a success.
        auto rebuildOn = [&](const std::string& spec, int step) -> bool {
            auto* sim = flowMain->getSimulatorPtr();

            // Path-dependent physics is NOT carried across the rebuild yet.
            // Guard every unsupported history-dependent feature and refuse
            // BEFORE the old simulator is touched (Copilot/ChatGPT review
            // 2026-09-10: an unsupported deck must fail, not silently shorten).
            const char* unsupported = nullptr;
            if (sim->problem().materialLawManager()->hysteresisConfig().enableHysteresis())
                unsupported = "saturation-function hysteresis";
            else if (sim->vanguard().eclState().aquifer().active())
                unsupported = "analytic/numeric aquifers";
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

            const auto state = extractAdaptiveState<TypeTag>(*sim);
            const auto invBefore = blackOilComponentInventory<TypeTag>(*sim);
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
