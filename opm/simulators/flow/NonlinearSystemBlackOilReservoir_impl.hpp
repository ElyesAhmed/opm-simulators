/*
  Copyright 2013, 2015 SINTEF ICT, Applied Mathematics.
  Copyright 2014, 2015 Dr. Blatt - HPC-Simulation-Software & Services
  Copyright 2014, 2015 Statoil ASA.
  Copyright 2015 NTNU
  Copyright 2015, 2016, 2017 IRIS AS

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

#ifndef OPM_NONLINEAR_SYSTEM_BLACK_OIL_RESERVOIR_IMPL_HEADER_INCLUDED
#define OPM_NONLINEAR_SYSTEM_BLACK_OIL_RESERVOIR_IMPL_HEADER_INCLUDED

#ifndef OPM_NONLINEAR_SYSTEM_BLACK_OIL_RESERVOIR_HEADER_INCLUDED
#include <config.h>
#include <opm/simulators/flow/NonlinearSystemBlackOilReservoir.hpp>
#endif

#include <dune/common/timer.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/istl/solvers.hh>

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/models/utils/parametersystem.hpp>

#include <opm/simulators/flow/AdaptiveRefinementProtection.hpp>
#include <opm/simulators/flow/countGlobalCells.hpp>
#include <opm/simulators/linalg/FlowLinearSolverParameters.hpp>
#include <opm/simulators/linalg/WellOperators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace {
    template <typename TypeTag>
    std::string_view
    make_string(const typename Opm::NonlinearSystemBlackOilReservoir<TypeTag>::DebugFlags f)
    {
        using F = typename Opm::NonlinearSystemBlackOilReservoir<TypeTag>::DebugFlags;

        switch (f) {
        case F::STRICT:   return "Strict";
        case F::RELAXED:  return "Relaxed";
        case F::TUNINGDP: return "TuningDP";
        }

        return "< ??? >";
    }
} // Anonymous namespace

namespace Opm {

template <class TypeTag>
NonlinearSystemBlackOilReservoir<TypeTag>::
NonlinearSystemBlackOilReservoir(Simulator& simulator,
              const ModelParameters& param,
              typename ParentType::WellModel& well_model,
              const bool terminal_output)
    : ParentType(simulator, param, well_model, terminal_output)
    , conv_monitor_(param.monitor_params_)
{
    // compute global sum of number of cells
    global_nc_ = detail::countGlobalCells(this->grid_);
    this->convergence_reports_.reserve(300); // Often insufficient, but avoids frequent moves.

    // Inexact-Newton adaptive linear tolerance. The tightest reduction it will
    // ever ask for is the statically configured --linear-solver-reduction, so
    // the linear solve is never made less accurate than the user requested.
    // Enabled for EITHER the CNV-ratio Eisenstat--Walker term
    // (--adaptive-linear-solver-reduction) or the estimator-driven
    // Criteria_alg term (--enable-aposteriori-linear-tolerance); both share
    // this object's clamps and few-iterations guard.
    const bool aposteriori_lin_tol =
        this->param_.enable_aposteriori_linear_tolerance_ && this->param_.enable_aposteriori_estimators_;
    adaptive_linear_reduction_ = AdaptiveLinearSolveReduction<Scalar>(
        this->param_.adaptive_linear_solver_reduction_ || aposteriori_lin_tol,
        this->param_.adaptive_linear_solver_reduction_gamma_,
        static_cast<Scalar>(Parameters::Get<Parameters::LinearSolverReduction>()),
        this->param_.adaptive_linear_solver_reduction_max_,
        this->param_.adaptive_linear_solver_reduction_min_iter_);
    if (adaptive_linear_reduction_.enabled() && terminal_output) {
        OpmLog::info("Using inexact-Newton adaptive tolerance for the linear solve "
                     "(--adaptive-linear-solver-reduction=true).");
    }

    if (this->param_.enable_aposteriori_estimators_) {
        aposteriori_estimator_ =
            std::make_unique<APosterioriSpatialTemporalEstimator<TypeTag>>(simulator);
        aposteriori_targets_.gammaTime = this->param_.aposteriori_gamma_time_;
        aposteriori_targets_.GammaTime = this->param_.aposteriori_gamma_time_upper_;
        aposteriori_targets_.GammaLin  = this->param_.aposteriori_gamma_lin_;
        aposteriori_targets_.GammaAlg  = this->param_.aposteriori_gamma_alg_;
        // Default (aposteriori_tol_mb_ <= 0): match the simulator's own MB
        // floor -- the paper's "non-negotiable" condition is only meaningful
        // if it's the same tolerance the simulator actually enforces. A
        // positive --aposteriori-tol-mb overrides it (experimental; relaxing
        // it lets Criteria_newton stop before MB has strictly converged).
        aposteriori_targets_.tolMB = (this->param_.aposteriori_tol_mb_ > Scalar{0})
            ? this->param_.aposteriori_tol_mb_
            : this->param_.tolerance_mb_;

        aposteriori_estimator_->setWeightExponent(this->param_.aposteriori_weight_exponent_);
        aposteriori_estimator_->setEpsilon(this->param_.aposteriori_epsilon_);
        aposteriori_estimator_->setCheapNorms(this->param_.aposteriori_cheap_norms_);
        aposteriori_estimator_->setDisableCkkWeight(this->param_.aposteriori_disable_ckk_weight_);
        aposteriori_estimator_->setSeparateNeumannMean(this->param_.aposteriori_separate_neumann_mean_);
        aposteriori_estimator_->setMobilityEnergyNorm(this->param_.aposteriori_mobility_energy_norm_);
        aposteriori_estimator_->setMobilityFloorFraction(this->param_.aposteriori_mobility_floor_fraction_);
        aposteriori_estimator_->setMvemStabilityEpsilon(this->param_.aposteriori_mvem_stability_epsilon_);
        aposteriori_estimator_->setUseLiftedRelperm(this->param_.aposteriori_use_lifted_relperm_);
        aposteriori_estimator_->setUseBubbleCorrection(this->param_.aposteriori_use_bubble_correction_);
        aposteriori_estimator_->setPressureReconstruction(
            this->param_.aposteriori_use_flux_taylor_pressure_
                ? APosterioriSpatialTemporalEstimator<TypeTag>::PressureRecon::FluxTaylorLift
                : (this->param_.aposteriori_use_connection_ls_gradient_
                    ? APosterioriSpatialTemporalEstimator<TypeTag>::PressureRecon::ConnectionLS
                    : APosterioriSpatialTemporalEstimator<TypeTag>::PressureRecon::PatchAverageLift));

        // The vertex-patch reconstruction and the linearization-defect mirror
        // lookup are not communicated across MPI partition boundaries, so
        // eta_lin / eta_sp are partition-dependent there. Refuse to let them
        // drive convergence in parallel (time-step control -- which only
        // reports or shrinks -- is left available).
        if (this->grid_.comm().size() > 1) {
            if (this->param_.enable_aposteriori_newton_stopping_ ||
                this->param_.enable_aposteriori_linear_tolerance_) {
                if (terminal_output)
                    OpmLog::warning(
                        "A posteriori estimator-driven Newton stopping / linear "
                        "tolerance is not partition-consistent and is DISABLED "
                        "for this MPI run; the estimators are still reported.");
                this->param_.enable_aposteriori_newton_stopping_ = false;
                this->param_.enable_aposteriori_linear_tolerance_ = false;
            }
        }

        // The endpoint-verified controller may be combined with estimator
        // Newton stopping. A failed algebraic endpoint check now latches the
        // next Newton solve to the configured strict reduction, so an unmet
        // target cannot repeatedly starve nonlinear convergence. Correction
        // solves recompute the full increment from zero with the requested
        // reduction calibrated by the measured physical endpoint shortfall.
        if (this->param_.enable_aposteriori_newton_stopping_
            && this->param_.enable_aposteriori_linear_tolerance_
            && this->param_.aposteriori_alg_max_resolves_ > 0) {
            if (terminal_output)
                OpmLog::info(fmt::format(
                    "Endpoint-verified Criteria_alg is enabled with up to {} "
                    "endpoint correction solve(s); an unmet endpoint forces "
                    "the next Newton linear solve to the strict tolerance.",
                    this->param_.aposteriori_alg_max_resolves_));
        }

        if (terminal_output) {
            OpmLog::info(fmt::format(
                "Evaluating a posteriori eta_sp / eta_time estimators "
                "(--enable-aposteriori-estimators=true; diagnostic only). "
                "Space/time band: [{}, {}], weight exponent l={}, lifted "
                "relperm={}, bubble correction={}, MB floor={:.3e}.",
                aposteriori_targets_.gammaTime, aposteriori_targets_.GammaTime,
                this->param_.aposteriori_weight_exponent_,
                this->param_.aposteriori_use_lifted_relperm_,
                this->param_.aposteriori_use_bubble_correction_,
                aposteriori_targets_.tolMB));
        }
    }
    // TODO: remember to fix!
    if (this->param_.nonlinear_solver_ == "nldd") {
        if (terminal_output) {
            OpmLog::info("Using Non-Linear Domain Decomposition solver (nldd).");
        }
        nlddSolver_ = std::make_unique<NonlinearSystemNldd<TypeTag>>(*this);
    } else if (this->param_.nonlinear_solver_ == "newton") {
        if (terminal_output) {
            OpmLog::info("Using Newton nonlinear solver.");
        }
    } else {
        OPM_THROW(std::runtime_error, "Unknown nonlinear solver option: " +
                                      this->param_.nonlinear_solver_);
    }
}

template <class TypeTag>
SimulatorReportSingle
NonlinearSystemBlackOilReservoir<TypeTag>::
prepareStep(const SimulatorTimerInterface& timer)
{
    OPM_TIMEFUNCTION();
    auto report = ParentType::prepareStep(timer);

    Dune::Timer perfTimer;
    perfTimer.start();

    unsigned numDof = this->simulator_.model().numGridDof();
    wasSwitched_.resize(numDof);
    std::fill(wasSwitched_.begin(), wasSwitched_.end(), false);

    if (this->param_.update_equations_scaling_) {
        OpmLog::error("Equation scaling not supported");
        //updateEquationsScaling();
    }

    if (hasNlddSolver()) {
        nlddSolver_->prepareStep();
    }

    report.pre_post_time += perfTimer.stop();

    auto getIdx = [](unsigned phaseIdx) -> int
    {
        if (FluidSystem::phaseIsActive(phaseIdx)) {
            const unsigned sIdx = FluidSystem::solventComponentIndex(phaseIdx);
            return FluidSystem::canonicalToActiveCompIdx(sIdx);
        }

        return -1;
    };
    const auto& schedule = this->simulator_.vanguard().schedule();
    auto& rst_conv = this->simulator_.problem().eclWriter().mutableOutputModule().getConv();
    rst_conv.init(this->simulator_.vanguard().globalNumCells(),
                  schedule[timer.reportStepNum()].rst_config(),
                  {getIdx(FluidSystem::oilPhaseIdx),
                   getIdx(FluidSystem::gasPhaseIdx),
                   getIdx(FluidSystem::waterPhaseIdx),
                   contiPolymerEqIdx,
                   contiBrineEqIdx,
                   contiSolventEqIdx});

    return report;
}

template <class TypeTag>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
initialLinearization(SimulatorReportSingle& report,
                     const int minIter,
                     const int maxIter,
                     const SimulatorTimerInterface& timer)
{
    aposteriori_well_linearized_early_ = false;
    ParentType::initialLinearization(report,
                                     minIter,
                                     maxIter,
                                     timer);                                 

    // -----------   Check if converged   -----------
    std::vector<Scalar> residual_norms;
    Dune::Timer perfTimer;
    perfTimer.reset();
    perfTimer.start();
    // the step is not considered converged until at least minIter iterations is done
    {
        auto convrep = getConvergence(timer, maxIter, residual_norms);
        report.converged = convrep.converged() &&
                           this->simulator_.problem().iterationContext().iteration() >= minIter;
        const auto severity = convrep.severityOfWorstFailure();

        // Complete eta_sp with the local componentwise FV balance defect at
        // the assembled current state. Do this before well elimination: the
        // reservoir residual then contains the physical accumulation, flux,
        // NNC, and well-source balance in each cell. The global signed OPM MB
        // check remains a separate conservation safeguard.
        if (aposteriori_estimator_
            && aposteriori_estimator_->hasPendingEquilibrationDefect()) {
            aposteriori_estimator_->completeEquilibrationDefect(
                this->simulator().model().linearizer().residual(),
                timer.currentStepLength());
            if (!aposteriori_rows_.empty()) {
                aposteriori_rows_.back()[0] =
                    aposteriori_estimator_->etaSpatialMimetic();
                aposteriori_rows_.back()[7] =
                    aposteriori_estimator_->etaEquilibration();
            }
        }

        // Complete the previous update's eta_lin with the nonlinear
        // well/NNC-source Taylor remainder. The current reservoir and well
        // equations have now been assembled at chi^k. Applying the well Schur
        // complement puts this residual on the same reduced-system footing as
        // the stored linear prediction R^{k-1}-J_eff dx.
        if (this->param_.enable_aposteriori_newton_stopping_
            && aposteriori_estimator_
            && aposteriori_estimator_->hasPendingSourceLinearizationDefect()) {
            this->wellModel().linearize(
                this->simulator().model().linearizer().jacobian(),
                this->simulator().model().linearizer().residual());
            aposteriori_well_linearized_early_ = true;
            aposteriori_estimator_->completeSourceLinearizationDefect(
                this->simulator().model().linearizer().residual(),
                timer.currentStepLength());
            if (!aposteriori_rows_.empty()) {
                aposteriori_rows_.back()[5] =
                    aposteriori_estimator_->etaLinearization();
            }
            if (aposteriori_estimator_->linearizationAvailable()) {
                this->aposteriori_eta_lin_prev_ =
                    aposteriori_estimator_->etaLinearization();
                const Scalar etaSpDarcy =
                    this->param_.aposteriori_use_total_spatial_budget_
                    ? aposteriori_estimator_->etaSpatialMimetic()
                    : aposteriori_estimator_->etaSpatialDarcyMimetic();
                const Scalar etaTime =
                    aposteriori_estimator_->temporalAvailable()
                        ? aposteriori_estimator_->etaTemporal()
                        : etaSpDarcy;
                this->aposteriori_max_sptime_prev_ =
                    std::max(etaSpDarcy, etaTime);
            }
        }

        // The estimator values were evaluated immediately after the preceding
        // Newton update, so they describe the state whose residual and MB were
        // just assembled above. Replace CNV only; all non-reservoir safeguards
        // remain standard OPM convergence requirements.
        if (!report.converged &&
            this->param_.enable_aposteriori_newton_stopping_ &&
            aposteriori_estimator_ &&
            !aposteriori_rows_.empty() &&
            this->simulator_.problem().iterationContext().iteration() >= minIter &&
            // never accept on the first Newton iteration (first solve)
            this->simulator_.problem().iterationContext().iteration() >= 2 &&
            severity <= ConvergenceReport::Severity::Normal)
        {
            // Which reservoir failures the a posteriori criterion is allowed
            // to override: always CNV (its whole purpose -- replace the CNV
            // discretization-error proxy with the weighted estimate). Also MB
            // ONLY when --aposteriori-tol-mb explicitly relaxes the MB
            // tolerance below the simulator's own: in that case the a
            // posteriori newtonConverged() gate (|MB| <= aposteriori tolMB)
            // becomes the MB arbiter instead of OPM's strict check. With the
            // default tol (aposteriori tolMB == tolerance_mb_) an MB failure
            // here still fails newtonConverged(), so this is a no-op then --
            // OPM's non-negotiable MB is preserved unless deliberately
            // loosened. (Copilot's earlier "MB=1" experiment did exactly this
            // loosening; see AposterioriTolMb's doc comment for the tradeoff.)
            const bool mbOverrideAllowed =
                !this->param_.aposteriori_separate_neumann_mean_ &&
                this->param_.aposteriori_tol_mb_ > this->param_.tolerance_mb_;
            const auto& failures = convrep.reservoirFailures();
            const bool onlyCnvFailures = !failures.empty() &&
                std::all_of(failures.begin(), failures.end(), [mbOverrideAllowed](const auto& failure) {
                    using T = ConvergenceReport::ReservoirFailure::Type;
                    return failure.type() == T::Cnv
                        || (mbOverrideAllowed && failure.type() == T::MassBalance);
                });
            const bool nonReservoirChecksPassed =
                !convrep.wellFailed() &&
                !convrep.wellGroupTargetsViolated() &&
                !convrep.networkNeedsMoreBalancing();

            // Plateau guard: Criteria_newton compares eta_lin against a
            // discretization-error estimate that the theory assumes is
            // iterate-independent. Empirically eta_sp(mim)/eta_time settle
            // within ~4 Newton iterations on a well-posed step, but can swing
            // by 100s of % while Newton oscillates (well control switch,
            // over-large step). Only accept once both have stabilised to
            // within the configured tolerance between the last two computed
            // iterates. Zero exposes the bare estimator criterion for a
            // reproducible ablation instead of relying on process state.
            const Scalar plateauTol =
                this->param_.aposteriori_newton_plateau_tolerance_;
            bool estimatorsPlateaued = plateauTol <= Scalar{0};
            if (!estimatorsPlateaued && aposteriori_rows_.size() >= 2) {
                const auto& rNow  = aposteriori_rows_[aposteriori_rows_.size() - 1];
                const auto& rPrev = aposteriori_rows_[aposteriori_rows_.size() - 2];
                const auto relChange = [](Scalar cur, Scalar prev) -> Scalar {
                    if (prev > Scalar{0}) return std::abs(cur - prev) / prev;
                    return (cur > Scalar{0}) ? Scalar{1} : Scalar{0};
                };
                const Scalar dMim  = relChange(rNow[0], rPrev[0]); // eta_sp(mim)
                const Scalar dTime = relChange(rNow[3], rPrev[3]); // eta_time
                estimatorsPlateaued = (dMim <= plateauTol) && (dTime <= plateauTol);
            }

            const Scalar etaSpMim =
                this->param_.aposteriori_use_total_spatial_budget_
                ? aposteriori_estimator_->etaSpatialMimetic()
                : aposteriori_estimator_->etaSpatialDarcyMimetic();
            const Scalar etaTime = aposteriori_estimator_->temporalAvailable()
                ? aposteriori_estimator_->etaTemporal() : etaSpMim;
            const bool estimatorPassed =
                aposteriori_estimator_->linearizationAvailable() &&
                APosteriori::newtonConverged(aposteriori_estimator_->etaLinearization(),
                                             etaSpMim,
                                             etaTime,
                                             this->last_mass_balance_residual_,
                                             aposteriori_targets_);
            // Opt-in validation trace: report every constituent of the
            // Criteria_newton decision without changing acceptance. This is
            // intentionally environment-gated because per-iteration output is
            // too verbose for normal simulator use.
            if (std::getenv("OPM_APOST_TRACE_REJECT") && !this->grid_.comm().rank()) {
                std::size_t cnvFailures = 0;
                std::size_t mbFailures = 0;
                std::size_t otherReservoirFailures = 0;
                for (const auto& failure : failures) {
                    using T = ConvergenceReport::ReservoirFailure::Type;
                    if (failure.type() == T::Cnv) {
                        ++cnvFailures;
                    }
                    else if (failure.type() == T::MassBalance) {
                        ++mbFailures;
                    }
                    else {
                        ++otherReservoirFailures;
                    }
                }
                OpmLog::info(fmt::format(
                    "  [a posteriori reject trace] iter={} cnv_fail={} mb_fail={} "
                    "other_res_fail={} only_cnv={} nonreservoir_ok={} estimator_ok={} "
                    "plateau_ok={} alg_ok={} eta_lin={:.3e} budget={:.3e} MB={:.3e} "
                    "MB_tol={:.3e}",
                    this->simulator_.problem().iterationContext().iteration(),
                    cnvFailures,
                    mbFailures,
                    otherReservoirFailures,
                    onlyCnvFailures,
                    nonReservoirChecksPassed,
                    estimatorPassed,
                    estimatorsPlateaued,
                    !aposteriori_alg_unmet_,
                    aposteriori_estimator_->etaLinearization(),
                    this->param_.aposteriori_gamma_lin_ * std::max(etaSpMim, etaTime),
                    this->last_mass_balance_residual_,
                    aposteriori_targets_.tolMB));
            }
            if (onlyCnvFailures && nonReservoirChecksPassed && estimatorPassed
                && estimatorsPlateaued && !aposteriori_alg_unmet_) {
                report.converged = true;
                if (!this->grid_.comm().rank()) {
                    OpmLog::info(fmt::format(
                        "  [a posteriori] Criteria_newton accepted iteration {} "
                        "(eta_lin={:.3e}, eta_lin_source={:.3e}, "
                        "eta_eq={:.3e}, max(eta_sp,D,eta_time)={:.3e}, MB={:.3e})",
                        this->simulator_.problem().iterationContext().iteration(),
                        aposteriori_estimator_->etaLinearization(),
                        aposteriori_estimator_->etaLinearizationSource(),
                        aposteriori_estimator_->etaEquilibration(),
                        std::max(etaSpMim, etaTime),
                        this->last_mass_balance_residual_));
                }
            }
        }
        if (report.converged &&
            convrep.cnvRelaxSource() != ConvergenceReport::CnvRelaxSource::None)
        {
            ++report.relaxed_cnv_acceptances;
        }
        this->convergence_reports_.back().report.push_back(std::move(convrep));

        // Throw if any NaN or too large residual found.
        if (severity == ConvergenceReport::Severity::NotANumber) {
            this->failureReport_ += report;
            OPM_THROW_PROBLEM(NumericalProblem, "NaN residual found!");
        } else if (severity == ConvergenceReport::Severity::TooLarge) {
            this->failureReport_ += report;
            OPM_THROW_NOLOG(NumericalProblem, "Too large residual found!");
        } else if (severity == ConvergenceReport::Severity::ConvergenceMonitorFailure) {
            this->failureReport_ += report;
            OPM_THROW_PROBLEM(ConvergenceMonitorFailure,
                              fmt::format(
                                  "Total penalty count exceeded cut-off-limit of {}",
                                  this->param_.monitor_params_.cutoff_
                              ));
        }
    }
    report.update_time += perfTimer.stop();
    this->residual_norms_history_.push_back(residual_norms);

}

template <class TypeTag>
template <class NonlinearSolverType>
SimulatorReportSingle
NonlinearSystemBlackOilReservoir<TypeTag>::
nonlinearIteration(const SimulatorTimerInterface& timer,
                   NonlinearSolverType& nonlinear_solver)
{
    // Model-level timestep initialization (once per timestep).
    // markTimestepInitialized() is called later in initialLinearization(),
    // after assembleReservoir() has triggered the well model's prepareTimeStep().
    if (this->simulator_.problem().iterationContext().needsTimestepInit()) {
        this->residual_norms_history_.clear();
        this->conv_monitor_.reset();
        this->current_relaxation_ = 1.0;
        this->dx_old_ = 0.0;
        this->adaptive_linear_reduction_.reset();
        this->aposteriori_rows_.clear();
        this->aposteriori_alg_rhs_.clear();
        this->aposteriori_eta_lin_prev_ = 0;
        // Keep the last converged discretization budget for the first linear
        // solve of the new timestep. Resetting it to zero forced one strict
        // solve per timestep even though the previous accepted state supplies
        // the only available predictor for Criteria_alg. Invalid/non-positive
        // values are still rejected at the forcing-term call site.
        this->aposteriori_alg_unmet_ = false;
        this->aposteriori_force_strict_linear_ = false;
        if (this->aposteriori_estimator_) {
            this->aposteriori_estimator_->resetNewtonIterateHistory();
        }
        this->convergence_reports_.push_back({timer.reportStepNum(), timer.currentStepNum(), {}});
        this->convergence_reports_.back().report.reserve(11);
    }

    SimulatorReportSingle result;
    if (this->param_.nonlinear_solver_ != "nldd") {
        result = this->nonlinearIterationNewton(timer, nonlinear_solver);
    }
    else {
        result = this->nlddSolver_->nonlinearIterationNldd(timer, nonlinear_solver);
    }

    auto& rst_conv = this->simulator_.problem().eclWriter().mutableOutputModule().getConv();
    rst_conv.update(this->simulator_.model().linearizer().residual());

    this->simulator_.problem().advanceIteration();

    // OPM_APOST_ITERATION_DIAGNOSTICS records the counterfactual estimator
    // stopping points while leaving OPM's standard Newton criterion in charge.
    // --enable-aposteriori-linear-tolerance also needs this per-iteration
    // refresh: its forcing term is Gamma_alg*max(eta_sp,eta_time)/eta_alg(0),
    // and without it eta_sp/eta_time (via aposteriori_max_sptime_prev_) only
    // update once per TIMESTEP (at final convergence), not once per Newton
    // iteration. Within a timestep eta_alg(0) legitimately shrinks toward
    // zero as Newton converges, so a target built from a stale, larger
    // max(eta_sp,eta_time) can saturate at the configured loosest reduction
    // for the entire back half of every Newton loop, regardless of Gamma_alg
    // -- diagnosed
    // 2026-09-14 on the L40 five-spot (linear-tolerance alone: 569 Newton
    // iterations over 44 fixed 5-day steps vs. 198 with estimators off
    // entirely, identical across Gamma_alg in {0.1,0.05,0.01}).
    const bool perIterEval =
        this->param_.enable_aposteriori_newton_stopping_
        || this->param_.enable_aposteriori_linear_tolerance_
        || std::getenv("OPM_APOST_ITERATION_DIAGNOSTICS") != nullptr;
    if (aposteriori_estimator_ &&
        (result.converged ||
         (perIterEval &&
          this->simulator_.problem().iterationContext().iteration()
              >= this->param_.aposteriori_first_eval_iter_))) {
        evalAposterioriEstimators(timer, result.converged);
    }
    return result;
}

template <class TypeTag>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
evalAposterioriEstimators(const SimulatorTimerInterface& timer, const bool converged)
{
    const Scalar dt = timer.currentStepLength();

    // Runaway-guard substep counter resets per REPORT period (needsTimestepInit
    // fires per substep, so it cannot be reset there). Suspension is local to
    // one report period: a large total number of accepted steps is not evidence
    // that eta_sp is defective, and must not silently disable estimator control
    // later in an otherwise well-behaved run.
    if (timer.reportStepNum() != aposteriori_last_report_step_) {
        aposteriori_last_report_step_ = timer.reportStepNum();
        aposteriori_period_steps_ = 0;
        aposteriori_ctrl_suspended_ = false;
    }

    // Refresh the near-well cell list from the *current* well model every
    // call: wells open/close over the schedule, and this was never wired at
    // all previously -- setWellCells() existed but nothing ever called it, so
    // the near-well weight D_K^l was dormant even when --aposteriori-weight-l
    // was set. Cheap (perforation counts are small relative to grid size).
    {
        std::vector<int> wellCells;
        for (const auto* w : this->wellModel().genericWells()) {
            for (int c : w->cells()) {
                wellCells.push_back(c);
            }
        }
        // A SOURCE-keyword cell is a flux singularity too -- include it in the
        // near-singularity distance set so the D_K^{l/2} weight (eq. eps_norm,
        // --aposteriori-weight-exponent) is measured from sources as well as
        // wells. With l>0 this moves the marking OFF the source onto the front.
        {
            const auto& sched = this->simulator_.vanguard().schedule();
            const auto& cim = this->simulator_.vanguard().cartesianIndexMapper();
            const auto& cd  = cim.cartesianDimensions();
            const long NXc = cd[0], NYc = cd[1];
            for (const auto& [ijk, sc] :
                 sched[timer.reportStepNum()].source()) {
                static_cast<void>(sc);
                const int comp = this->simulator_.vanguard().compressedIndexForInterior(
                    static_cast<int>((static_cast<long>(ijk[2]) * NYc + ijk[1]) * NXc + ijk[0]));
                if (comp >= 0) wellCells.push_back(comp);
            }
        }
        aposteriori_estimator_->setWellCells(std::move(wellCells));
    }

    // Schedule-wide protected mask (built once): well completions + future
    // connections + per-vertical-well k-spans + exact SOURCE cells, optionally
    // dilated by OPM_APOST_PROTECT_HALO. ONE builder shared with the driver
    // preflight (AdaptiveRefinementProtection.hpp) so both see the same set.
    if (!aposteriori_protected_built_) {
        aposteriori_protected_built_ = true;
        const auto& schedule = this->simulator_.vanguard().schedule();
        const auto& cd = this->simulator_.vanguard().cartesianIndexMapper()
                             .cartesianDimensions();
        int phalo = 0;
        if (const char* h = std::getenv("OPM_APOST_PROTECT_HALO"))
            phalo = std::max(0, std::atoi(h));
        aposteriori_estimator_->setProtectedRefinementCells(
            buildProtectedRefinementCells(schedule,
                {static_cast<int>(cd[0]), static_cast<int>(cd[1]),
                 static_cast<int>(cd.size() > 2 ? cd[2] : 1)}, phalo));
    }

    // Evaluate eta_sp / eta_time for the current Newton iterate. The final
    // convergence check has already completed eta_eq for the row produced
    // after the preceding update, which represents this same state; do not
    // replace it with a newly pending row whose balance residual would never
    // be assembled. When estimators were not evaluated per iteration, build
    // the sole converged row here and complete it immediately from the current
    // (uneliminated) nonlinear reservoir residual.
    const bool reuseCompletedConvergedRow =
        converged && !aposteriori_rows_.empty()
        && !aposteriori_estimator_->hasPendingEquilibrationDefect();
    if (!reuseCompletedConvergedRow) {
        aposteriori_estimator_->compute(dt, /*commitHistory=*/false);
        if (converged) {
            aposteriori_estimator_->completeEquilibrationDefect(
                this->simulator().model().linearizer().residual(), dt);
        }
    }

    // etaSpMim is the MVEM (mimetic virtual element) flux-energy replacement
    // for T1+T3, following Vohralik & Yousef CMAME 331 (2018) Sections 3 and
    // 6 (Definition 3.3, Lemmas 3.5-3.7, Remark 3.15): a consistency term
    // M^c (using the FULL permeability tensor via an SPD solve, reproducing
    // T1^2 exactly) plus a stability term M^s (a face-DOF-projector kernel
    // measure, replacing T3's c_KK^{-1/2}*|div| surrogate). This SUPERSEDES
    // an earlier, broken attempt (2026-09-05) that divided the raw defect by
    // OPM's own per-connection trans_f directly -- that conflated the
    // consistency and kernel parts under a single weight and, worse, trans_f
    // carries deck multipliers (NTG etc.) inconsistent with T1/T3's raw-K
    // basis, producing an aggregate LARGER than even the discredited T1+T3
    // (2.1e11 vs 3.6e10 on a traced step). The corrected construction
    // (mvemFluxMassMatrix, unit-tested: N^T C/|K|=I consistency,
    // constant-flux reproduction, and exact match to the Pi0 formula for the
    // consistency term) instead behaves as the theory predicts -- on SPE9,
    // etaSpMim tracks 10-20% above etaSpT1 (never a blowup) and stays a
    // consistent 3-4x below the discredited etaSp (T1+T3) across steps.
    // etaSpMim is what Criteria_space_time_balance and
    // --enable-aposteriori-timestep-control actually use; etaSpT1 and etaSp
    // (T1+T3) remain for reporting/comparison. This still covers ONLY the
    // flux-energy term of the paper's full three-term Theorem 3.12 estimate,
    // and the M^s stability scaling (epsilon=1e-2 default) is the one
    // MFD-family design choice per Lemma 3.7 -- see the doc comments at
    // mvemFluxMassMatrix and etaSpatialMimetic() for both caveats in full.
    const Scalar etaSpMim  = aposteriori_estimator_->etaSpatialMimetic();
    const Scalar etaSpControl =
        this->param_.aposteriori_use_total_spatial_budget_
        ? etaSpMim : aposteriori_estimator_->etaSpatialDarcyMimetic();
    const Scalar etaSpT1   = aposteriori_estimator_->etaSpatialT1();
    const Scalar etaSp     = aposteriori_estimator_->etaSpatial();
    const Scalar etaTime   = aposteriori_estimator_->etaTemporal();
    const bool   haveT     = aposteriori_estimator_->temporalAvailable();
    const Scalar etaLinW   = aposteriori_estimator_->etaLinearization();
    const bool   haveLinW  = aposteriori_estimator_->linearizationAvailable();

    // Snapshot max(eta_sp,eta_time) for the next linear solve's Criteria_alg
    // forcing term (--enable-aposteriori-linear-tolerance): the linear solve
    // is driven only until the algebraic error is small vs. the
    // discretization error. eta_sp/eta_time just came from compute() above
    // unconditionally, so this must NOT be gated on eta_lin's own
    // availability (haveLinW) -- eta_lin is a Newton-stopping-only quantity
    // (recordLinearizationDefect() only runs when
    // --enable-aposteriori-newton-stopping is set). Coupling the two meant
    // --enable-aposteriori-linear-tolerance used WITHOUT Newton-stopping left
    // aposteriori_max_sptime_prev_ at its reset value of 0 forever, so
    // forcingTermFromEstimator's own "maxSpTime <= 0" guard silently fell
    // back to the 0.5 stability-ceiling constant on every solve regardless of
    // Gamma_alg -- diagnosed 2026-09-13 (330 timesteps / 2663 Newton on the
    // L40 five-spot vs. 47 / 198 with estimators off entirely, identical
    // across Gamma_alg in {0.1,0.05,0.01}).
    this->aposteriori_max_sptime_prev_ =
        std::max(etaSpControl, haveT ? etaTime : etaSpControl);

    if (haveLinW && std::isfinite(etaLinW)) {
        this->aposteriori_eta_lin_prev_ = etaLinW;
    }

    // Level-0 proxy for eta_lin: the largest per-component convergence measure
    // of the current iterate (CNV, cf. Remark rem:cnv in the paper).  NOT
    // dimensionally comparable to eta_sp/eta_time (a weighted energy norm) --
    // kept only as a familiar reference column; etaLinW is the one used below
    // to evaluate Criteria_newton.
    Scalar etaLinCnv = 0.0;
    if (!this->residual_norms_history_.empty()) {
        for (Scalar r : this->residual_norms_history_.back()) {
            etaLinCnv = std::max(etaLinCnv, std::abs(r));
        }
    }

    if (!reuseCompletedConvergedRow) {
        aposteriori_rows_.push_back(
            {etaSpMim, etaSpT1, etaSp, haveT ? etaTime : Scalar{0}, etaLinCnv,
             haveLinW ? etaLinW : Scalar{0},
             aposteriori_estimator_->etaAlgebraic(),
             aposteriori_estimator_->etaEquilibration()});
    }

    // Spatial map dump: set OPM_APOST_DUMP_STEP=<substep number> to write, for
    // every Newton iteration of that substep, a per-cell CSV
    // (cell, x, y, z, eta_sp, eta_time, eta_lin, eta_alg) named
    // apost_cells_step<n>_it<k>.csv. eta_lin -> 0 at Newton convergence by
    // construction, so an early iterate (it1/it2) shows its spatial structure.
    // OPM_APOST_DUMP_STEP=<n> matches timer.currentStepNum(); =-1 (or "last")
    // overwrites two fixed files on EVERY estimator call, so after the run they
    // hold the FINAL substep's field -- robust on TSTEP decks where
    // currentStepNum() is not the report index:
    //   apost_cells_last.csv        -- latest iterate  (eta_sp/time/alg; eta_lin~0)
    //   apost_cells_last_early.csv  -- first Newton iterate of the step (eta_lin structure)
    if (const char* s = std::getenv("OPM_APOST_DUMP_STEP")) {
        const bool everyStep = (std::string(s) == "last" || std::atoi(s) < 0);
        if (everyStep) {
            aposteriori_estimator_->dumpCellEstimators("apost_cells_last.csv");
            if (aposteriori_rows_.size() <= 1)
                aposteriori_estimator_->dumpCellEstimators("apost_cells_last_early.csv");
        }
        else if (std::atoi(s) == timer.currentStepNum()) {
            aposteriori_estimator_->dumpCellEstimators(
                fmt::format("apost_cells_step{}_it{}.csv",
                            timer.currentStepNum(), aposteriori_rows_.size()));
        }
    }

    // --- Algorithm 6.1, spatial mesh-refinement marking step ---------------
    // eta_sp,K >= zeta_ref  * max_K eta_sp,K  -> mark REFINE
    // eta_sp,K <= zeta_deref * max_K eta_sp,K -> mark DEREFINE
    // OPM's black-oil Flow has no solution-driven dynamic AMR, so this reports
    // the refine/derefine cell sets (and whether the balancing loop's spatial
    // condition is met) rather than adapting the grid. zeta from env, default
    // 0.5 / 0.1. Set OPM_APOST_REFINE_DUMP=1 for a per-step marks CSV.
    if (converged) {
        const Scalar zRef = []() {
            const char* s = std::getenv("OPM_APOST_ZETA_REF");
            return s ? static_cast<Scalar>(std::atof(s)) : Scalar{0.5};
        }();
        const Scalar zDrf = []() {
            const char* s = std::getenv("OPM_APOST_ZETA_DEREF");
            return s ? static_cast<Scalar>(std::atof(s)) : Scalar{0.1};
        }();
        const auto marks = aposteriori_estimator_->refinementMarks(zRef, zDrf);
        const int nRef = marks.first;
        const int nDrf = marks.second;
        if (!this->grid_.comm().rank() && this->terminalOutputEnabled()) {
            OpmLog::info(fmt::format(
                "  [Algorithm 6.1] spatial balancing: {} cell(s) >= {:.2f}*max(eta_sp,K) "
                "(REFINE), {} cell(s) <= {:.2f}*max (DEREFINE) -- {} "
                "[indicator only: no dynamic AMR in black-oil Flow]",
                nRef, static_cast<double>(zRef), nDrf, static_cast<double>(zDrf),
                (nRef == 0) ? "spatial mesh accepted"
                            : "spatial refinement WOULD trigger a re-solve"));
        }
        if (std::getenv("OPM_APOST_REFINE_DUMP")) {
            aposteriori_estimator_->dumpRefinementMarks(
                fmt::format("apost_refine_step{}.csv", timer.currentStepNum()),
                zRef, zDrf);
        }

    }

    // Diagnostic only: does Criteria_newton -- now dimensionally consistent,
    // eta_lin and max(eta_sp_mim,eta_time) both in the *,K energy norm, and
    // the MB floor matched to the simulator's own tolerance_mb_ -- agree with
    // the point where OPM's own Newton loop actually stopped?  This does NOT
    // drive anything; it is logged for comparison in the final table row.
    const bool wouldStopNewton = haveLinW &&
        APosteriori::newtonConverged(
            etaLinW, etaSpControl, haveT ? etaTime : etaSpControl,
                                     this->last_mass_balance_residual_, aposteriori_targets_);

    if (!converged) {
        if (!this->grid_.comm().rank() && wouldStopNewton) {
            OpmLog::info(fmt::format(
                "  [a posteriori, diagnostic only] Criteria_newton would already be "
                "satisfied at k={} (eta_lin={:.3e} <= Gamma_lin*max(eta_sp,eta_time)); "
                "OPM's own Newton loop continues.", aposteriori_rows_.size(), etaLinW));
        }
        return;
    }

    // --- flush the per-Newton-iteration table for this step ---
    // Control uses etaSpMim throughout -- see its declaration above.
    const Scalar ratio = (etaSpControl > 0.0 && haveT) ? etaTime / etaSpControl
                                                : std::numeric_limits<Scalar>::quiet_NaN();
    // The estimator-driven rescale needs a usable spatial reference: a
    // non-finite or collapsed eta_sp(mim) (e.g. from incomplete cell-face
    // geometry) must leave dt untouched, never drive it.
    const bool discOk = std::isfinite(etaSpControl) && etaSpControl > Scalar{0}
                        && (!haveT || std::isfinite(etaTime));
    const bool   band  = !haveT || !discOk
        || APosteriori::spaceTimeBalanced(
            etaSpControl, etaTime, aposteriori_targets_);
    // In limiter mode (the production default -- see
    // aposterioriTimestepGrowthOverrideEnabled()), the caller applies
    // min(nativeDt, dtNew), which already bounds growth by whatever
    // AdaptiveTimeStepping's own heuristic allows; an independent maxGrow
    // cap here would be redundant at best and, if tighter than native's own
    // allowance (as the conservative maxGrow=1.25 default is), would still
    // needlessly shrink relative to native even when eta_time is nowhere
    // near excessive -- so growth is left uncapped here in that mode. Shrink
    // is capped in both modes: an unbounded single-rescale cut is not
    // something min(native, ...) protects against either way.
    const Scalar effMaxGrow = this->param_.aposteriori_timestep_growth_override_
        ? this->param_.aposteriori_max_grow_
        : std::numeric_limits<Scalar>::max();
    const Scalar dtNew = (haveT && discOk)
        ? APosteriori::rescaledTimeStep(
            dt, etaSpControl, etaTime, aposteriori_targets_,
                                        /*dtMin=*/Scalar{0},
                                        /*dtMax=*/std::numeric_limits<Scalar>::max(),
                                        effMaxGrow,
                                        this->param_.aposteriori_max_shrink_)
        : dt;

    if (this->param_.enable_aposteriori_timestep_control_ && haveT && discOk) {
        // Runaway guard: a well-behaved report period needs at most a few tens
        // of estimator-driven substeps. Far more means the estimator is
        // over-refining -- its eta_sp is almost certainly under-counted
        // (incomplete cell-face geometry, NNC, non-Cartesian cells) rather than
        // the step genuinely being that small. Suspend the override for the
        // rest of the period and let OPM's native controller finish it.
        constexpr int kMaxPeriodSteps = 40;
        ++aposteriori_period_steps_;
        ++aposteriori_total_ctrl_steps_;
        if (!aposteriori_ctrl_suspended_
            && aposteriori_period_steps_ > kMaxPeriodSteps) {
            aposteriori_ctrl_suspended_ = true;
            if (!this->grid_.comm().rank())
                OpmLog::warning(fmt::format(
                    "  [a posteriori] estimator-driven time-step control DISABLED "
                    "for the rest of report period {} at substep {} (this-period {}, "
                    "eta_time/eta_sp = {:.2f}) -- eta_sp is under-counted for this "
                    "grid (incomplete face geometry / NNC / non-Cartesian cells). "
                    "Control will resume at the next report boundary.",
                    timer.reportStepNum(), aposteriori_total_ctrl_steps_,
                    aposteriori_period_steps_,
                    etaSpControl > 0
                        ? static_cast<double>(etaTime / etaSpControl) : 0.0));
        }
        if (aposteriori_ctrl_suspended_)
            aposteriori_suggested_dt_.reset();
        else
            aposteriori_suggested_dt_ = dtNew;
    }

    if (!this->grid_.comm().rank()) {
        std::ostringstream os;
        os << fmt::format(
            "\n  a posteriori error components -- step {}  (dt = {:.4g} d, rescale -> {:.4g} d, "
            "space/time {}{})\n",
            timer.currentStepNum(), dt / (24.0 * 3600.0), dtNew / (24.0 * 3600.0),
            band ? "in band" : "OUT of band",
            this->param_.enable_aposteriori_timestep_control_ ? ", DRIVING next dt" : "");
        os << "    k |eta_sp(total)| eta_eq     | eta_sp(T1) |eta_sp(T1+T3)|  eta_time  | eta_lin(CNV)| eta_lin(wtd)|eta_alg(rhs)|eta_alg(post)| d(sp) d(tm)\n";
        os << "  ---------------------------------------------------------------------------------------------------------------------------------------\n";
        for (std::size_t k = 0; k < aposteriori_rows_.size(); ++k) {
            const auto& r = aposteriori_rows_[k];
            Scalar dSp = 0.0, dTm = 0.0;
            if (k > 0) {
                const auto& p = aposteriori_rows_[k - 1];
                if (p[0] > 0.0) dSp = std::abs(r[0] - p[0]) / p[0];
                if (p[3] > 0.0) dTm = std::abs(r[3] - p[3]) / p[3];
            }
            // aposteriori_rows_[k] is pushed by evalAposterioriEstimators(),
            // gated to start at iteration aposteriori_first_eval_iter_
            // (default 1) -- so aposteriori_rows_[0] is Newton iteration 1's
            // row. aposteriori_alg_rhs_ is pushed unconditionally (whenever
            // the current iteration is not yet converged), so
            // aposteriori_alg_rhs_[0] is iteration 0's value. The two vectors
            // are therefore offset by exactly one iteration throughout, and
            // aposteriori_rows_[k]'s matching entry is aposteriori_alg_rhs_[k+1]
            // (iteration k+1), not aposteriori_alg_rhs_[k] (iteration k) --
            // diagnosed 2026-09-14 via an exact numeric match between the
            // displayed eta_alg(rhs) at row k+1 and eta_eq at row k (both
            // reflect the same underlying iterate).
            const Scalar etaAlgRhs = (k + 1) < aposteriori_alg_rhs_.size()
                ? aposteriori_alg_rhs_[k + 1]
                : std::numeric_limits<Scalar>::quiet_NaN();
            os << fmt::format("  {:3d} | {:11.3e} | {:10.3e} | {:10.3e} | {:11.3e} | {:10.3e} | {:11.3e} | {:11.3e} | {:10.3e} | {:11.3e} | {:5.1f}% {:5.1f}%\n",
                              k + 1, r[0], r[7], r[1], r[2], r[3], r[4], r[5], etaAlgRhs, r[6],
                              100.0 * dSp, 100.0 * dTm);
        }
        const char* etaSpComposition = this->param_.aposteriori_separate_neumann_mean_
            ? "  eta_sp(total) is the mimetic Darcy defect; eta_eq is reported separately (H1/R mode);\n"
            : "  eta_sp(total) is the local sum of the mimetic Darcy defect and eta_eq;\n";
        os << fmt::format("  ----------------------------------------------------------------------------------------------\n"
                          "  eta_sp(total) / eta_time should plateau while eta_lin(wtd) falls "
                          "=> spatial/temporal error is split out.  (ratio eta_time/eta_sp(mim) = {:.3f})\n"
                          "{}"
                          "  eta_eq is the unsigned componentwise FV balance residual in the weighted\n"
                          "  Neumann dual norm (OPM's global signed MB remains a separate safeguard).\n"
                          "  The Darcy part is the flux-energy replacement for T1+T3 (eq. 3.13,\n"
                          "  Vohralik & Yousef CMAME 2018),\n"
                          "  using the lowest-order mimetic/VEM flux mass matrix M_K = M^c + M^s\n"
                          "  (full permeability tensor). eta_sp(T1) and\n"
                          "  eta_sp(T1+T3) are reported for comparison only (T3 alone is an uncertified\n"
                          "  surrogate -- see APosterioriSpatialTemporalEstimator::compute()).\n"
                          "  eta_lin(CNV) is a familiar but dimensionally-inconsistent reference only; "
                          "eta_lin(wtd) combines geometric-face flux, storage, and reduced-balance\n"
                          "  well/NNC source Taylor defects.",
                          haveT ? ratio : 0.0, etaSpComposition);
        OpmLog::info(os.str());
    }
}

template <class TypeTag>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
acceptAposterioriStep(const SimulatorTimerInterface& timer)
{
    if (!aposteriori_estimator_)
        return;

    // Newton convergence is not sufficient: NonlinearSolver may still reject
    // the step using its relative-change timestep acceptance test. Commit all
    // accepted-step-only estimator state only after that test passes.
    aposteriori_estimator_->commitTemporalHistory();
    aposteriori_estimator_->accumulateSpatialEnergy();

    if (!this->grid_.comm().rank()) {
        if (const char* path = std::getenv("OPM_APOST_HISTORY_CSV");
            path != nullptr && *path != '\0' && !aposteriori_rows_.empty()) {
            std::ofstream output(
                path, aposteriori_history_header_written_ ? std::ios::app : std::ios::trunc);
            if (!output) {
                throw std::runtime_error(
                    "Unable to open a posteriori history file '" + std::string(path) + "'");
            }
            if (!aposteriori_history_header_written_) {
                output << "accepted_step,time_days,report_step,substep,dt_days,"
                          "newton_iterations,eta_sp,eta_eq,eta_time,eta_lin_first,eta_lin_last,"
                          "eta_lin_accum_last,eta_lin_flux_last,eta_lin_source_last,"
                          "eta_alg_first,eta_alg_last,"
                          "eta_alg_rhs_first,eta_alg_rhs_last,"
                          "eta_time_over_eta_sp,material_balance\n";
                aposteriori_history_header_written_ = true;
            }
            const auto& first = aposteriori_rows_.front();
            const auto& last = aposteriori_rows_.back();
            // aposteriori_rows_.front() is Newton iteration 1's row (see the
            // per-iteration table's comment above); aposteriori_alg_rhs_[0]
            // is iteration 0's value, so the matching entry is index 1, not
            // front(). aposteriori_rows_.back() is the converged iteration,
            // which never gets an aposteriori_alg_rhs_ push at all (that push
            // is skipped once report.converged is true) -- .back() is the
            // best available approximation (the last non-converged
            // iteration's value), not an exact match for "last".
            const Scalar etaAlgRhsFirst = aposteriori_alg_rhs_.size() > 1
                ? aposteriori_alg_rhs_[1]
                : std::numeric_limits<Scalar>::quiet_NaN();
            const Scalar etaAlgRhsLast = aposteriori_alg_rhs_.empty()
                ? std::numeric_limits<Scalar>::quiet_NaN()
                : aposteriori_alg_rhs_.back();
            const Scalar ratio = last[0] > Scalar{0} ? last[3] / last[0] : Scalar{0};
            const Scalar day = unit::convert::to(
                this->simulator_.time() + timer.currentStepLength(), unit::day);
            output << std::setprecision(17)
                   << ++aposteriori_accepted_step_ << ',' << day << ','
                   << timer.reportStepNum() << ',' << timer.currentStepNum() << ','
                   << unit::convert::to(timer.currentStepLength(), unit::day) << ','
                   << aposteriori_rows_.size() << ','
                   << last[0] << ',' << last[7] << ',' << last[3] << ','
                   << first[5] << ',' << last[5] << ','
                   << aposteriori_estimator_->etaLinearizationAccumulation() << ','
                   << aposteriori_estimator_->etaLinearizationFlux() << ','
                   << aposteriori_estimator_->etaLinearizationSource() << ','
                   << first[6] << ',' << last[6] << ','
                   << etaAlgRhsFirst << ',' << etaAlgRhsLast << ','
                   << ratio << ','
                   << this->last_mass_balance_residual_ << '\n';
        }
    }
    if constexpr (requires {
        this->simulator_.problem().setAposterioriSpatialEnergy(
            aposteriori_estimator_->accumulatedSpatialEnergy(),
            aposteriori_estimator_->accumulatedPhaseComponentSpatialEnergy(
                FluidSystem::waterPhaseIdx),
            aposteriori_estimator_->accumulatedPhaseComponentSpatialEnergy(
                FluidSystem::oilPhaseIdx),
            aposteriori_estimator_->accumulatedPhaseComponentSpatialEnergy(
                FluidSystem::gasPhaseIdx),
            aposteriori_estimator_->latestSpatialEnergy(),
            aposteriori_estimator_->latestPhaseComponentSpatialEnergy(
                FluidSystem::waterPhaseIdx),
            aposteriori_estimator_->latestPhaseComponentSpatialEnergy(
                FluidSystem::oilPhaseIdx),
            aposteriori_estimator_->latestPhaseComponentSpatialEnergy(
                FluidSystem::gasPhaseIdx));
    }) {
        if (this->simulator_.problem().setAposterioriSpatialEnergy(
                aposteriori_estimator_->accumulatedSpatialEnergy(),
                aposteriori_estimator_->accumulatedPhaseComponentSpatialEnergy(
                    FluidSystem::waterPhaseIdx),
                aposteriori_estimator_->accumulatedPhaseComponentSpatialEnergy(
                    FluidSystem::oilPhaseIdx),
                aposteriori_estimator_->accumulatedPhaseComponentSpatialEnergy(
                    FluidSystem::gasPhaseIdx),
                aposteriori_estimator_->latestSpatialEnergy(),
                aposteriori_estimator_->latestPhaseComponentSpatialEnergy(
                    FluidSystem::waterPhaseIdx),
                aposteriori_estimator_->latestPhaseComponentSpatialEnergy(
                    FluidSystem::oilPhaseIdx),
                aposteriori_estimator_->latestPhaseComponentSpatialEnergy(
                    FluidSystem::gasPhaseIdx))) {
            aposteriori_estimator_->resetAccumulatedSpatialEnergy();
        }
    }
    else if constexpr (requires {
        this->simulator_.problem().setAposterioriSpatialEnergy(
            aposteriori_estimator_->accumulatedSpatialEnergy());
    }) {
        if (this->simulator_.problem().setAposterioriSpatialEnergy(
                aposteriori_estimator_->accumulatedSpatialEnergy())) {
            aposteriori_estimator_->resetAccumulatedSpatialEnergy();
        }
    }

    // OPM_APOST_DUMP_EVERY_REPORT=1: one per-cell estimator CSV per REPORT step
    // (apost_cells_report<N>.csv), for time-sequence plots of where eta_sp sits
    // relative to the saturation/pressure fields. Independent of refinement.
    if (!this->grid_.comm().rank() && std::getenv("OPM_APOST_DUMP_EVERY_REPORT")
        && timer.reportStepNum() != aposteriori_dump_report_step_) {
        aposteriori_dump_report_step_ = timer.reportStepNum();
        aposteriori_estimator_->dumpCellEstimators(
            fmt::format("apost_cells_report{}.csv", timer.reportStepNum()));
    }

    const char* rq = std::getenv("OPM_APOST_REFINE_REQUEST");
    if (!rq || this->grid_.comm().rank())
        return;

    const Scalar zRef = []() {
        const char* s = std::getenv("OPM_APOST_ZETA_REF");
        return s ? static_cast<Scalar>(std::atof(s)) : Scalar{0.5};
    }();
    double accumTheta = -1.0;
    if (const char* s = std::getenv("OPM_APOST_ACCUM_THETA"))
        accumTheta = std::atof(s);

    std::string spec;
    if (accumTheta > 0.0 && accumTheta < 1.0) {
        const auto marks =
            aposteriori_estimator_->refinementMarksAccumulated(accumTheta);
        spec = aposteriori_estimator_->refinementBoxSpecAccumulated(accumTheta);
        if (this->terminalOutputEnabled()) {
            OpmLog::info(fmt::format(
                "  [Algorithm 6.1] accumulated Dorfler(theta={:.2f}): {} seed cell(s)"
                " -> {} coarse cell(s) after halo/protection; "
                "captured {:.1f}% of total spatial energy",
                accumTheta, marks.second, marks.first,
                100.0 * aposteriori_estimator_->lastRetainedEnergyFraction()));
        }
    }
    else {
        spec = aposteriori_estimator_->refinementBoxSpec(zRef);
    }

    // Atomic publish (temp file + rename) so the driver never reads a
    // half-written spec.
    const std::string path =
        (std::string(rq) == "1") ? "apost_refine_request.txt" : rq;
    const std::string tmp = path + ".tmp";
    {
        std::ofstream os(tmp, std::ios::trunc);
        if (os) os << spec << '\n';
    }
    std::rename(tmp.c_str(), path.c_str());

    // History: one entry per REPORT step (not per accepted substep).
    if (timer.reportStepNum() != aposteriori_refine_hist_step_) {
        aposteriori_refine_hist_step_ = timer.reportStepNum();
        std::ofstream hs("apost_refine_history.txt", std::ios::app);
        if (hs)
            hs << "report " << timer.reportStepNum() << ": " << spec << '\n';
    }
}

template <class TypeTag>
template <class NonlinearSolverType>
SimulatorReportSingle
NonlinearSystemBlackOilReservoir<TypeTag>::
nonlinearIterationNewton(const SimulatorTimerInterface& timer,
                         NonlinearSolverType& nonlinear_solver)
{
    OPM_TIMEFUNCTION();

    SimulatorReportSingle report;
    Dune::Timer perfTimer;

    this->initialLinearization(report,
                               this->param_.newton_min_iter_,
                               this->param_.newton_max_iter_,
                               timer);

    if (!report.converged) {
        perfTimer.reset();
        perfTimer.start();
        report.total_newton_iterations = 1;

        const unsigned nc = this->simulator_.model().numGridDof();
        BVector x(nc);

        linear_solve_setup_time_ = 0.0;

        // The physical, post-Schur right-hand side, preserved before the
        // linear solver receives it: solveJacobianSystem() may use the
        // linearizer residual as mutable scaling/workspace storage, so
        // reading it again after the solve does not reconstruct b - A_eff*x.
        std::optional<BVector> algebraicRhs;

        // TEMPORARY (2026-09-14): reinstated for one round of figure
        // generation, flagged for removal again once that's done -- see the
        // matching removal note below and the OPM_APOST_LINDUMP block after
        // solveJacobianSystem(). Snapshot of the matrix for that same,
        // opt-in, off-by-default diagnostic.
        std::optional<Mat> linearDumpMatrix;

        // Endpoint-verified Criteria_alg state for this Newton linear solve.
        // The absolute target is expressed in the physical estimator norm;
        // the requested reduction is the tolerance passed to ISTL.
        std::optional<Scalar> algebraicAbsoluteTarget;
        std::optional<Scalar> algebraicRawReduction;
        double algebraicRequestedReduction =
            static_cast<double>(this->adaptive_linear_reduction_.reductionMin());
        const char* algebraicForcingSource = "static";
        int linearIterationsThisNewton = 0;
        int algebraicCorrections = 0;

        // Inexact-Newton: relax the linear tolerance to the current nonlinear
        // (linearization) error level for this solve. The converged nonlinear
        // solution is unchanged; only the linear iteration count is affected.
        auto& linSolver = this->simulator_.model().newtonMethod().linearSolver();
        if (!this->param_.enable_aposteriori_linear_tolerance_ &&
            this->adaptive_linear_reduction_.enabled()) {
            const auto forcing =
                this->adaptive_linear_reduction_.forcingTerm(this->residual_norms_history_);
            linSolver.setLinearSolveReduction(
                forcing.has_value() ? std::optional<double>(static_cast<double>(*forcing))
                                    : std::nullopt);
        }

        try {
            if (!aposteriori_well_linearized_early_) {
                this->wellModel().linearize(
                    this->simulator().model().linearizer().jacobian(),
                    this->simulator().model().linearizer().residual());
            }
            aposteriori_well_linearized_early_ = false;

            std::optional<Scalar> etaAlg0;
            if (aposteriori_estimator_
                && this->param_.enable_aposteriori_estimators_) {
                algebraicRhs =
                    this->simulator_.model().linearizer().residual();
                etaAlg0 = aposteriori_estimator_->computeAlgebraicEstimator(
                    *algebraicRhs, timer.currentStepLength());
                this->aposteriori_alg_rhs_.push_back(*etaAlg0);
            }

            // TEMPORARY (2026-09-14, marked for removal): a selected Krylov
            // dump needs an immutable matrix snapshot; taking A after
            // solveJacobianSystem() cannot establish the true iteration-zero
            // system.
            if (const char* ld = std::getenv("OPM_APOST_LINDUMP");
                ld != nullptr && algebraicRhs
                && this->grid_.comm().size() == 1) {
                int wantReport = -1, wantStep = -1, wantK = -1;
                const int fields = std::sscanf(
                    ld, "%d,%d,%d", &wantReport, &wantStep, &wantK);
                const int curStep = timer.currentStepNum();
                const int curReport = timer.reportStepNum();
                const int curK =
                    this->simulator_.problem().iterationContext().iteration();
                const bool selected = fields == 3
                    ? curReport == wantReport && curStep == wantStep && curK == wantK
                    : fields == 2 && curStep == wantReport && curK == wantStep;
                if (selected) {
                    linearDumpMatrix = this->simulator_.model()
                                           .linearizer().jacobian().istlMatrix();
                }
            }

            // Paper's Criteria_alg: stop the linear solve once
            //   eta_alg <= Gamma_alg * max(eta_sp, eta_time).
            // eta_alg scales linearly with the linear residual, so with
            // eta_alg^{(0)} the estimator of the un-reduced (x=0) residual the
            // relative reduction target is
            //   Gamma_alg * max(eta_sp,eta_time) / eta_alg^{(0)}.
            // Evaluated AFTER wellModel().linearize() so eta_alg^{(0)} is on the
            // scale of the reduced (well-eliminated) system actually solved.
            if (this->param_.enable_aposteriori_linear_tolerance_ && etaAlg0) {
                const Scalar budget = this->aposteriori_max_sptime_prev_;
                if (*etaAlg0 > Scalar{0} && std::isfinite(budget) && budget > Scalar{0}) {
                    algebraicAbsoluteTarget = this->aposteriori_targets_.GammaAlg * budget;
                    algebraicRawReduction = *algebraicAbsoluteTarget / *etaAlg0;
                }

                if (this->aposteriori_force_strict_linear_) {
                    algebraicForcingSource = "strict-after-unmet";
                    this->aposteriori_force_strict_linear_ = false;
                    linSolver.setLinearSolveReduction(algebraicRequestedReduction);
                }
                else {
                    const auto forcing =
                        this->adaptive_linear_reduction_.forcingTermFromEstimator(
                            *etaAlg0, budget, this->aposteriori_targets_.GammaAlg);
                    if (forcing) {
                        algebraicForcingSource = "criteria-alg";
                        algebraicRequestedReduction = static_cast<double>(*forcing);
                        linSolver.setLinearSolveReduction(algebraicRequestedReduction);
                    }
                    else {
                        algebraicForcingSource =
                            this->adaptive_linear_reduction_.previousIterations()
                                    < this->adaptive_linear_reduction_.minimumIterations()
                                ? "fixed-cheap-previous"
                                : "fixed-missing-budget";
                        linSolver.setLinearSolveReduction(std::nullopt);
                    }
                }
            }

            solveJacobianSystem(x);

            report.linear_solve_setup_time += linear_solve_setup_time_;
            report.linear_solve_time += perfTimer.stop();
            report.total_linear_iterations += linearIterationsLastSolve();
            linearIterationsThisNewton = linearIterationsLastSolve();

            if (this->adaptive_linear_reduction_.enabled()) {
                this->adaptive_linear_reduction_.recordLinearIterations(linearIterationsLastSolve());
            }

            // TEMPORARY (2026-09-14, marked for removal): diagnostic-only
            // illustrative re-solve for figure generation. Set
            // OPM_APOST_LINDUMP=<report>,<step>,<k> (or <step>,<k> for a
            // fixed-timestep run). Runs a SELF-CONTAINED BiCGSTAB+ILU0 on a
            // snapshot of the physical-unit well-eliminated system
            // A_eff = A + wellOp (taken before any solver rescaling) and
            // evaluates eta_alg on the true residual r = b - A_eff x_m at
            // successive iteration caps m = 0,1,2,... Nothing in OPM's own
            // solver is touched, so the Newton trajectory of the run is
            // unchanged; only a CSV is written. This curve is NOT OPM's real
            // solver trajectory (confirmed 2026-09-14: Dune's raw verbose
            // Defect and eta_alg are not proportional to each other, so
            // there is no way to reconstruct eta_alg's real intermediate
            // Krylov-iteration values from OPM's actual solve -- only its
            // two real endpoints, eta_alg(rhs) and eta_alg(post), are real).
            if (const char* ld = std::getenv("OPM_APOST_LINDUMP")) {
                int wantReport = -1, wantStep = -1, wantK = -1;
                const int fields = std::sscanf(
                    ld, "%d,%d,%d", &wantReport, &wantStep, &wantK);
                const int curStep = timer.currentStepNum();
                const int curReport = timer.reportStepNum();
                const int curK = this->simulator_.problem().iterationContext().iteration();
                const bool selected = fields == 3
                    ? curReport == wantReport && curStep == wantStep && curK == wantK
                    : fields == 2 && curStep == wantReport && curK == wantStep;
                if (selected && aposteriori_estimator_ && algebraicRhs
                    && linearDumpMatrix && this->grid_.comm().size() == 1) {
                    const Mat& A0 = *linearDumpMatrix;
                    const BVector& b0 = *algebraicRhs;
                    const Scalar dtd = timer.currentStepLength();
                    const double b2 = std::max(b0.two_norm(), 1e-300);
                    const Scalar eSp = aposteriori_estimator_->etaSpatialMimetic();
                    const Scalar eTm = aposteriori_estimator_->temporalAvailable()
                        ? aposteriori_estimator_->etaTemporal() : eSp;
                    const Scalar eLin = aposteriori_estimator_->etaLinearization();
                    const Scalar gAlg = this->aposteriori_targets_.GammaAlg;

                    // Copy A0 (blocks are Opm::MatrixBlock) into a plain
                    // FieldMatrix BCRS so Dune::SeqILU can be built on it.
                    static constexpr int bs = Mat::block_type::rows;
                    using FBlock = Dune::FieldMatrix<double, bs, bs>;
                    using FMat = Dune::BCRSMatrix<FBlock>;
                    FMat Afm(A0.N(), A0.M(), FMat::random);
                    for (auto r = A0.begin(); r != A0.end(); ++r) {
                        Afm.setrowsize(r.index(), r->size());
                    }
                    Afm.endrowsizes();
                    for (auto r = A0.begin(); r != A0.end(); ++r) {
                        for (auto c = r->begin(); c != r->end(); ++c) {
                            Afm.addindex(r.index(), c.index());
                        }
                    }
                    Afm.endindices();
                    for (auto r = A0.begin(); r != A0.end(); ++r) {
                        for (auto c = r->begin(); c != r->end(); ++c) {
                            for (int p = 0; p < bs; ++p) {
                                for (int q = 0; q < bs; ++q) {
                                    Afm[r.index()][c.index()][p][q] = (*c)[p][q];
                                }
                            }
                        }
                    }

                    WellModelAsLinearOperator<typename ParentType::WellModel, BVector, BVector>
                        wellOp(this->wellModel());
                    WellModelMatrixAdapter<FMat, BVector, BVector> opEff(Afm, wellOp);
                    Dune::SeqILU<FMat, BVector, BVector> ilu(Afm, 0.92);
                    Dune::SeqScalarProduct<BVector> sp;

                    const auto trueResid = [&](const BVector& xm) {
                        BVector Ax(xm.size());
                        Ax = 0.0;
                        opEff.apply(xm, Ax);        // A x + wellOp x
                        BVector r(b0);
                        r -= Ax;
                        return r;
                    };
                    const std::string dumpPath = fields == 3
                        ? fmt::format("apost_lindump_report{}_step{}_k{}.csv",
                                      curReport, curStep, curK)
                        : fmt::format("apost_lindump_step{}_k{}.csv", curStep, curK);
                    std::ofstream os(dumpPath);
                    const int opmIterations = linearIterationsLastSolve();
                    os << "iter,rel_resid,eta_alg,eta_sp,eta_time,eta_lin,"
                          "gamma_alg_abs_target,opm_iterations\n";
                    const auto row = [&](int m, const BVector& r) {
                        const Scalar ea = aposteriori_estimator_->computeAlgebraicEstimator(r, dtd);
                        os << fmt::format("{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},"
                                          "{:.6e},{}\n",
                                          m, r.two_norm() / b2, static_cast<double>(ea),
                                          static_cast<double>(eSp), static_cast<double>(eTm),
                                          static_cast<double>(eLin),
                                          static_cast<double>(gAlg * std::max(eSp, eTm)),
                                          opmIterations);
                    };
                    row(0, b0);   // x = 0
                    int lastIt = 0;
                    for (int m = 1; m <= 60; ++m) {
                        BVector xm(b0.size());
                        xm = 0.0;
                        BVector rhs(b0);   // solver overwrites rhs
                        Dune::BiCGSTABSolver<BVector> solver(opEff, sp, ilu, 1e-14, m, 0);
                        Dune::InverseOperatorResult res;
                        solver.apply(xm, rhs, res);
                        const int itDone = res.iterations > 0 ? res.iterations : m;
                        if (itDone == lastIt) {
                            continue;   // converged; no new iterate
                        }
                        lastIt = itDone;
                        const BVector r = trueResid(xm);
                        row(itDone, r);
                        if (r.two_norm() / b2 < 1e-10) {
                            break;
                        }
                    }
                    OpmLog::info(fmt::format(
                        "  [a posteriori] wrote {}", dumpPath));
                }
            }

            // eta_alg (eq. est_alg): the weighted estimator of the residual of
            // the EFFECTIVE linear system actually solved, r_alg = b - A_eff x.
            // With --matrix-add-well-contributions=false (the default) ISTL
            // solves with a WellModelMatrixAdapter that applies the well
            // operator matrix-free, so the sparse matrix alone is not A_eff.
            const auto linearResidual = [&]() {
                const auto& A = this->simulator_.model().linearizer().jacobian().istlMatrix();
                const auto& b = *algebraicRhs;
                BVector Ax(x.size());
                Ax = 0.0;
                A.mv(x, Ax);
                if (!this->param_.matrix_add_well_contributions_) {
                    WellModelAsLinearOperator<typename ParentType::WellModel,
                                              BVector, BVector>
                        wellOp(this->wellModel());
                    wellOp.applyscaleadd(1.0, x, Ax);   // Ax += well(x)
                }
                BVector rAlg(b);
                rAlg -= Ax;
                return rAlg;
            };
            const auto evalEtaAlg = [&]() {
                const BVector rAlg = linearResidual();
                return this->aposteriori_estimator_->computeAlgebraicEstimator(
                    rAlg, timer.currentStepLength());
            };

            this->aposteriori_alg_unmet_ = false;
            if (this->aposteriori_estimator_ && this->param_.enable_aposteriori_estimators_) {
                Scalar etaAlg = evalEtaAlg();
                const Scalar etaAlgBeforeCorrection = etaAlg;

                // Endpoint-verified Criteria_alg. solveJacobianSystem() does
                // not preserve x as a Krylov warm start, so each correction
                // recomputes the full Newton increment from zero. Each request
                // tightens the preceding request by the measured physical
                // target/eta_alg shortfall, with a factor-two safety margin.
                if (this->param_.enable_aposteriori_linear_tolerance_
                    && this->param_.aposteriori_alg_max_resolves_ > 0
                    && algebraicAbsoluteTarget) {
                    const Scalar target = *algebraicAbsoluteTarget;
                    double correctionReduction = algebraicRequestedReduction;
                    for (int resolve = 0;
                         resolve < this->param_.aposteriori_alg_max_resolves_
                             && std::isfinite(target) && target > Scalar{0}
                             && etaAlg > target;
                         ++resolve) {
                        correctionReduction = std::clamp(
                            correctionReduction
                                * static_cast<double>(target / etaAlg) * 0.5,
                            static_cast<double>(this->adaptive_linear_reduction_.reductionMin()),
                            static_cast<double>(this->adaptive_linear_reduction_.reductionMax()));
                        linSolver.setLinearSolveReduction(
                            std::optional<double>(correctionReduction));
                        BVector correctedX(nc);
                        correctedX = 0.0;
                        // The first solve may use the linearizer residual as
                        // scaling/workspace storage. Restore the immutable
                        // post-Schur RHS before solving the same physical
                        // system again; otherwise the correction sees the
                        // mutated near-zero workspace residual.
                        this->simulator_.model().linearizer().residual() =
                            *algebraicRhs;
                        solveJacobianSystem(correctedX);
                        x = std::move(correctedX);
                        report.total_linear_iterations += linearIterationsLastSolve();
                        linearIterationsThisNewton += linearIterationsLastSolve();
                        ++algebraicCorrections;
                        etaAlg = evalEtaAlg();
                    }
                    this->aposteriori_alg_unmet_ =
                        std::isfinite(target) && target > Scalar{0}
                        && etaAlg > target;
                    this->aposteriori_force_strict_linear_ =
                        this->aposteriori_alg_unmet_;
                    if (this->aposteriori_alg_unmet_ && !this->grid_.comm().rank()) {
                        OpmLog::info(fmt::format(
                            "  [a posteriori] Criteria_alg still not met after "
                            "{} endpoint correction(s); estimator Newton "
                            "acceptance is disabled and the next linear solve "
                            "is forced to the strict tolerance.",
                            algebraicCorrections));
                    }
                }

                if (this->adaptive_linear_reduction_.enabled()) {
                    this->adaptive_linear_reduction_.recordLinearIterations(
                        linearIterationsThisNewton);
                }

                if (std::getenv("OPM_APOST_TRACE_LINEAR") != nullptr
                    && !this->grid_.comm().rank()) {
                    const double nan = std::numeric_limits<double>::quiet_NaN();
                    const double target = algebraicAbsoluteTarget
                        ? static_cast<double>(*algebraicAbsoluteTarget) : nan;
                    const double raw = algebraicRawReduction
                        ? static_cast<double>(*algebraicRawReduction) : nan;
                    const bool checked = algebraicAbsoluteTarget.has_value();
                    OpmLog::info(fmt::format(
                        "  [a posteriori linear] step={} k={} source={} "
                        "raw_reduction={:.6e} requested_reduction={:.6e} "
                        "eta_alg_0={:.6e} target={:.6e} "
                        "eta_alg_first={:.6e} corrections={} "
                        "eta_alg_final={:.6e} pass={} linear_iterations={}",
                        timer.currentStepNum(),
                        this->simulator_.problem().iterationContext().iteration(),
                        algebraicForcingSource, raw, algebraicRequestedReduction,
                        static_cast<double>(etaAlg0.value_or(Scalar{0})), target,
                        static_cast<double>(etaAlgBeforeCorrection),
                        algebraicCorrections, static_cast<double>(etaAlg),
                        checked ? (etaAlg <= *algebraicAbsoluteTarget ? "true" : "false")
                                : "not-available",
                        linearIterationsThisNewton));
                }
            }
            if (this->aposteriori_estimator_
                && this->param_.enable_aposteriori_newton_stopping_) {
                this->aposteriori_estimator_->recordPredictedLinearResidual(
                    linearResidual());
            }
        }
        catch (...) {
            report.linear_solve_setup_time += linear_solve_setup_time_;
            report.linear_solve_time += perfTimer.stop();
            report.total_linear_iterations += linearIterationsLastSolve();

            if (this->adaptive_linear_reduction_.enabled()) {
                linSolver.setLinearSolveReduction(std::nullopt);
            }
            this->failureReport_ += report;
            throw;
        }

        // Restore the configured static tolerance for any later consumer of
        // the same linear solver (e.g. well-only or NLDD local solves).
        if (this->adaptive_linear_reduction_.enabled()) {
            linSolver.setLinearSolveReduction(std::nullopt);
        }

        perfTimer.reset();
        perfTimer.start();

        this->wellModel().postSolve(x);

        if (this->param_.use_update_stabilization_) {
            bool isOscillate = false;
            bool isStagnate = false;
            nonlinear_solver.detectOscillations(this->residual_norms_history_,
                                                this->residual_norms_history_.size() - 1,
                                                isOscillate,
                                                isStagnate);
            if (isOscillate) {
                this->current_relaxation_ -= nonlinear_solver.relaxIncrement();
                this->current_relaxation_ = std::max(this->current_relaxation_, nonlinear_solver.relaxMax());
                // The detector reads reservoir residual history and the response damps the
                // reservoir update, but the oscillation may originate in the well/group
                // control layer.  Record what was unsatisfied here so the two can be told
                // apart.
                auto source = ConvergenceReport::OscillationSource::NotDetected;
                if (!this->convergence_reports_.empty() &&
                    !this->convergence_reports_.back().report.empty())
                {
                    auto& convrep = this->convergence_reports_.back().report.back();
                    source = classifyOscillationSource(convrep);
                    convrep.setOscillationSource(source);
                }
                if (this->terminalOutputEnabled()) {
                    OpmLog::info("    Oscillating behavior detected (" + to_string(source)
                                 + "): Relaxation set to "
                                 + std::to_string(this->current_relaxation_));
                }
            }
            nonlinear_solver.stabilizeNonlinearUpdate(x, this->dx_old_, this->current_relaxation_);
        }

        // Capture the Newton-linearized flux/accumulation defect (eq.
        // Newton_it_flux, lin_fv_balance) for the rigorous eta_lin, using the
        // FINAL (stabilized) increment x -- this MUST happen before
        // updateSolution(x) below, while intensiveQuantities() still reflect
        // chi^{k-1,n}, the state the linear system was actually built at.
        // In diagnostic mode the same rigorous eta_lin is recorded without
        // changing OPM's standard Newton stopping decision.
        if (this->aposteriori_estimator_ && this->param_.aposteriori_rigorous_lin_ &&
            (this->param_.enable_aposteriori_newton_stopping_
             || std::getenv("OPM_APOST_ITERATION_DIAGNOSTICS") != nullptr) &&
            this->simulator_.problem().iterationContext().iteration() + 1
                >= this->param_.aposteriori_first_eval_iter_) {
            this->aposteriori_estimator_->recordLinearizationDefect(timer.currentStepLength(), x);
        }

        this->updateSolution(x);
        report.update_time += perfTimer.stop();
    }

    return report;
}

template <class TypeTag>
typename NonlinearSystemBlackOilReservoir<TypeTag>::Scalar
NonlinearSystemBlackOilReservoir<TypeTag>::
relativeChange() const
{
    Scalar resultDelta = 0.0;
    Scalar resultDenom = 0.0;

    const auto& elemMapper = this->simulator_.model().elementMapper();
    const auto& gridView = this->simulator_.gridView();
    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
        unsigned globalElemIdx = elemMapper.index(elem);
        const auto& priVarsNew = this->simulator_.model().solution(/*timeIdx=*/0)[globalElemIdx];

        Scalar pressureNew;
        pressureNew = priVarsNew[Indices::pressureSwitchIdx];

        Scalar saturationsNew[FluidSystem::numPhases] = { 0.0 };
        Scalar oilSaturationNew = 1.0;
        if (FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx) &&
            FluidSystem::numActivePhases() > 1 &&
            priVarsNew.primaryVarsMeaningWater() == PrimaryVariables::WaterMeaning::Sw)
        {
            saturationsNew[FluidSystem::waterPhaseIdx] = priVarsNew[Indices::waterSwitchIdx];
            oilSaturationNew -= saturationsNew[FluidSystem::waterPhaseIdx];
        }

        if (FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx) &&
            FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx) &&
            priVarsNew.primaryVarsMeaningGas() == PrimaryVariables::GasMeaning::Sg)
        {
            assert(Indices::compositionSwitchIdx != std::numeric_limits<unsigned>::max());
            saturationsNew[FluidSystem::gasPhaseIdx] = priVarsNew[Indices::compositionSwitchIdx];
            oilSaturationNew -= saturationsNew[FluidSystem::gasPhaseIdx];
        }

        if (FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)) {
            saturationsNew[FluidSystem::oilPhaseIdx] = oilSaturationNew;
        }

        const auto& priVarsOld = this->simulator_.model().solution(/*timeIdx=*/1)[globalElemIdx];

        Scalar pressureOld;
        pressureOld = priVarsOld[Indices::pressureSwitchIdx];

        Scalar saturationsOld[FluidSystem::numPhases] = { 0.0 };
        Scalar oilSaturationOld = 1.0;

        // NB fix me! adding pressures changes to saturation changes does not make sense
        Scalar tmp = pressureNew - pressureOld;
        resultDelta += tmp*tmp;
        resultDenom += pressureNew*pressureNew;

        if (FluidSystem::numActivePhases() > 1) {
            if (priVarsOld.primaryVarsMeaningWater() == PrimaryVariables::WaterMeaning::Sw) {
                saturationsOld[FluidSystem::waterPhaseIdx] =
                    priVarsOld[Indices::waterSwitchIdx];
                oilSaturationOld -= saturationsOld[FluidSystem::waterPhaseIdx];
            }

            if (priVarsOld.primaryVarsMeaningGas() == PrimaryVariables::GasMeaning::Sg)
            {
                assert(Indices::compositionSwitchIdx != std::numeric_limits<unsigned>::max());
                saturationsOld[FluidSystem::gasPhaseIdx] =
                    priVarsOld[Indices::compositionSwitchIdx];
                oilSaturationOld -= saturationsOld[FluidSystem::gasPhaseIdx];
            }

            if (FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)) {
                saturationsOld[FluidSystem::oilPhaseIdx] = oilSaturationOld;
            }
            for (unsigned phaseIdx = 0; phaseIdx < FluidSystem::numPhases; ++ phaseIdx) {
                Scalar tmpSat = saturationsNew[phaseIdx] - saturationsOld[phaseIdx];
                resultDelta += tmpSat*tmpSat;
                resultDenom += saturationsNew[phaseIdx]*saturationsNew[phaseIdx];
                assert(std::isfinite(resultDelta));
                assert(std::isfinite(resultDenom));
            }
        }
    }

    resultDelta = gridView.comm().sum(resultDelta);
    resultDenom = gridView.comm().sum(resultDenom);

    return resultDenom > 0.0 ? resultDelta / resultDenom : 0.0;
}

template <class TypeTag>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
solveJacobianSystem(BVector& x)
{
    auto& jacobian = this->simulator_.model().linearizer().jacobian().istlMatrix();
    auto& residual = this->simulator_.model().linearizer().residual();
    auto& linSolver = this->simulator_.model().newtonMethod().linearSolver();

    const int numSolvers = linSolver.numAvailableSolvers();
    if (numSolvers > 1 && (linSolver.getSolveCount() % 100 == 0)) {
        if (this->terminal_output_) {
            OpmLog::debug("\nRunning speed test for comparing available linear solvers.");
        }

        Dune::Timer perfTimer;
        std::vector<double> times(numSolvers);
        std::vector<double> setupTimes(numSolvers);

        x = 0.0;
        std::vector<BVector> x_trial(numSolvers, x);
        for (int solver = 0; solver < numSolvers; ++solver) {
            linSolver.setActiveSolver(solver);
            perfTimer.start();
            linSolver.prepare(jacobian, residual);
            setupTimes[solver] = perfTimer.stop();
            perfTimer.reset();
            linSolver.setResidual(residual);
            perfTimer.start();
            linSolver.solve(x_trial[solver]);
            times[solver] = setupTimes[solver] + perfTimer.stop();
            perfTimer.reset();
            if (this->terminal_output_) {
                OpmLog::debug(fmt::format(fmt::runtime("Solver time {}: {}"), solver, times[solver]));
            }
        }

        int fastest_solver = std::ranges::min_element(times) - times.begin();
        // Use timing on rank 0 to determine fastest, must be consistent across ranks.
        this->grid_.comm().broadcast(&fastest_solver, 1, 0);
        this->linear_solve_setup_time_ = setupTimes[fastest_solver];
        x = x_trial[fastest_solver];
        linSolver.setActiveSolver(fastest_solver);
    }
    else {
        x = 0.0;

        Dune::Timer perfTimer;
        perfTimer.start();
        linSolver.prepare(jacobian, residual);
        this->linear_solve_setup_time_ = perfTimer.stop();
        linSolver.setResidual(residual);
        // actually, the error needs to be calculated after setResidual in order to
        // account for parallelization properly. since the residual of ECFV
        // discretizations does not need to be synchronized across processes to be
        // consistent, this is not relevant for OPM-flow...
        linSolver.solve(x);
    }
}

template <class TypeTag>
bool
NonlinearSystemBlackOilReservoir<TypeTag>::
shouldStoreSolutionUpdate() const
{
    return this->param_.tolerance_max_dp_ > 0.0 || this->param_.tolerance_max_ds_ > 0.0
        || this->param_.tolerance_max_drs_ > 0.0 || this->param_.tolerance_max_drv_ > 0.0;
}

template <class TypeTag>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
prepareSolutionUpdate()
{
    // Init. solution update vector
    unsigned nc = this->simulator_.model().numGridDof();
    solUpd_.resize(nc);

    const auto& elemMapper = this->simulator_.model().elementMapper();
    const auto& gridView = this->simulator_.gridView();
    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
        // Copy solution vector to transfer primary variables meaning
        unsigned globalElemIdx = elemMapper.index(elem);
        solUpd_[globalElemIdx] = this->simulator_.model().solution(/*timeIdx=*/0)[globalElemIdx];

        // Ensure each element is zero
        std::ranges::fill(solUpd_[globalElemIdx], 0.0);
    }
}

template <class TypeTag>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
storeSolutionUpdate(const GlobalEqVector& dx)
{
    const auto& elemMapper = this->simulator_.model().elementMapper();
    const auto& gridView = this->simulator_.gridView();
    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
        // Get cell vectors
        unsigned globalElemIdx = elemMapper.index(elem);
        auto& value = solUpd_[globalElemIdx];
        const auto& update = dx[globalElemIdx];
        assert(value.size() == update.size());

        // Transfer update from dx to solution update container (SolutionVector type)
        std::ranges::copy(update, value.begin());
    }
}

template <class TypeTag>
typename NonlinearSystemBlackOilReservoir<TypeTag>::MaxSolutionUpdateData
NonlinearSystemBlackOilReservoir<TypeTag>::
getMaxSolutionUpdate(const std::vector<unsigned>& ixCells)
{
    static constexpr bool enableSolvent =
        Indices::solventSaturationIdx != std::numeric_limits<unsigned>::max();
    static constexpr bool enableBrine =
        Indices::saltConcentrationIdx != std::numeric_limits<unsigned>::max();

    // Init output
    Scalar dPMax = 0.0;
    Scalar dSMax = 0.0;
    Scalar dRsMax = 0.0;
    Scalar dRvMax = 0.0;

    // Loop over solution update, get the correct variables and calculate max.
    for (const auto& ix : ixCells) {
        const auto& value = solUpd_[ix];
        for (unsigned pvIdx = 0; pvIdx < value.size(); ++pvIdx) {
            if (pvIdx == Indices::pressureSwitchIdx) {
                dPMax = std::max(dPMax, std::abs(value[pvIdx]));
            }
            else if ((pvIdx == Indices::waterSwitchIdx
                       && value.primaryVarsMeaningWater() == PrimaryVariables::WaterMeaning::Sw)
                      || (pvIdx == Indices::compositionSwitchIdx
                          && value.primaryVarsMeaningGas() == PrimaryVariables::GasMeaning::Sg)
                      || (enableSolvent && pvIdx == Indices::solventSaturationIdx
                          && value.primaryVarsMeaningSolvent() == PrimaryVariables::SolventMeaning::Ss)
                      || (enableBrine && enableSaltPrecipitation && pvIdx == Indices::saltConcentrationIdx
                          && value.primaryVarsMeaningBrine() == PrimaryVariables::BrineMeaning::Sp) ) {
                dSMax = std::max(dSMax, std::abs(value[pvIdx]));
            }
            else if (pvIdx == Indices::compositionSwitchIdx
                     && value.primaryVarsMeaningGas() == PrimaryVariables::GasMeaning::Rs) {
                dRsMax = std::max(dRsMax, std::abs(value[pvIdx]));
            }
            else if (pvIdx == Indices::compositionSwitchIdx
                     && value.primaryVarsMeaningGas() == PrimaryVariables::GasMeaning::Rv) {
                dRvMax = std::max(dRvMax, std::abs(value[pvIdx]));
            }
        }
    }

    // Communicate max values
    dPMax = this->grid_.comm().max(dPMax);
    dSMax = this->grid_.comm().max(dSMax);
    dRsMax = this->grid_.comm().max(dRsMax);
    dRvMax = this->grid_.comm().max(dRvMax);

    return { dPMax, dSMax, dRsMax, dRvMax };
}

template <class TypeTag>
std::tuple<typename NonlinearSystemBlackOilReservoir<TypeTag>::Scalar,
           typename NonlinearSystemBlackOilReservoir<TypeTag>::Scalar>
NonlinearSystemBlackOilReservoir<TypeTag>::
convergenceReduction(Parallel::Communication comm,
                     const Scalar pvSumLocal,
                     const Scalar numAquiferPvSumLocal,
                     std::vector< Scalar >& R_sum,
                     std::vector< Scalar >& maxCoeff,
                     std::vector< Scalar >& B_avg)
{
    return ParentType::convergenceReduction(comm,
                                            pvSumLocal,
                                            numAquiferPvSumLocal,
                                            R_sum,
                                            maxCoeff,
                                            B_avg);
}

template <class TypeTag>
std::pair<typename NonlinearSystemBlackOilReservoir<TypeTag>::Scalar,
          typename NonlinearSystemBlackOilReservoir<TypeTag>::Scalar>
NonlinearSystemBlackOilReservoir<TypeTag>::
localConvergenceData(std::vector<Scalar>& R_sum,
                     std::vector<Scalar>& maxCoeff,
                     std::vector<Scalar>& B_avg,
                     std::vector<int>& maxCoeffCell)
{
    OPM_TIMEBLOCK(localConvergenceData);
    Scalar pvSumLocal = 0.0;
    Scalar numAquiferPvSumLocal = 0.0;
    const auto& model = this->simulator_.model();
    const auto& problem = this->simulator_.problem();

    const auto& residual = this->simulator_.model().linearizer().residual();

    ElementContext elemCtx(this->simulator_);
    const auto& gridView = this->simulator().gridView();
    IsNumericalAquiferCell isNumericalAquiferCell(gridView.grid());
    OPM_BEGIN_PARALLEL_TRY_CATCH();
    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
        elemCtx.updatePrimaryStencil(elem);
        elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);

        const unsigned cell_idx = elemCtx.globalSpaceIndex(/*spaceIdx=*/0, /*timeIdx=*/0);
        const auto& intQuants = elemCtx.intensiveQuantities(/*spaceIdx=*/0, /*timeIdx=*/0);
        const auto& fs = intQuants.fluidState();

        const auto pvValue = problem.referencePorosity(cell_idx, /*timeIdx=*/0) *
                             model.dofTotalVolume(cell_idx);
        pvSumLocal += pvValue;

        if (isNumericalAquiferCell(elem)) {
            numAquiferPvSumLocal += pvValue;
        }

        this->getMaxCoeff(cell_idx, intQuants, fs, residual, pvValue,
                          B_avg, R_sum, maxCoeff, maxCoeffCell);
    }

    OPM_END_PARALLEL_TRY_CATCH("NonlinearSystemBlackOilReservoir::localConvergenceData() failed: ", this->grid_.comm());

    // compute local average in terms of global number of elements
    const int bSize = B_avg.size();
    for (int i = 0; i < bSize; ++i) {
        B_avg[i] /= Scalar(this->global_nc_);
    }

    return {pvSumLocal, numAquiferPvSumLocal};
}

template <class TypeTag>
typename NonlinearSystemBlackOilReservoir<TypeTag>::CnvPvSplitData
NonlinearSystemBlackOilReservoir<TypeTag>::
characteriseCnvPvSplit(const std::vector<Scalar>& B_avg, const double dt)
{
    OPM_TIMEBLOCK(computeCnvErrorPv);

    // 0: cnv <= tolerance_cnv
    // 1: tolerance_cnv < cnv <= tolerance_cnv_relaxed
    // 2: tolerance_cnv_relaxed < cnv
    constexpr auto numPvGroups = std::vector<double>::size_type{3};

    auto cnvPvSplit = std::pair<std::vector<double>, std::vector<int>> {
        std::piecewise_construct,
        std::forward_as_tuple(numPvGroups),
        std::forward_as_tuple(numPvGroups)
    };

    auto maxCNV = [&B_avg, dt](const auto& residual, const double pvol)
    {
        return (dt / pvol) *
            std::inner_product(residual.begin(), residual.end(),
                               B_avg.begin(), Scalar{0},
                               [](const Scalar m, const auto& x)
                               {
                                   using std::abs;
                                   return std::max(m, abs(x));
                               }, std::multiplies<>{});
    };

    auto& [splitPV, cellCntPV] = cnvPvSplit;

    const auto& model = this->simulator().model();
    const auto& problem = this->simulator().problem();
    const auto& residual = model.linearizer().residual();
    const auto& gridView = this->simulator().gridView();

    const IsNumericalAquiferCell isNumericalAquiferCell(gridView.grid());

    ElementContext elemCtx(this->simulator());

    std::vector<unsigned> ixCells;

    OPM_BEGIN_PARALLEL_TRY_CATCH();
    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
        // Skip cells of numerical Aquifer
        if (isNumericalAquiferCell(elem)) {
            continue;
        }

        elemCtx.updatePrimaryStencil(elem);

        const unsigned cell_idx = elemCtx.globalSpaceIndex(/*spaceIdx=*/0, /*timeIdx=*/0);
        const auto pvValue = problem.referencePorosity(cell_idx, /*timeIdx=*/0)
            * model.dofTotalVolume(cell_idx);

        const auto maxCnv = maxCNV(residual[cell_idx], pvValue);

        const auto ix = (maxCnv > this->param_.tolerance_cnv_)
            + (maxCnv > this->param_.tolerance_cnv_relaxed_);

        splitPV[ix] += static_cast<double>(pvValue);
        ++cellCntPV[ix];

        // For dP and dS check, we need cell indices of [1] violations
        if ( ix > 0 &&
             (this->param_.tolerance_max_dp_ > 0.0 || this->param_.tolerance_max_ds_ > 0.0
              || this->param_.tolerance_max_drs_ > 0.0 || this->param_.tolerance_max_drv_ > 0.0 ) ) {
            ixCells.push_back(cell_idx);
        }
    }

    OPM_END_PARALLEL_TRY_CATCH("NonlinearSystemBlackOilReservoir::characteriseCnvPvSplit() failed: ",
                               this->grid_.comm());

    this->grid_.comm().sum(splitPV  .data(), splitPV  .size());
    this->grid_.comm().sum(cellCntPV.data(), cellCntPV.size());

    return { cnvPvSplit, ixCells };
}

template <class TypeTag>
ConvergenceReport
NonlinearSystemBlackOilReservoir<TypeTag>::
getReservoirConvergence(const double reportTime,
                        const double dt,
                        const int maxIter,
                        std::vector<Scalar>& B_avg,
                        std::vector<Scalar>& residual_norms)
{
    OPM_TIMEBLOCK(getReservoirConvergence);
    using Vector = std::vector<Scalar>;

    const auto& iterCtx = this->simulator_.problem().iterationContext();

    ConvergenceReport report{reportTime};

    const int numComp = numEq;

    Vector R_sum(numComp, Scalar{0});
    Vector maxCoeff(numComp, std::numeric_limits<Scalar>::lowest());
    std::vector<int> maxCoeffCell(numComp, -1);

    const auto [pvSumLocal, numAquiferPvSumLocal] =
        this->localConvergenceData(R_sum, maxCoeff, B_avg, maxCoeffCell);

    // compute global sum and max of quantities
    const auto& [pvSum, numAquiferPvSum] =
        this->convergenceReduction(this->grid_.comm(),
                                   pvSumLocal,
                                   numAquiferPvSumLocal,
                                   R_sum, maxCoeff, B_avg);

    auto cnvSplitData = this->characteriseCnvPvSplit(B_avg, dt);
    report.setCnvPoreVolSplit(cnvSplitData.cnvPvSplit,
                              pvSum - numAquiferPvSum);

    // For each iteration, we need to determine whether to use the
    // relaxed tolerances.  To disable the usage of relaxed
    // tolerances, you can set the relaxed tolerances as the strict
    // tolerances.  If min_strict_mb_iter = -1 (default) we use a
    // relaxed tolerance for the mass balance for the last
    // iterations.  For positive values we use the relaxed tolerance
    // after the given number of iterations
    const bool relax_final_iteration_mb =
        this->param_.min_strict_mb_iter_ < 0 && iterCtx.iteration() == maxIter;

    const bool relax_iter_mb = this->param_.min_strict_mb_iter_ >= 0 &&
                               iterCtx.shouldRelax(this->param_.min_strict_mb_iter_);

    const bool use_relaxed_mb = relax_final_iteration_mb
        || relax_iter_mb;

    // If min_strict_cnv_iter = -1 we use a relaxed tolerance for
    // the cnv for the last iterations.  For positive values we use
    // the relaxed tolerance after the given number of iterations.
    // We also use relaxed tolerances for cells with total
    // pore-volume less than relaxed_max_pv_fraction_.  Default
    // value of relaxed_max_pv_fraction_ is 0.03
    const bool relax_final_iteration_cnv =
        this->param_.min_strict_cnv_iter_ < 0 && iterCtx.iteration() == maxIter;

    const bool relax_iter_cnv = this->param_.min_strict_cnv_iter_ >= 0 &&
                                iterCtx.shouldRelax(this->param_.min_strict_cnv_iter_);

    // Note trailing parentheses here, just before the final
    // semicolon.  This is an immediately invoked function
    // expression which calculates a single boolean value.
    const auto relax_pv_fraction_cnv =
        [&report, this, eligible = pvSum - numAquiferPvSum]()
    {
        const auto& cnvPvSplit = report.cnvPvSplit().first;

        // [1]: tol < cnv <= relaxed
        // [2]: relaxed < cnv
        Scalar cnvPvSum = static_cast<Scalar>(cnvPvSplit[1] + cnvPvSplit[2]);
        return cnvPvSum < this->param_.relaxed_max_pv_fraction_ * eligible &&
            cnvPvSum > 0.0;
    }();

    // If tolerances for solution changes are met, we use the
    // relaxed cnv tolerance. Note that all tolerances > 0.0
    // must be met to enable use of relaxed cnv tolerance.
    MaxSolutionUpdateData maxSolUpd;
    const bool use_dp_tol = this->param_.tolerance_max_dp_ > 0.0;
    const bool use_ds_tol = this->param_.tolerance_max_ds_ > 0.0;
    const bool use_drs_tol = this->param_.tolerance_max_drs_ > 0.0;
    const bool use_drv_tol = this->param_.tolerance_max_drv_ > 0.0;
    const bool use_dsol_tol = use_dp_tol || use_ds_tol || use_drs_tol || use_drv_tol;
    bool relax_dsol_cnv = false;
    if (!iterCtx.isFirstGlobalIteration() && use_dsol_tol) {
        maxSolUpd = getMaxSolutionUpdate(cnvSplitData.ixCells);
        relax_dsol_cnv =
            (!use_dp_tol || (maxSolUpd.dPMax > 0.0 && maxSolUpd.dPMax < this->param_.tolerance_max_dp_)) &&
            (!use_ds_tol || (maxSolUpd.dSMax > 0.0 && maxSolUpd.dSMax < this->param_.tolerance_max_ds_)) &&
            (!use_drs_tol || (maxSolUpd.dRsMax > 0.0 && maxSolUpd.dRsMax < this->param_.tolerance_max_drs_)) &&
            (!use_drv_tol || (maxSolUpd.dRvMax > 0.0 && maxSolUpd.dRvMax < this->param_.tolerance_max_drv_));
    }

    // Determine if relaxed CNV tolerances should be used
    const bool use_relaxed_cnv = relax_final_iteration_cnv
        || relax_pv_fraction_cnv
        || relax_iter_cnv
        || relax_dsol_cnv;

    // Ensure that CNV convergence criteria is met when max.
    // solution change tolerances have been fulfilled
    Scalar tolerance_cnv_relaxed = relax_dsol_cnv ? 1e20 : this->param_.tolerance_cnv_relaxed_;

    const auto tol_cnv = use_relaxed_cnv ? tolerance_cnv_relaxed : this->param_.tolerance_cnv_;
    const auto tol_mb  = use_relaxed_mb ? this->param_.tolerance_mb_relaxed_ : this->param_.tolerance_mb_;

    // Record which condition granted the relaxation, so the accepted state's quality is
    // visible in the output. Order mirrors the code's own precedence.
    {
        using RS = ConvergenceReport::CnvRelaxSource;
        const auto source = relax_dsol_cnv          ? RS::SolChange
                          : relax_pv_fraction_cnv   ? RS::PvFraction
                          : relax_final_iteration_cnv ? RS::FinalIter
                          : relax_iter_cnv          ? RS::IterCount
                                                    : RS::None;
        report.setCnvRelaxation(source, static_cast<double>(tol_cnv));
    }
    const auto tol_cnv_energy = use_relaxed_cnv ? this->param_.tolerance_cnv_energy_relaxed_ : this->param_.tolerance_cnv_energy_;
    const auto tol_eb = use_relaxed_mb ?  this->param_.tolerance_energy_balance_relaxed_ : this->param_.tolerance_energy_balance_;

    // Finish computation
    std::vector<Scalar> CNV(numComp);
    std::vector<Scalar> mass_balance_residual(numComp);
    for (int compIdx = 0; compIdx < numComp; ++compIdx)
    {
        CNV[compIdx]                    = B_avg[compIdx] * dt * maxCoeff[compIdx];
        mass_balance_residual[compIdx]  = std::abs(B_avg[compIdx]*R_sum[compIdx]) * dt / pvSum;
        residual_norms.push_back(CNV[compIdx]);
    }
    // Exposed to the a posteriori Criteria_newton diagnostic (the field-level MB,
    // matching what the simulator's own convergence check actually enforces).
    this->last_mass_balance_residual_ = *std::max_element(mass_balance_residual.begin(),
                                                           mass_balance_residual.end());

    using CR = ConvergenceReport;
    for (int compIdx = 0; compIdx < numComp; ++compIdx) {
        const Scalar res[2] = {
            mass_balance_residual[compIdx], CNV[compIdx],
        };

        const CR::ReservoirFailure::Type types[2] = {
            CR::ReservoirFailure::Type::MassBalance,
            CR::ReservoirFailure::Type::Cnv,
        };

        Scalar tol[2] = { tol_mb, tol_cnv, };
        if (has_energy_ && compIdx == contiEnergyEqIdx) {
            tol[0] = tol_eb;
            tol[1] = tol_cnv_energy;
        }

        this->addReservoirConvergenceMetrics(
            report,
            compIdx,
            this->compNames_.name(compIdx),
            std::span<const Scalar>{res},
            std::span<const CR::ReservoirFailure::Type>{types},
            std::span<const Scalar>{tol},
            maxResidualAllowed(),
            [this](const std::string& message)
            {
                if (this->terminal_output_) {
                    OpmLog::debug(message);
                }
            });
    }

    // Compute the Newton convergence per cell.
    this->convergencePerCell(B_avg, dt, tol_cnv, tol_cnv_energy);

    // Output of residuals.
    if (this->terminal_output_) {
        // Only rank 0 does print to std::cout
        if (iterCtx.isFirstGlobalIteration()) {
            std::string msg = "Iter";
            for (int compIdx = 0; compIdx < numComp; ++compIdx) {
                msg += "    MB(";
                msg += this->compNames_.name(compIdx)[0];
                msg += ")  ";
            }

            for (int compIdx = 0; compIdx < numComp; ++compIdx) {
                msg += "    CNV(";
                msg += this->compNames_.name(compIdx)[0];
                msg += ") ";
            }

            if (use_dsol_tol) {
                msg += use_dp_tol ? "    DP     " : "";
                msg += use_ds_tol ? "    DS     " : "";
                msg += use_drs_tol ? "    DRS    " : "";
                msg += use_drv_tol ? "    DRV    " : "";
            }

            msg += "   MBFLAG";
            msg += "  CNVFLAG";

            OpmLog::debug(msg);
        }

        std::ostringstream ss;
        const std::streamsize oprec = ss.precision(3);
        const std::ios::fmtflags oflags = ss.setf(std::ios::scientific);

        ss << std::setw(4) << iterCtx.iteration();
        for (int compIdx = 0; compIdx < numComp; ++compIdx) {
            ss << std::setw(11) << mass_balance_residual[compIdx];
        }

        for (int compIdx = 0; compIdx < numComp; ++compIdx) {
            ss << std::setw(11) << CNV[compIdx];
        }

        if (use_dsol_tol) {
            auto print_dsol =
            [&] (bool use_tol, Scalar dsol) {
                if (!use_tol) {
                    return;
                }
                if (iterCtx.isFirstGlobalIteration() || dsol <= 0.0) {
                    ss << std::string(5, ' ') << "-" << std::string(5, ' ');
                }
                else {
                    ss << std::setw(11) << dsol;
                }
            };
            print_dsol(use_dp_tol, maxSolUpd.dPMax);
            print_dsol(use_ds_tol, maxSolUpd.dSMax);
            print_dsol(use_drs_tol, maxSolUpd.dRsMax);
            print_dsol(use_drv_tol, maxSolUpd.dRvMax);
        }

        const auto mb_flag = use_relaxed_mb
            ? DebugFlags::RELAXED
            : DebugFlags::STRICT;

        const auto cnv_flag = relax_dsol_cnv ?
            DebugFlags::TUNINGDP
            : (use_relaxed_cnv
               ? DebugFlags::RELAXED
               : DebugFlags::STRICT);

        ss << std::setw(9) << make_string<TypeTag>(mb_flag)
           << std::setw(9) << make_string<TypeTag>(cnv_flag);

        ss.precision(oprec);
        ss.flags(oflags);

        OpmLog::debug(ss.str());
    }

    return report;
}

template <class TypeTag>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
convergencePerCell(const std::vector<Scalar>& B_avg,
                   const double dt,
                   const double tol_cnv,
                   const double tol_cnv_energy)
{
    auto& rst_conv = this->simulator_.problem().eclWriter().mutableOutputModule().getConv();
    if (!rst_conv.hasConv()) {
        return;
    }

    if (this->simulator_.problem().iterationContext().isFirstGlobalIteration()) {
        rst_conv.prepareConv();
    }

    const auto& residual = this->simulator_.model().linearizer().residual();
    const auto& gridView = this->simulator_.gridView();
    const IsNumericalAquiferCell isNumericalAquiferCell(gridView.grid());
    ElementContext elemCtx(this->simulator());
    std::vector<int> convNewt(residual.size(), 0);
    OPM_BEGIN_PARALLEL_TRY_CATCH();
    unsigned idx = 0;
    const int numComp = B_avg.size();
    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
        elemCtx.updatePrimaryStencil(elem);

        const unsigned cell_idx = elemCtx.globalSpaceIndex(/*spaceIdx=*/0, /*timeIdx=*/0);
        const auto pvValue = this->simulator_.problem().referencePorosity(cell_idx, /*timeIdx=*/0) *
                             this->simulator_.model().dofTotalVolume(cell_idx);
        for (int compIdx = 0; compIdx < numComp; ++compIdx) {
            const auto tol = (has_energy_ && compIdx == contiEnergyEqIdx) ? tol_cnv_energy : tol_cnv;
            const Scalar cnv = std::abs(B_avg[compIdx] * residual[cell_idx][compIdx]) * dt / pvValue;
            if (std::isnan(cnv) || cnv > maxResidualAllowed() || cnv < 0.0 || cnv > tol) {
                convNewt[idx] = 1;
                break;
            }
        }
        ++idx;
    }
    OPM_END_PARALLEL_TRY_CATCH("NonlinearSystemBlackOilReservoir::convergencePerCell() failed: ",
                               this->grid_.comm());
    rst_conv.updateNewton(convNewt);
}

template <class TypeTag>
ConvergenceReport
NonlinearSystemBlackOilReservoir<TypeTag>::
getConvergence(const SimulatorTimerInterface& timer,
               const int maxIter,
               std::vector<Scalar>& residual_norms)
{
    OPM_TIMEBLOCK(getConvergence);
    // Get convergence reports for reservoir and wells.
    std::vector<Scalar> B_avg(numEq, 0.0);
    auto report = getReservoirConvergence(timer.simulationTimeElapsed(),
                                          timer.currentStepLength(),
                                          maxIter, B_avg, residual_norms);
    {
        OPM_TIMEBLOCK(getWellConvergence);
        report += this->wellModel().getWellConvergence(B_avg,
                                                       /*checkWellGroupControlsAndNetwork*/report.converged());
    }

    conv_monitor_.checkPenaltyCard(report, this->simulator_.problem().iterationContext().iteration());

    return report;
}

template <class TypeTag>
std::vector<std::vector<typename NonlinearSystemBlackOilReservoir<TypeTag>::Scalar> >
NonlinearSystemBlackOilReservoir<TypeTag>::
computeFluidInPlace(const std::vector<int>& /*fipnum*/) const
{
    OPM_TIMEBLOCK(computeFluidInPlace);
    // assert(true)
    // return an empty vector
    std::vector<std::vector<Scalar> > regionValues(0, std::vector<Scalar>(0,0.0));
    return regionValues;
}

template <class TypeTag>
const SimulatorReport&
NonlinearSystemBlackOilReservoir<TypeTag>::
localAccumulatedReports() const
{
    if (!hasNlddSolver()) {
        OPM_THROW(std::runtime_error, "Cannot get local reports from a model without NLDD solver");
    }
    return nlddSolver_->localAccumulatedReports();
}

template <class TypeTag>
const std::vector<SimulatorReport>&
NonlinearSystemBlackOilReservoir<TypeTag>::
domainAccumulatedReports() const
{
    if (!nlddSolver_)
        OPM_THROW(std::runtime_error, "Cannot get domain reports from a model without NLDD solver");
    return nlddSolver_->domainAccumulatedReports();
}

template <class TypeTag>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
writeNonlinearIterationsPerCell(const std::filesystem::path& odir) const
{
    if (hasNlddSolver()) {
        nlddSolver_->writeNonlinearIterationsPerCell(odir);
    }
}

template <class TypeTag>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
writePartitions(const std::filesystem::path& odir) const
{
    if (hasNlddSolver()) {
        nlddSolver_->writePartitions(odir);
        return;
    }

    const auto& elementMapper = this->simulator().model().elementMapper();
    const auto& cartMapper = this->simulator().vanguard().cartesianIndexMapper();

    const auto& grid = this->simulator().vanguard().grid();
    const auto& comm = grid.comm();
    const auto nDigit = 1 + static_cast<int>(std::floor(std::log10(comm.size())));

    std::ofstream pfile {odir / fmt::format("{1:0>{0}}", nDigit, comm.rank())};

    for (const auto& cell : elements(grid.leafGridView(), Dune::Partitions::interior)) {
        pfile << comm.rank() << ' '
              << cartMapper.cartesianIndex(elementMapper.index(cell)) << ' '
              << comm.rank() << '\n';
    }
}

template <class TypeTag>
template<class FluidState, class Residual>
void
NonlinearSystemBlackOilReservoir<TypeTag>::
getMaxCoeff(const unsigned cell_idx,
            const IntensiveQuantities& intQuants,
            const FluidState& fs,
            const Residual& modelResid,
            const Scalar pvValue,
            std::vector<Scalar>& B_avg,
            std::vector<Scalar>& R_sum,
            std::vector<Scalar>& maxCoeff,
            std::vector<int>& maxCoeffCell)
{
    for (unsigned phaseIdx = 0; phaseIdx < FluidSystem::numPhases; ++phaseIdx)
    {
        if (!FluidSystem::phaseIsActive(phaseIdx)) {
            continue;
        }

        const unsigned sIdx = FluidSystem::solventComponentIndex(phaseIdx);
        const unsigned compIdx = FluidSystem::canonicalToActiveCompIdx(sIdx);

        B_avg[compIdx] += 1.0 / fs.invB(phaseIdx).value();
        const auto R2 = modelResid[cell_idx][compIdx];

        R_sum[compIdx] += R2;
        const Scalar Rval = std::abs(R2) / pvValue;
        if (Rval > maxCoeff[compIdx]) {
            maxCoeff[compIdx] = Rval;
            maxCoeffCell[compIdx] = cell_idx;
        }
    }

    if constexpr (has_solvent_) {
        B_avg[contiSolventEqIdx] +=
            1.0 / intQuants.solventInverseFormationVolumeFactor().value();
        const auto R2 = modelResid[cell_idx][contiSolventEqIdx];
        R_sum[contiSolventEqIdx] += R2;
        maxCoeff[contiSolventEqIdx] = std::max(maxCoeff[contiSolventEqIdx],
                                               std::abs(R2) / pvValue);
    }
    if constexpr (has_extbo_) {
        B_avg[contiZfracEqIdx] += 1.0 / fs.invB(FluidSystem::gasPhaseIdx).value();
        const auto R2 = modelResid[cell_idx][contiZfracEqIdx];
        R_sum[ contiZfracEqIdx ] += R2;
        maxCoeff[contiZfracEqIdx] = std::max(maxCoeff[contiZfracEqIdx],
                                             std::abs(R2) / pvValue);
    }
    if constexpr (has_polymer_) {
        B_avg[contiPolymerEqIdx] += 1.0 / fs.invB(FluidSystem::waterPhaseIdx).value();
        const auto R2 = modelResid[cell_idx][contiPolymerEqIdx];
        R_sum[contiPolymerEqIdx] += R2;
        maxCoeff[contiPolymerEqIdx] = std::max(maxCoeff[contiPolymerEqIdx],
                                               std::abs(R2) / pvValue);
    }
    if constexpr (has_foam_) {
        B_avg[ contiFoamEqIdx ] += 1.0 / fs.invB(FluidSystem::gasPhaseIdx).value();
        const auto R2 = modelResid[cell_idx][contiFoamEqIdx];
        R_sum[contiFoamEqIdx] += R2;
        maxCoeff[contiFoamEqIdx] = std::max(maxCoeff[contiFoamEqIdx],
                                            std::abs(R2) / pvValue);
    }
    if constexpr (has_brine_) {
        B_avg[ contiBrineEqIdx ] += 1.0 / fs.invB(FluidSystem::waterPhaseIdx).value();
        const auto R2 = modelResid[cell_idx][contiBrineEqIdx];
        R_sum[contiBrineEqIdx] += R2;
        maxCoeff[contiBrineEqIdx] = std::max(maxCoeff[contiBrineEqIdx],
                                             std::abs(R2) / pvValue);
    }

    if constexpr (has_polymermw_) {
        static_assert(has_polymer_);

        B_avg[contiPolymerMWEqIdx] += 1.0 / fs.invB(FluidSystem::waterPhaseIdx).value();
        // the residual of the polymer molecular equation is scaled down by a 100, since molecular weight
        // can be much bigger than 1, and this equation shares the same tolerance with other mass balance equations
        // TODO: there should be a more general way to determine the scaling-down coefficient
        const auto R2 = modelResid[cell_idx][contiPolymerMWEqIdx] / 100.;
        R_sum[contiPolymerMWEqIdx] += R2;
        maxCoeff[contiPolymerMWEqIdx] = std::max(maxCoeff[contiPolymerMWEqIdx],
                                                 std::abs(R2) / pvValue);
    }

    if constexpr (has_energy_) {
        B_avg[contiEnergyEqIdx] += 1.0;
        const auto R2 = modelResid[cell_idx][contiEnergyEqIdx];
        R_sum[contiEnergyEqIdx] += R2;
        maxCoeff[contiEnergyEqIdx] = std::max(maxCoeff[contiEnergyEqIdx],
                                              std::abs(R2) / pvValue);
    }

    if constexpr (has_bioeffects_) {
        B_avg[contiMicrobialEqIdx] += 1.0 / fs.invB(FluidSystem::waterPhaseIdx).value();
        const auto R1 = modelResid[cell_idx][contiMicrobialEqIdx];
        R_sum[contiMicrobialEqIdx] += R1;
        maxCoeff[contiMicrobialEqIdx] = std::max(maxCoeff[contiMicrobialEqIdx],
                                                std::abs(R1) / pvValue);
        B_avg[contiBiofilmEqIdx] += 1.0 / fs.invB(FluidSystem::waterPhaseIdx).value();
        const auto R2 = modelResid[cell_idx][contiBiofilmEqIdx];
        R_sum[contiBiofilmEqIdx] += R2;
        maxCoeff[contiBiofilmEqIdx] = std::max(maxCoeff[contiBiofilmEqIdx],
                                               std::abs(R2) / pvValue);
        if constexpr (has_micp_) {
            B_avg[contiOxygenEqIdx] += 1.0 / fs.invB(FluidSystem::waterPhaseIdx).value();
            const auto R3 = modelResid[cell_idx][contiOxygenEqIdx];
            R_sum[contiOxygenEqIdx] += R3;
            maxCoeff[contiOxygenEqIdx] = std::max(maxCoeff[contiOxygenEqIdx],
                                                std::abs(R3) / pvValue);
            B_avg[contiUreaEqIdx] += 1.0 / fs.invB(FluidSystem::waterPhaseIdx).value();
            const auto R4 = modelResid[cell_idx][contiUreaEqIdx];
            R_sum[contiUreaEqIdx] += R4;
            maxCoeff[contiUreaEqIdx] = std::max(maxCoeff[contiUreaEqIdx],
                                                std::abs(R4) / pvValue);
            B_avg[contiCalciteEqIdx] += 1.0 / fs.invB(FluidSystem::waterPhaseIdx).value();
            const auto R5 = modelResid[cell_idx][contiCalciteEqIdx];
            R_sum[contiCalciteEqIdx] += R5;
            maxCoeff[contiCalciteEqIdx] = std::max(maxCoeff[contiCalciteEqIdx],
                                                std::abs(R5) / pvValue);
            }
    }
}

} // namespace Opm

#endif // OPM_NONLINEAR_SYSTEM_BLACK_OIL_RESERVOIR_IMPL_HEADER_INCLUDED
