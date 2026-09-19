/*
  Copyright 2015 SINTEF ICT, Applied Mathematics.

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

#ifndef OPM_BLACKOILMODELPARAMETERS_HEADER_INCLUDED
#define OPM_BLACKOILMODELPARAMETERS_HEADER_INCLUDED

#include <opm/simulators/flow/SubDomain.hpp>

#include <string>

namespace Opm::Parameters {

template<class Scalar>
struct DbhpMaxRel { static constexpr Scalar value = 1.0; };

template<class Scalar>
struct DwellFractionMax { static constexpr Scalar value = 0.2; };

struct EclDeckFileName { static constexpr auto value = ""; };

template<class Scalar>
struct InjMultOscThreshold { static constexpr Scalar value = 0.1; };

template<class Scalar>
struct InjMultDampMult { static constexpr Scalar value = 0.9; };

template<class Scalar>
struct InjMultMinDampFactor { static constexpr Scalar value = 0.05; };

template<class Scalar>
struct MaxResidualAllowed { static constexpr Scalar value = 1e7; };

template<class Scalar>
struct RelaxedMaxPvFraction { static constexpr Scalar value = 0.03; };

template<class Scalar>
struct ToleranceMb { static constexpr Scalar value = 1e-7; };

template<class Scalar>
struct ToleranceMbRelaxed { static constexpr Scalar value = 1e-6; };

//TODO change to a simpler number with fewer digits
//converting J -> RM3 (entalpy / (cp * deltaK * rho) assuming change of 1e-5K of water
template<class Scalar>
struct ToleranceEnergyBalance { static constexpr Scalar value = 1e-7*41.82; };

template<class Scalar>
struct ToleranceEnergyBalanceRelaxed { static constexpr Scalar value = 1e-6*41.82; };

template<class Scalar>
struct ToleranceCnv { static constexpr Scalar value = 1e-2; };

template<class Scalar>
struct ToleranceCnvRelaxed { static constexpr Scalar value = 1.0; };

template<class Scalar>
struct ToleranceCnvEnergy { static constexpr Scalar value = 1e-2*41.82; };

template<class Scalar>
struct ToleranceCnvEnergyRelaxed { static constexpr Scalar value = 1.0*41.82; };

template<class Scalar>
struct ToleranceMaxDp { static constexpr Scalar value = 0.0; };

template<class Scalar>
struct ToleranceMaxDs { static constexpr Scalar value = 0.0; };

template<class Scalar>
struct ToleranceMaxDrs { static constexpr Scalar value = 0.0; };

template<class Scalar>
struct ToleranceMaxDrv { static constexpr Scalar value = 0.0; };

template<class Scalar>
struct ToleranceWells { static constexpr Scalar value = 1e-4; };

template<class Scalar>
struct ToleranceWellControl { static constexpr Scalar value = 1e-7; };

struct MaxWelleqIter { static constexpr int value = 30; };

template<class Scalar>
struct MaxSinglePrecisionDays { static constexpr Scalar value = 20.0; };

struct MinStrictCnvIter { static constexpr int value = -1; };
struct MinStrictMbIter { static constexpr int value = -1; };

// Inexact-Newton adaptive tolerance for the linear solve (Eisenstat--Walker).
struct AdaptiveLinearSolverReduction { static constexpr bool value = false; };

template<class Scalar>
struct AdaptiveLinearSolverReductionGamma { static constexpr Scalar value = 0.9; };

template<class Scalar>
struct AdaptiveLinearSolverReductionMax { static constexpr Scalar value = 0.1; };

// Only loosen the linear tolerance if the previous linear solve took at least
// this many iterations (below it the preconditioner already overshoots and
// loosening only hurts the Newton update).
struct AdaptiveLinearSolverReductionMinIter { static constexpr int value = 10; };

// Evaluate the a posteriori spatial (eta_sp) and temporal (eta_time) error
// estimators on each converged step and print the balancing table.
struct EnableAposterioriEstimators { static constexpr bool value = false; };

// Replace only the reservoir CNV stopping test by Criteria_newton. Material
// balance, wells, groups, network balancing, minimum iterations, and severe
// convergence failures remain governed by the standard OPM checks.
struct EnableAposterioriNewtonStopping { static constexpr bool value = false; };

// Drive the inexact-Newton linear-solve tolerance from Criteria_alg: target =
// Gamma_alg * max(eta_sp,eta_time) / eta_alg^(0), with eta_alg^(0) the weighted
// algebraic estimator of the well-eliminated residual (evaluated after
// wellModel().linearize()). The weighted eta_alg after the solve is measured
// and, if it still exceeds 1.2*target, corrected by up to
// --aposteriori-alg-max-resolves further guarded re-solves, each one
// re-deriving its own reduction target from the just-measured shortfall
// rather than repeating the first guess. See
// AposterioriAlgMaxResolves's own comment for why this reconstructed-residual
// correction is trustworthy: on the L40 five-spot diagnostic sweep
// (2026-09-13, apost_lindump_*.csv, 8 Newton solves spanning the full 220-day
// run), eta_alg(r) followed a power law C*r^p in the solver's own relative
// residual reduction r with p in [0.968,0.986] and R^2>0.998 in every case --
// i.e. very close to the linear model this formula already assumes, with
// <=1.5x error at the single first-solve measurement. Requires
// --enable-aposteriori-estimators.
struct EnableAposterioriLinearTolerance { static constexpr bool value = false; };

// Build eta_lin from the rigorous Newton-linearized flux defect (Theta_lin
// via a two-pass local Jacobian-vector product in recordLinearizationDefect).
// Default true. Setting it false skips recordLinearizationDefect entirely --
// the single biggest per-iteration estimator cost -- and eta_lin falls back
// to the cheap iterate-to-iterate flux-change proxy. The proxy is enough for
// the Newton-stopping decision (eta_lin only needs to be small relative to
// Gamma_lin*max(eta_sp,eta_time)); the rigorous term is only worth its cost
// when eta_lin itself is being reported/studied.
struct AposterioriRigorousLin { static constexpr bool value = true; };

// Evaluate ALL the *,K energy-norm terms (eta_D, eta_time, eta_lin's flux
// term, and the eta_lin iterate-diff proxy) on the same footing. Default
// false = full rigour: eta_D / eta_lin-flux via the mimetic matrix
// z^T M_K z (full permeability tensor + stability term), eta_time / proxy
// via the full-tensor P0 form D_K^l |K| (v.K^{-1}v). True = the cheap
// approximation for all of them: eta_D / eta_lin-flux drop to the T1-only
// Pi0 moment (diagonal K, no stability), eta_time / proxy to the diagonal
// |K_diag^{-1/2} v|. eta_lin's accumulation-defect term follows the paper's
// own c_KK^{-1/2} scalar formula either way.
struct AposterioriCheapNorms { static constexpr bool value = false; };

// Ablation switch: drop the c_KK (smallest eigenvalue of K|_K) weight from
// the Neumann-restoring zeroth-order term of the equilibration indicator
// eta_eq,K (eq. eps_norm in the paper), i.e. evaluate it as if c_KK=1 on
// every cell instead of the local permeability. Default false = paper
// formula (both terms of the augmented norm carry units of permeability,
// giving K-robustness on a high-contrast deck). True is provided only to
// see empirically how much that weighting matters on such a deck; it is
// not expected to be reliable on strongly heterogeneous permeability.
struct AposterioriDisableCkkWeight { static constexpr bool value = false; };

// Experimental Neumann quotient-space treatment for the spatial estimator.
// eta_eq remains computed and reported, but is not added to eta_sp or the
// per-cell/component fields used for mesh marking. The global constant mode
// remains guarded by OPM's material-balance test. Default false preserves the
// paper implementation until reference-effectivity tests justify a change.
struct AposterioriSeparateNeumannMean { static constexpr bool value = false; };

// Relative diagonal regularisation for the component mobility matrix induced
// by Appendix A.4: L = C diag(lambda_beta) C^T. The added diagonal is this
// fraction times max(lambda_beta) times the squared norm of that component's
// row of C. This preserves the norm under consistent component-unit scaling.
// It handles phase disappearance and a nearly singular oil/gas mixing matrix
// but is not part of the guaranteed K^{-1} residual estimator.
template<class Scalar>
struct AposterioriMobilityFloorFraction { static constexpr Scalar value = 1e-12; };

// Experimental coupled black-oil energy. Instead of measuring each conserved
// mass-component defect independently in K^{-1}, use the mobility induced by
// u=Cv: (C diag(lambda_beta) C^T tensor K)^{-1}. This includes b, Rs and Rv
// and operates on the actual component defect. It is an effectivity experiment,
// not part of the current component-residual reliability result.
struct AposterioriMobilityEnergyNorm { static constexpr bool value = false; };

// Use the complete spatial estimator (Darcy plus local equilibration defect)
// as the stopping/control budget. False keeps the Darcy-only budget, which
// avoids letting the current nonlinear residual enlarge its own tolerance.
struct AposterioriUseTotalSpatialBudget { static constexpr bool value = false; };

// Skip the (expensive) per-iteration estimator evaluation -- compute() and
// recordLinearizationDefect() -- on Newton iterations before this one. The
// early iterates are never a Criteria_newton accept candidate (the plateau
// guard needs >=2 evaluated rows and iteration>=2 anyway), so evaluating
// them only costs time. 1 = evaluate every iteration (default, unchanged
// behaviour).
struct AposterioriFirstEvalIter { static constexpr int value = 1; };

// Actually drive the next AdaptiveTimeStepping step size from the a
// posteriori space/time-balance rescale (eq. Criteria_space_time_balance)
// instead of only reporting it. Requires EnableAposterioriEstimators; a
// no-op otherwise. Newton/linear stopping are NOT affected -- only the
// suggested next dt. Applied inside AdaptiveTimeStepping's substep loop, so
// it governs every substep. A runaway (a report period over-refined past
// ~40 substeps, or >250 estimator substeps over the run) latches the
// override OFF for the rest of the run and native control resumes.
struct EnableAposterioriTimestepControl { static constexpr bool value = false; };

// Lower/upper edge of the space/time balancing band, eq. Criteria_space_time_balance:
// gamma_time * eta_sp <= eta_time <= Gamma_time * eta_sp.
template<class Scalar>
struct AposterioriGammaTime { static constexpr Scalar value = 0.5; };
template<class Scalar>
struct AposterioriGammaTimeUpper { static constexpr Scalar value = 2.0; };

// The Neumann-scaled weight exponent l in (0,2) (eq. eq:eps_norm / Assumption
// assum:ell): D_K^l bounds the near-well weight d_Lambda^l. l=0 (default)
// reproduces the plain energy norm with no near-well damping.
template<class Scalar>
struct AposterioriWeightExponent { static constexpr Scalar value = 0.0; };

// Evaluate lambda_beta(s_hat) at the lifted vertex-patch saturation via the
// real MaterialLaw instead of the FV cell mobility. Off by operational
// default because patch averaging can smear sharp fronts.
struct AposterioriUseLiftedRelperm { static constexpr bool value = false; };

// Point-value bubble correction of the lifted saturation/pressure to the FV
// cell mean (eq. eq:averaging_bubble, point-value form only).
struct AposterioriUseBubbleCorrection { static constexpr bool value = true; };

// Use the transmissibility-weighted least-squares fit of the raw connection
// pressure drops for grad p_hat. This is the operational default because it
// produced decreasing estimates under refinement on both five-spot and SPE10
// layer 85. Set false to recover the paper's vertex patch-average H1 lift.
struct AposterioriUseConnectionLSGradient { static constexpr bool value = true; };

// Build the H1 pressure lift by first reconstructing a Darcy-flux-consistent
// pressure gradient in every cell, Taylor-extrapolating the cell pressure to
// its vertices, and averaging the extrapolated values over each vertex patch.
// When false, the default H1 lift averages the raw cell pressures at vertices.
// Takes precedence over AposterioriUseConnectionLSGradient when enabled.
struct AposterioriUseFluxTaylorPressure { static constexpr bool value = false; };

// The admissible relative linearization error Gamma_lin in (0,1] (eq.
// Criteria_newton): eta_lin <= Gamma_lin * max(eta_sp, eta_time).
template<class Scalar>
struct AposterioriGammaLin { static constexpr Scalar value = 0.01; };

// Maximum relative change of eta_sp and eta_time between successive Newton
// iterates before Criteria_newton may accept. Set to zero to apply the bare
// estimator criterion without the empirical plateau guard.
template<class Scalar>
struct AposterioriNewtonPlateauTolerance { static constexpr Scalar value = 0.0; };

// The admissible relative algebraic error Gamma_alg in (0,1] (eq.
// Criteria_alg): used by --enable-aposteriori-linear-tolerance as the
// linear-solve forcing term Gamma_alg * max(eta_sp,eta_time) / eta_alg^(0).
template<class Scalar>
struct AposterioriGammaAlg { static constexpr Scalar value = 0.01; };

// How many full-increment correction solves endpoint-verified Criteria_alg may
// perform. A correction recomputes the full increment from zero and tightens
// the preceding request by 0.5*target/eta_alg. If the target remains unmet,
// estimator Newton acceptance
// is disabled for that iteration and the next Newton solve is latched to the
// configured strict reduction. 0 keeps prediction-only mode; 1 enables the
// endpoint-verified controller.
struct AposterioriAlgMaxResolves { static constexpr int value = 2; };

// Material-balance tolerance for the a posteriori Criteria_newton gate. The
// paper makes MB non-negotiable, so the default (<= 0) means "use the
// simulator's own tolerance_mb_" -- identical to OPM's own convergence
// requirement, i.e. the a posteriori criterion can never accept a state OPM
// itself would reject on MB. A positive value overrides it: larger relaxes
// the gate (the extreme, ~1.0, disables it -- the field-scaled MB residual
// is always below that), letting Criteria_newton stop Newton before MB has
// converged to the strict tolerance. That trades global mass conservation
// for iteration count and can drift field totals over a run -- experimental.
template<class Scalar>
struct AposterioriTolMb { static constexpr Scalar value = -1.0; };

// The Neumann-scaling parameter epsilon > 0 (eq. eq:eps_norm), used in the
// nonlinear-accumulation-defect term of eta_lin. Paper recommendation: 1.
template<class Scalar>
struct AposterioriEpsilon { static constexpr Scalar value = 1.0; };

// mvemFluxMassMatrix's stability-term floor: D_K[f] = max(consistency part,
// this * trace(M^c)/nf). Was hardcoded to 1e-2 with no override; exposed here
// to test whether the MVEM stability term M^s's failure to shrink under
// h-refinement (confirmed empirically on SPE3: rigorous eta_sp grows under
// refinement while the M^s-free cheap/T1-only norm shrinks correctly) is
// sensitive to this floor. Mimetic theory (Lemma 3.7) only guarantees
// spectral equivalence at this floor, not that it vanishes with h.
template<class Scalar>
struct AposterioriMvemStabilityEpsilon { static constexpr Scalar value = 1e-2; };

// Maximum per-rescale growth/shrink factor applied to the suggested next dt
// (eq. Criteria_space_time_balance). Only reached when
// EnableAposterioriTimestepGrowthOverride=true (bidirectional/experimental
// mode); the production default (that flag false) never lets the estimator
// request more growth than AdaptiveTimeStepping's own native suggestion --
// see that flag's doc comment. Default maxGrow=1.25 is a conservative choice
// for the experimental bidirectional mode: a comparison on SPE9 (2026-09-04)
// found that a maxGrow of 3.0 saturated on nearly every step (this deck's
// eta_time/eta_sp ratio sits far below the default band), making the raw
// override strictly more aggressive than AdaptiveTimeStepping's own
// heuristic (~2.2x observed cap) and producing MORE Newton/linear iterations
// and oscillation events than leaving timestep control off entirely.
template<class Scalar>
struct AposterioriMaxGrow { static constexpr Scalar value = 1.25; };
template<class Scalar>
struct AposterioriMaxShrink { static constexpr Scalar value = 0.5; };

// Production default (false): the a posteriori override acts purely as a
// LIMITER on AdaptiveTimeStepping's own native suggestion --
// min(nativeDt, estimatorDt) -- so OPM's own convergence-history-based
// growth heuristic (and --solver-max-growth) governs all growth, and the
// estimator can only ever SHRINK a step, when eta_time is excessive relative
// to eta_sp. A small eta_time/eta_sp ratio (the common case observed so far)
// therefore cannot force aggressive growth on its own.
// Set true to instead let the estimator's rescale directly override the
// native suggestion in both directions (bounded by AposterioriMaxGrow /
// AposterioriMaxShrink) -- an experimental mode for studying the estimator's
// own growth policy in isolation from AdaptiveTimeStepping's heuristic.
struct EnableAposterioriTimestepGrowthOverride { static constexpr bool value = false; };

struct SolveWelleqInitially { static constexpr bool value = true; };
struct PreSolveNetwork { static constexpr bool value = true; };
struct UpdateEquationsScaling { static constexpr bool value = false; };
struct UseUpdateStabilization { static constexpr bool value = true; };
struct MatrixAddWellContributions { static constexpr bool value = false; };

struct UseMultisegmentWell { static constexpr bool value = true; };

// Convert plain (single-segment) wells into multisegment wells at
// construction, so the wellbore hydrostatic head is handled implicitly.
// A string rather than a bool because more than one topology is useful:
// "per-connection" gives one segment per connection. "none" is the default.
struct ConvertToMultisegmentWell { static constexpr auto value = "none"; };

template<class Scalar>
struct TolerancePressureMsWells { static constexpr Scalar value = 0.01*1e5; };

template<class Scalar>
struct MaxPressureChangeMsWells { static constexpr Scalar value = 10*1e5; };

struct MaxNewtonIterationsWithInnerWellIterations { static constexpr int value = 99; };
struct MaxInnerIterMsWells { static constexpr int value = 100; };
struct MaxInnerIterWells { static constexpr int value = 50; };
struct MaxWellStatusSwitchInInnerIterWells { static constexpr int value = 99; };
struct MaxWellStatusSwitchForWells { static constexpr int value = 99; };
struct ShutUnsolvableWells { static constexpr bool value = true; };
struct AlternativeWellRateInit { static constexpr bool value = true; };
struct StrictOuterIterWells { static constexpr int value = 6; };
struct StrictInnerIterWells { static constexpr int value = 40; };

template<class Scalar>
struct RegularizationFactorWells { static constexpr Scalar value = 100.0; };

struct EnableWellOperabilityCheck { static constexpr bool value = true; };
struct EnableWellOperabilityCheckIter { static constexpr bool value = false; };
struct DebugEmitCellPartition { static constexpr bool value = false; };

template<class Scalar>
struct RelaxedWellFlowTol { static constexpr Scalar value = 1e-3; };

template<class Scalar>
struct RelaxedPressureTolMsw { static constexpr Scalar value = 1e4; };

struct MaximumNumberOfWellSwitches { static constexpr int value = 3; };
struct MaximumNumberOfGroupSwitches { static constexpr int value = 3; };
struct UseAverageDensityMsWells { static constexpr bool value = false; };
struct LocalWellSolveControlSwitching { static constexpr bool value = true; };
struct UseImplicitIpr { static constexpr bool value = true; };
struct CheckGroupConstraintsInnerWellIterations { static constexpr bool value = true; };

// Network solver parameters
struct NetworkMaxStrictOuterIterations { static constexpr int value = 10; };
struct NetworkMaxOuterIterations { static constexpr int value = 3; };
struct NetworkMaxSubIterations { static constexpr int value = 100; };
template<class Scalar>
struct NetworkPressureUpdateDampingFactor { static constexpr Scalar value = 0.1; };
template<class Scalar>
struct NetworkMaxPressureUpdateInBars { static constexpr Scalar value = 5.0; };
// Reservoir coupling: when false (default) the master exchanges node pressures
// and slave rates with the slaves once per master inner network sub-iteration
// (tight coupling).  When true, the exchange happens only once per master outer network
// iteration (loose coupling).
struct RcNetworkLooseCoupling { static constexpr bool value = false; };
struct NonlinearSolver { static constexpr auto value = "newton"; };
struct LocalSolveApproach { static constexpr auto value = "gauss-seidel"; };
struct MaxLocalSolveIterations { static constexpr int value = 20; };
struct NewtonMinIterations { static constexpr int value = 2; };

struct WellGroupConstraintsMaxIterations { static constexpr int value = 1; };
template<class Scalar>
struct GroupControlFractionTolerance { static constexpr Scalar value = 1e-4; };
template<class Scalar>
struct LocalToleranceScalingMb { static constexpr Scalar value = 1.0; };

template<class Scalar>
struct LocalToleranceScalingCnv { static constexpr Scalar value = 0.1; };
struct NlddNumInitialNewtonIter { static constexpr int value = 1; };
template<class Scalar>
struct NlddRelativeMobilityChangeTol { static constexpr Scalar value = 0.1; };
struct NumLocalDomains { static constexpr int value = 0; };

template<class Scalar>
struct LocalDomainsPartitioningImbalance { static constexpr Scalar value = 1.03; };

struct LocalDomainsPartitioningMethod { static constexpr auto value = "zoltan"; };
struct LocalDomainsPartitionWellNeighborLevels { static constexpr int value = 1; };
struct LocalDomainsOrderingMeasure { static constexpr auto value = "maxpressure"; };

struct ConvergenceMonitoring { static constexpr bool value = false; };
struct ConvergenceMonitoringCutOff { static constexpr int value = 6; };
template<class Scalar>
struct ConvergenceMonitoringDecayFactor { static constexpr Scalar value = 0.75; };


template<class Scalar>
struct NupcolGroupRateTolerance { static constexpr Scalar value = 0.001; };

} // namespace Opm::Parameters

namespace Opm {

/// Solver parameters for the NonlinearSystemBlackOilReservoir.
template <class Scalar>
struct BlackoilModelParameters
{
public:
    /// Max relative change in bhp in single iteration.
    Scalar dbhp_max_rel_;
    /// Max absolute change in well volume fraction in single iteration.
    Scalar dwell_fraction_max_;
    /// Injectivity multiplier oscillation threshold
    Scalar inj_mult_osc_threshold_;
    /// Injectivity multiplier dampening multiplier
    Scalar inj_mult_damp_mult_;
    /// Minimum damping factor for injectivity multipliers
    Scalar inj_mult_min_damp_factor_;
    /// Absolute max limit for residuals.
    Scalar max_residual_allowed_;
    //// Max allowed pore volume faction where CNV is violated. Below the
    //// relaxed tolerance tolerance_cnv_relaxed_ is used.
    Scalar relaxed_max_pv_fraction_;
    /// Relative mass balance tolerance (total mass balance error).
    Scalar tolerance_mb_;
    /// Relaxed mass balance tolerance (can be used when iter >= min_strict_mb_iter_).
    Scalar tolerance_mb_relaxed_;
    /// Relative energy balance tolerance (total energy balance error).
    Scalar tolerance_energy_balance_;
    /// Relaxed energy balance tolerance (can be used when iter >= min_strict_mb_iter_).
    Scalar tolerance_energy_balance_relaxed_;
    /// Local convergence tolerance (max of local saturation errors).
    Scalar tolerance_cnv_;
    /// Relaxed local convergence tolerance (can be used when iter >= min_strict_cnv_iter_ && cnvViolatedPV < relaxed_max_pv_fraction_).
    Scalar tolerance_cnv_relaxed_;
    /// Local energy convergence tolerance (max of local energy errors).
    Scalar tolerance_cnv_energy_;
    /// Relaxed local energy convergence tolerance (can be used when iter >= min_strict_cnv_iter_ && cnvViolatedPV < relaxed_max_pv_fraction_).
    Scalar tolerance_cnv_energy_relaxed_;
    /// Max pressure change during a Newton iteration (TUNINGDP item = TRGDDP)
    Scalar tolerance_max_dp_;
    /// Max saturation change during a Newton iteration (TUNINGDP item = TRGDDS)
    Scalar tolerance_max_ds_;
    /// Max RS change during a Newton iteration (TUNINGDP item = TRGDDRS)
    Scalar tolerance_max_drs_;
    /// Max RV change during a Newton iteration (TUNINGDP item = TRGDDRV)
    Scalar tolerance_max_drv_;
    /// Well convergence tolerance.
    Scalar tolerance_wells_;
    /// Tolerance for the well control equations
    //  TODO: it might need to distinguish between rate control and pressure control later
    Scalar tolerance_well_control_;
    /// Tolerance for the pressure equations for multisegment wells
    Scalar tolerance_pressure_ms_wells_;
    /// Relaxed tolerance for for the well flow residual
    Scalar relaxed_tolerance_flow_well_;

    /// Relaxed tolerance for the MSW pressure solution
    Scalar relaxed_tolerance_pressure_ms_well_;

    /// Maximum pressure change over an iteratio for ms wells
    Scalar max_pressure_change_ms_wells_;

    /// Maximum inner iteration number for ms wells
    int max_inner_iter_ms_wells_;

    /// Strict inner iteration number for wells
    int strict_inner_iter_wells_;

    /// Newton iteration where wells are stricly convergent
    int strict_outer_iter_wells_;

    /// Regularization factor for wells
    Scalar regularization_factor_wells_;

    /// Maximum newton iterations with inner well iterations
    int max_niter_inner_well_iter_;

    /// Whether to shut unsolvable well
    bool shut_unsolvable_wells_;

    /// Maximum inner iteration number for standard wells
    int max_inner_iter_wells_;

    /// Maximum iteration number of the well equation solution
    int max_welleq_iter_;

    /// Tolerance for time step in seconds where single precision can be used
    /// for solving for the Jacobian
    Scalar maxSinglePrecisionTimeStep_;

    /// Minimum number of Newton iterations before we can use relaxed CNV convergence criterion
    int min_strict_cnv_iter_;

    /// Minimum number of Newton iterations before we can use relaxed MB convergence criterion
    int min_strict_mb_iter_;

    /// Use an inexact-Newton (Eisenstat--Walker) adaptive tolerance for the linear solve
    bool adaptive_linear_solver_reduction_;

    /// Safety factor for the adaptive linear-solve forcing term
    Scalar adaptive_linear_solver_reduction_gamma_;

    /// Loosest relative reduction permitted for the adaptive linear solve
    Scalar adaptive_linear_solver_reduction_max_;

    /// Minimum previous-solve linear iteration count for the adaptive linear tolerance to engage
    int adaptive_linear_solver_reduction_min_iter_;

    /// Evaluate and print the a posteriori eta_sp / eta_time balancing table
    bool enable_aposteriori_estimators_;

    /// Allow Criteria_newton to accept an iterate that fails only standard CNV
    bool enable_aposteriori_newton_stopping_;

    /// Drive the linear-solve tolerance from the Criteria_alg target
    /// Gamma_alg * max(eta_sp, eta_time) / eta_alg^(0)
    bool enable_aposteriori_linear_tolerance_;

    /// Build eta_lin rigorously (recordLinearizationDefect); false => cheap proxy
    bool aposteriori_rigorous_lin_;

    /// Evaluate all *,K energy norms cheaply (diagonal K, no stability term)
    bool aposteriori_cheap_norms_;

    /// Ablation: eta_eq,K's Neumann-restoring term uses c_KK=1 instead of
    /// the local permeability's smallest eigenvalue
    bool aposteriori_disable_ckk_weight_;

    /// Experimental H1/R spatial norm: report eta_eq separately and use the
    /// tensor Darcy estimator alone for spatial marking
    bool aposteriori_separate_neumann_mean_;

    /// Experimental coupled black-oil component-mobility energy for eta_sp
    bool aposteriori_mobility_energy_norm_;

    /// Use eta_sp,total instead of eta_sp,D as the control budget
    bool aposteriori_use_total_spatial_budget_;

    /// Relative diagonal regularisation for the induced mobility matrix
    Scalar aposteriori_mobility_floor_fraction_;

    /// mvemFluxMassMatrix's stability-term floor fraction (was hardcoded 1e-2)
    Scalar aposteriori_mvem_stability_epsilon_;

    /// Skip per-iteration estimator evaluation before this Newton iteration
    int aposteriori_first_eval_iter_;

    /// Actually drive the next timestep size from the a posteriori estimators
    bool enable_aposteriori_timestep_control_;

    /// Lower/upper edge of the eta_time/eta_sp balancing band
    Scalar aposteriori_gamma_time_;
    Scalar aposteriori_gamma_time_upper_;

    /// Neumann-scaled near-well weight exponent l (0 = inactive)
    Scalar aposteriori_weight_exponent_;

    /// Evaluate lambda_beta at the lifted (vertex patch-average) saturation
    bool aposteriori_use_lifted_relperm_;

    /// Bubble-correct the lifted point value to the FV cell mean
    bool aposteriori_use_bubble_correction_;

    /// Use the connection-drop LS gradient instead of the H1 vertex-patch lift
    bool aposteriori_use_connection_ls_gradient_;

    /// Use flux-derived cell gradients for Taylor extrapolation to H1 vertices
    bool aposteriori_use_flux_taylor_pressure_;

    /// Admissible relative linearization error Gamma_lin
    Scalar aposteriori_gamma_lin_;

    /// Relative eta_sp/eta_time plateau tolerance; zero disables the guard
    Scalar aposteriori_newton_plateau_tolerance_;

    /// Admissible relative algebraic error Gamma_alg (linear-solve forcing term)
    Scalar aposteriori_gamma_alg_;

    /// Max extra tighter linear re-solves to enforce the weighted Criteria_alg
    int aposteriori_alg_max_resolves_;

    /// MB tolerance for the Criteria_newton gate (<=0 => use tolerance_mb_)
    Scalar aposteriori_tol_mb_;

    /// Neumann-scaling parameter epsilon > 0
    Scalar aposteriori_epsilon_;

    /// Max per-rescale growth/shrink factor on the suggested next dt
    /// (only reached when aposteriori_timestep_growth_override_ is true)
    Scalar aposteriori_max_grow_;
    Scalar aposteriori_max_shrink_;

    /// false (production default): override acts as a limiter,
    /// min(native AdaptiveTimeStepping suggestion, estimator suggestion) --
    /// true (experimental): estimator suggestion applied directly, both
    /// directions, bounded by aposteriori_max_grow_/aposteriori_max_shrink_
    bool aposteriori_timestep_growth_override_;

    /// Solve well equation initially
    bool solve_welleq_initially_;

    /// Pre solve and iterate network model
    bool pre_solve_network_;

    /// Update scaling factors for mass balance equations
    bool update_equations_scaling_;

    /// Try to detect oscillation or stagnation
    bool use_update_stabilization_;

    /// Whether to use MultisegmentWell to handle multisegment wells
    /// it is something temporary before the multisegment well model is considered to be
    /// well developed and tested.
    /// if it is false, we will handle multisegment wells as standard wells, which will be
    /// the default behavoir for the moment. Later, we might set it to be true by default if necessary
    bool use_multisegment_well_;

    /// The file name of the deck
    std::string deck_file_name_;

    /// Whether to add influences of wells between cells to the matrix and preconditioner matrix
    bool matrix_add_well_contributions_;

    /// Whether to check well operability
    bool check_well_operability_;
    /// Whether to check well operability during iterations
    bool check_well_operability_iter_;

    /// Maximum number of times a well can switch to the same control
    int max_number_of_well_switches_;

    /// Maximum number of times group can switch to the same control
    int max_number_of_group_switches_;

    /// Whether to approximate segment densities by averaging over segment and its outlet
    bool use_average_density_ms_wells_;

    /// Whether to allow control switching during local well solutions
    bool local_well_solver_control_switching_;

    /// Whether to use implicit IPR for thp stability checks and solution search
    bool use_implicit_ipr_;

    /// Whether to allow checking/changing to group controls during inner well iterations
    bool check_group_constraints_inner_well_iterations_;

    /// Maximum number of iterations in the network solver before relaxing tolerance
    int network_max_strict_outer_iterations_;

    /// Maximum number of iterations in the network solver before giving up
    int network_max_outer_iterations_;

    /// Maximum number of sub-iterations to update network pressures (within a single well/group control update)
    int network_max_sub_iterations_;

    /// Damping factor in the inner network pressure update iterations
    Scalar network_pressure_update_damping_factor_;

    /// Maximum pressure update in the inner network pressure update iterations
    Scalar network_max_pressure_update_in_bars_;

    /// Reservoir coupling: use loose (per-outer-iteration) master/slave network
    /// coupling instead of the default tight (per-sub-iteration) coupling.
    bool rc_network_loose_coupling_;

    /// Maximum number of iterations in the well/group switch algorithm
    int well_group_constraints_max_iterations_;

    /// Maximum number of status switches (open<->stop> in local well iterations
    int max_well_status_switch_inner_iter_;

    /// Maximum number of status switches (open<->stop> during a time step
    int max_well_status_switch_;

    /// Lower tolerance for the guiderate ratio or well-fraction of active phase before switching to group fallback control
    Scalar group_control_fraction_tolerance_;

    /// Nonlinear solver type: newton or nldd
    std::string nonlinear_solver_;

    /// 'jacobi' and 'gauss-seidel' supported
    DomainSolveApproach local_solve_approach_{DomainSolveApproach::Jacobi};

    /// Maximum number of Newton iterations per time step
    int newton_max_iter_;

    /// Minimum number of Newton iterations per time step
    int newton_min_iter_;

    int max_local_solve_iterations_;

    Scalar local_tolerance_scaling_mb_;
    Scalar local_tolerance_scaling_cnv_;

    int nldd_num_initial_newton_iter_{1};
    /// Threshold for single cell relative mobility change in NLDD
    Scalar nldd_relative_mobility_change_tol_;
    int num_local_domains_{0};
    Scalar local_domains_partition_imbalance_{1.03};
    std::string local_domains_partition_method_;
    int local_domains_partition_well_neighbor_levels_{1};
    DomainOrderingMeasure local_domains_ordering_{DomainOrderingMeasure::MaxPressure};

    bool write_partitions_{false};

    /// Struct holding convergence monitor params
    struct ConvergenceMonitorParams
    {
        /// Whether to enable convergence monitoring
        bool enabled_;
        /// Cut-off limit for convergence monitoring
        int cutoff_;
        /// Decay factor used in convergence monitoring
        Scalar decay_factor_;
    };

    ConvergenceMonitorParams monitor_params_; //!< Convergence monitoring parameters

    // Relative tolerance of group rates (VREP, REIN)
    // If violated the nupcol wellstate is updated
    Scalar nupcol_group_rate_tolerance_;

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
      // Only dynamic state is serialized. Static configuration remains sourced
      // from CLI/deck parameters and defaults on construction.
      serializer(tolerance_cnv_);
      serializer(tolerance_cnv_relaxed_);
      serializer(tolerance_mb_);
      serializer(tolerance_mb_relaxed_);
      serializer(newton_max_iter_);
      serializer(newton_min_iter_);
      serializer(tolerance_max_dp_);
      serializer(tolerance_max_ds_);
      serializer(tolerance_max_drs_);
      serializer(tolerance_max_drv_);
    }

    static BlackoilModelParameters serializationTestObject();

    bool operator==(const BlackoilModelParameters& other) const;

    /// Construct from user parameters or defaults.
    BlackoilModelParameters();

    static void registerParameters();
};

} // namespace Opm

#endif // OPM_BLACKOILMODELPARAMETERS_HEADER_INCLUDED
