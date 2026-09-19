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

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 * \brief The three nested error-balancing criteria of the a posteriori
 *        framework of Ahmed et al. (paper Section "Balancing criteria").
 *
 * All logic here is pure arithmetic on the estimator scalars
 * (eta_sp, eta_time, eta_lin, eta_alg, eta_W, eta_st) and the user targets
 * (gamma_time, Gamma_time, Gamma_lin, Gamma_alg, Gamma_W); it carries no
 * simulator dependency and is unit tested in isolation.
 */
#ifndef OPM_APOSTERIORI_BALANCING_CRITERIA_HPP
#define OPM_APOSTERIORI_BALANCING_CRITERIA_HPP

#include <algorithm>
#include <cmath>
#include <limits>

namespace Opm::APosteriori {

/*!
 * \brief User targets for the balancing criteria.
 *
 * Defaults follow the paper's recommendations (epsilon = 1, moderate factors).
 */
template<class Scalar>
struct BalancingTargets
{
    Scalar gammaTime {Scalar{0.5}};   //!< lower band for eta_time / eta_sp
    Scalar GammaTime {Scalar{2}};     //!< upper band for eta_time / eta_sp
    Scalar GammaLin  {Scalar{0.1}};   //!< admissible relative linearization error, in (0,1]
    Scalar GammaAlg  {Scalar{0.1}};   //!< admissible relative algebraic error
    Scalar GammaW    {Scalar{1}};     //!< well / phase-state sub-problem trigger
    Scalar tolMB     {Scalar{1e-7}};  //!< non-negotiable material-balance floor
};

/*!
 * \brief Time-step balancing, eq. (Criteria_space_time_balance):
 *        gamma_time eta_sp <= eta_time <= Gamma_time eta_sp.
 */
template<class Scalar>
bool spaceTimeBalanced(Scalar etaSp, Scalar etaTime, const BalancingTargets<Scalar>& t)
{
    // Invalid / unavailable inputs: report "balanced" so nothing rescales the
    // step. Callers that must distinguish "in band" from "no usable estimate"
    // check finiteness/positivity of eta_sp themselves before acting (see the
    // discOk gate in NonlinearSystemBlackOilReservoir_impl.hpp).
    if (!(etaSp > Scalar{0}) || !std::isfinite(etaSp) || !std::isfinite(etaTime))
        return true;
    return (etaTime >= t.gammaTime * etaSp) && (etaTime <= t.GammaTime * etaSp);
}

/*!
 * \brief Re-scaling of eq. (Criteria_space_time_balance):
 *        tau <- tau * r* eta_sp / eta_time, clamped.
 *
 * The paper's own literal formula for this rescale carries a square root,
 * derived by informally invoking backward Euler's O(tau^2) local truncation
 * error. That derivation does not apply as-is to the estimators as they are
 * actually built here: with eta_sp = sqrt(tau) ||W_a - u_a||_{*,K} and
 * eta_time = sqrt(tau/3) ||u_a^n - u_a^{n-1}||_{*,K}, a Taylor expansion for a
 * smooth trajectory gives
 *
 *   ||W_a - u_a||_{*,K}      = A_0 + O(tau)          (tau-independent limit)
 *   ||u_a^n - u_a^{n-1}||_{*,K} = B_0 tau + O(tau^2)    (one-step state change)
 *
 * so eta_sp = O(sqrt(tau)) and eta_time = O(tau^{3/2}), hence the ratio
 * r(tau) = eta_time/eta_sp = O(tau) is LINEAR in tau, not quadratic. Moving
 * r(tau) to a target r* by rescaling tau therefore uses the linear law
 * tau_new = tau_old * (r* / r) = tau_old * r* eta_sp / eta_time, with no
 * square root; the sqrt formula would under-correct by roughly sqrt(r* / r)
 * relative to what a single rescale actually needs for this construction of eta_sp
 * and eta_time. (A different estimator pair whose ratio genuinely scales as
 * tau^2 -- e.g. if eta_time itself were built without the extra sqrt(tau)
 * prefactor -- would need the sqrt form instead; that is not the case here.)
 *
 * Aiming at the (log) band centre r* = sqrt(gamma_time Gamma_time) makes a
 * single rescale land inside the band from any starting ratio, to leading
 * order in the asymptotic regime the Taylor expansion above assumes.
 *
 * Returns \p dt unchanged when there is nothing to balance (eta_time ~ 0 or
 * eta_sp ~ 0) or the estimates are not finite.
 */
template<class Scalar>
Scalar rescaledTimeStep(Scalar dt,
                        Scalar etaSp,
                        Scalar etaTime,
                        const BalancingTargets<Scalar>& t,
                        Scalar dtMin = Scalar{0},
                        Scalar dtMax = std::numeric_limits<Scalar>::max(),
                        Scalar maxGrow = Scalar{3},
                        Scalar maxShrink = Scalar{0.2})
{
    if (!(dt > Scalar{0}) || !(etaSp > Scalar{0}) || !(etaTime > Scalar{0})
        || !std::isfinite(etaTime))
        return dt;
    if (spaceTimeBalanced(etaSp, etaTime, t))
        return dt;
    const Scalar rStar = std::sqrt(std::max(t.gammaTime * t.GammaTime, Scalar{0}));
    Scalar factor = rStar * etaSp / etaTime;
    factor = std::clamp(factor, maxShrink, maxGrow);
    return std::clamp(dt * factor, dtMin, dtMax);
}

/*!
 * \brief Newton stopping, eq. (Criteria_newton):
 *        eta_lin <= Gamma_lin max{eta_sp, eta_time}   AND   MB <= tol_MB.
 *
 * The material-balance condition is non-negotiable: convergence is never
 * declared while MB exceeds the floor, whatever the estimator ratio.
 */
template<class Scalar>
bool newtonConverged(Scalar etaLin,
                     Scalar etaSp,
                     Scalar etaTime,
                     Scalar materialBalance,
                     const BalancingTargets<Scalar>& t)
{
    const bool mbOk = std::abs(materialBalance) <= t.tolMB;
    if (!mbOk)
        return false;
    const Scalar disc = std::max(etaSp, etaTime);
    // A non-positive or non-finite discretization estimate is *no usable
    // budget*, not "converged": a collapsed eta_sp (incomplete cell-face
    // geometry) or a NaN must never let the a posteriori criterion accept an
    // iterate. The caller then keeps OPM's native convergence decision.
    if (!(disc > Scalar{0}) || !std::isfinite(disc) || !std::isfinite(etaLin))
        return false;
    return etaLin <= t.GammaLin * disc;
}

/*!
 * \brief Linear-solver stopping, eq. (Criteria_alg):
 *        eta_alg <= Gamma_alg max(eta_sp, eta_time).
 *
 * Loose in the early Newton iterations (large eta_lin), tight near nonlinear
 * convergence.  Returns the relative reduction target to hand the Krylov
 * solver, clamped to [\p redMin, \p redMax]; std::nullopt-like semantics are
 * left to the caller (pass redMax when eta_lin is not yet available).
 */
template<class Scalar>
Scalar linearSolveTarget(Scalar etaAlgInitial,
                         Scalar etaSp,
                         Scalar etaTime,
                         const BalancingTargets<Scalar>& t,
                         Scalar redMin,
                         Scalar redMax)
{
    const Scalar disc = std::max(etaSp, etaTime);
    if (!(etaAlgInitial > Scalar{0}) || !(disc > Scalar{0})
        || !std::isfinite(etaAlgInitial) || !std::isfinite(disc))
        return redMax;
    const Scalar target = t.GammaAlg * disc / etaAlgInitial;
    return std::clamp(target, redMin, redMax);
}

template<class Scalar>
bool linearConverged(Scalar etaAlg,
                     Scalar etaSp,
                     Scalar etaTime,
                     const BalancingTargets<Scalar>& t)
{
    const Scalar disc = std::max(etaSp, etaTime);
    return (disc > Scalar{0}) && std::isfinite(disc) && std::isfinite(etaAlg)
        && etaAlg <= t.GammaAlg * disc;
}

/*!
 * \brief Sub-problem branching trigger (paper "Sub-problem branching"):
 *        eta_W (or eta_st) > Gamma_W max{eta_sp, eta_time}
 *        => iterate the well / phase-state sub-problem with the reservoir
 *           frozen, rather than cutting tau.
 */
template<class Scalar>
bool wellSubProblemDominates(Scalar etaW, Scalar etaSp, Scalar etaTime,
                             const BalancingTargets<Scalar>& t)
{
    const Scalar disc = std::max(etaSp, etaTime);
    return (disc > Scalar{0}) && (etaW > t.GammaW * disc);
}

template<class Scalar>
bool stateSubProblemDominates(Scalar etaSt, Scalar etaSp, Scalar etaTime,
                              const BalancingTargets<Scalar>& t)
{
    const Scalar disc = std::max(etaSp, etaTime);
    return (disc > Scalar{0}) && (etaSt > t.GammaW * disc);
}

} // namespace Opm::APosteriori

#endif // OPM_APOSTERIORI_BALANCING_CRITERIA_HPP
