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
 * \brief Eisenstat--Walker style adaptive forcing term for the Newton linear solve.
 */
#ifndef OPM_ADAPTIVE_LINEAR_SOLVE_REDUCTION_HPP
#define OPM_ADAPTIVE_LINEAR_SOLVE_REDUCTION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace Opm {

/*!
 * \brief Adaptive (inexact-Newton) relative tolerance for the linear solve.
 *
 * A fully implicit solve does not need the linear system solved more accurately
 * than the current nonlinear (linearization) error: solving it tighter only
 * spends CPR/AMG iterations that do not move the Newton iterate.  This class
 * implements the classic Eisenstat--Walker "choice 2" forcing term
 *
 *     eta_k = gamma * ( ||F_k|| / ||F_{k-1}|| )^2 ,
 *
 * clamped to [reduction_min, reduction_max] and safeguarded against an
 * over-rapid decrease, where ||F_k|| is the nonlinear residual measure at the
 * current Newton iterate.  This is exactly the algebraic-vs-linearization
 * balancing criterion of the a posteriori framework
 * (Ahmed et al., criterion for stopping the algebraic solver): the algebraic
 * error is kept a fixed factor below the current nonlinear error.
 *
 * The \e converged nonlinear solution is unchanged; only the number of linear
 * iterations differs.  Disabled by default.
 */
template<class Scalar>
class AdaptiveLinearSolveReduction
{
public:
    AdaptiveLinearSolveReduction() = default;

    /*!
     * \param enabled        Whether the adaptive tolerance is used at all.
     * \param gamma          Safety factor in (0, 1].
     * \param reduction_min  Tightest relative reduction (typically the value of
     *                       --linear-solver-reduction); the adaptive tolerance
     *                       never asks for more accuracy than this.
     * \param reduction_max  Loosest relative reduction permitted.
     */
    AdaptiveLinearSolveReduction(bool enabled,
                                 Scalar gamma,
                                 Scalar reduction_min,
                                 Scalar reduction_max,
                                 int min_iterations = 0)
        : enabled_(enabled)
        , gamma_(gamma)
        , reduction_min_(reduction_min)
        , reduction_max_(std::max(reduction_max, reduction_min))
        , min_iterations_(min_iterations)
        , prev_target_(std::max(reduction_max, reduction_min))
    {}

    //! \brief Call once at the start of every time step (Newton restart).
    void reset()
    {
        prev_target_ = reduction_max_;
        prev_iterations_ = min_iterations_; // assume "expensive" until proven cheap
    }

    /*!
     * \brief Record the iteration count of the linear solve just performed.
     *
     * Used to gate the adaptive tolerance: if the linear solver reaches its
     * tolerance in very few iterations, the linear solve is not the bottleneck
     * and loosening it only degrades the Newton update, so we keep the
     * statically configured tolerance.
     */
    void recordLinearIterations(int iterations)
    { prev_iterations_ = iterations; }

    //! \brief Whether the feature is active.
    bool enabled() const
    { return enabled_; }

    /*!
     * \brief Relative reduction target for the linear solve about to be done.
     *
     * \param history  The Newton residual-norm history: \c history[i] holds the
     *                 per-component convergence measures (CNV) at Newton iterate
     *                 \c i, and \c history.back() the current iterate.
     * \return \c std::nullopt if the feature is disabled (use the statically
     *         configured tolerance), otherwise the relative reduction target.
     */
    std::optional<Scalar>
    forcingTerm(const std::vector<std::vector<Scalar>>& history)
    {
        if (!enabled_) {
            return std::nullopt;
        }

        // Guard: if the previous linear solve reached its tolerance in very few
        // iterations, the CPR/AMG preconditioner is already overshooting and
        // loosening the tolerance buys no time while it degrades the Newton
        // update. Keep the configured static tolerance in that case.
        if (prev_iterations_ < min_iterations_) {
            return std::nullopt;
        }

        const std::size_t n = history.size();
        Scalar target;
        if (n < 2 || norm(history[n - 2]) <= Scalar{0}) {
            // First solve of this step: no nonlinear-progress information yet.
            target = reduction_max_;
        }
        else {
            const Scalar ratio = norm(history[n - 1]) / norm(history[n - 2]);
            target = gamma_ * ratio * ratio; // Eisenstat--Walker choice 2 (exponent 2)

            // Safeguard: do not let the forcing term fall faster than the
            // nonlinear convergence actually warrants.
            const Scalar safeguarded = gamma_ * prev_target_ * prev_target_;
            if (safeguarded > Scalar{0.1}) {
                target = std::max(target, safeguarded);
            }
        }

        target = std::clamp(target, reduction_min_, reduction_max_);
        prev_target_ = target;
        return target;
    }

    /*!
     * \brief Forcing term for the paper's Criteria_alg: stop the linear solve
     *        once
     *
     *        eta_alg <= Gamma_alg * max(eta_sp, eta_time) .
     *
     *        eta_alg is a norm of the linear residual, so it scales linearly
     *        with the relative residual reduction r: eta_alg(r) ~ r *
     *        eta_alg^{(0)}, where eta_alg^{(0)} is the estimator of the
     *        un-reduced residual b. Hence the reduction target is
     *
     *        target = Gamma_alg * max(eta_sp, eta_time) / eta_alg^{(0)} .
     *
     *        eta_alg^{(0)} is the correct scale (commensurate with
     *        max(eta_sp,eta_time)), unlike eta_lin which is much smaller near
     *        Newton convergence.
     *
     *        A stricter solve also satisfies Criteria_alg.  The request is
     *        therefore bounded by the same cost and Newton-stability guards as
     *        the Eisenstat--Walker controller: keep the static tolerance after
     *        an already-cheap solve, never ask for more accuracy than
     *        --linear-solver-reduction, and never loosen beyond the configured
     *        adaptive reduction maximum.
     *
     * \param etaAlg0     eta_alg of the un-reduced residual b (rAlg = b).
     * \param maxSpTime   max(eta_sp, eta_time) at the most recent iterate.
     * \param gammaAlg    Gamma_alg in (0,1].
     * \return \c std::nullopt only when disabled; else the relative reduction
     *         target.
     */
    std::optional<Scalar>
    forcingTermFromEstimator(Scalar etaAlg0, Scalar maxSpTime, Scalar gammaAlg)
    {
        if (!enabled_)
            return std::nullopt;

        // A stricter solve still satisfies Criteria_alg.  If the previous
        // solve was already cheap, loosening its tolerance cannot save useful
        // work but can substantially degrade the Newton correction.
        if (prev_iterations_ < min_iterations_)
            return std::nullopt;

        // Without a valid discretization-error budget there is no estimator
        // basis for relaxing the user's configured linear tolerance.
        if (!(etaAlg0 > Scalar{0}) || !std::isfinite(maxSpTime) || maxSpTime <= Scalar{0})
            return std::nullopt;

        Scalar target = gammaAlg * (maxSpTime / etaAlg0);
        target = std::clamp(target, reduction_min_, reduction_max_);
        prev_target_ = target;
        return target;
    }

    //! The relative reduction most recently returned by a forcingTerm* call
    //! (starting point for an enforcement re-solve).
    Scalar lastTarget() const { return prev_target_; }
    Scalar reductionMin() const { return reduction_min_; }

    //! \brief Max-norm of a per-component residual measure vector.
    static Scalar norm(const std::vector<Scalar>& v)
    {
        Scalar m{0};
        for (const Scalar x : v) {
            m = std::max(m, std::abs(x));
        }
        return m;
    }

private:
    bool   enabled_        {false};
    Scalar gamma_          {Scalar{0.9}};
    Scalar reduction_min_  {Scalar{1e-2}};
    Scalar reduction_max_  {Scalar{0.1}};
    int    min_iterations_ {0};
    Scalar prev_target_    {Scalar{0.1}};
    int    prev_iterations_{0};
};

} // namespace Opm

#endif // OPM_ADAPTIVE_LINEAR_SOLVE_REDUCTION_HPP
