// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
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
 * \brief Unit tests for the coupled well/NNC source Taylor remainder
 *        (Opm::APosteriori::sourceLinearizationDefect) that completes
 *        eta_lin's reduced-balance term in
 *        APosterioriSpatialTemporalEstimator::completeSourceLinearizationDefect().
 *
 * These are deliberately free-function/synthetic tests -- no Simulator, no
 * grid -- following the same pattern as test_aposteriori_reconstruction.cpp.
 * They validate the balance identity
 *
 *   Delta Q_K = Delta A_K/tau - sum_geom(F_lin - F_new) - (R_new - R_lin)
 *
 * itself and its Taylor-remainder scaling, independent of the surrounding
 * simulator machinery (which test_aposteriori_estimator_compile.cpp already
 * covers for compilation, and full Flow regression runs cover for the
 * lifecycle end to end).
 */
#include "config.h"

#define BOOST_TEST_MODULE TestAPosterioriSourceDefect

#include <boost/test/unit_test.hpp>

#include <opm/simulators/flow/APosterioriReconstruction.hpp>

#include <cmath>
#include <vector>

using namespace Opm::APosteriori;

// ===========================================================================
//  1. Balance identity
// ===========================================================================

BOOST_AUTO_TEST_CASE(BalanceIdentityMatchesDefinition)
{
    // Arbitrary, otherwise-unrelated numbers: the defect must equal exactly
    // linAccumRateDef - linGeomFluxDefect - (residualNew - residualLin), no
    // more, no less -- this is the whole point of the reduced-balance
    // construction (eq. in completeSourceLinearizationDefect's doc comment).
    const double linAccum = 3.7;
    const double linFlux  = -1.2;
    const double rNew     = 0.85;
    const double rLin     = 0.10;

    const double got = sourceLinearizationDefect(linAccum, linFlux, rNew, rLin);
    const double expected = linAccum - linFlux - (rNew - rLin);
    BOOST_CHECK_CLOSE(got, expected, 1e-12);
    BOOST_CHECK_CLOSE(got, 3.7 - (-1.2) - (0.85 - 0.10), 1e-12);
}

BOOST_AUTO_TEST_CASE(ZeroWhenResidualExactlyPredicted)
{
    // Reduced-residual consistency: if the well-eliminated residual assembled
    // at chi^k (R_new) differs from the stored linear prediction R_lin by
    // EXACTLY the already-captured accumulation/flux mismatch, the source
    // Taylor remainder must vanish identically -- no double counting and no
    // spurious leftover from an inconsistent residual footing.
    const double linAccum = 2.5;
    const double linFlux  = 0.9;
    const double rLin     = -4.0;
    const double rNew     = rLin + (linAccum - linFlux); // residual grew by exactly NA - flux

    BOOST_CHECK_SMALL(sourceLinearizationDefect(linAccum, linFlux, rNew, rLin), 1e-12);

    // Degenerate all-zero case.
    BOOST_CHECK_SMALL(sourceLinearizationDefect(0.0, 0.0, 0.0, 0.0), 1e-12);
    // Equal nonzero residuals, zero accumulation/flux mismatch.
    BOOST_CHECK_SMALL(sourceLinearizationDefect(0.0, 0.0, 7.3, 7.3), 1e-12);
}

BOOST_AUTO_TEST_CASE(SignConventionResidualOvershootIsNegative)
{
    // With no accumulation/flux mismatch, a residual that came in HIGHER than
    // the linear-solve prediction (e.g. a well ramped harder than the
    // linearization assumed) must show up as a NEGATIVE source defect, and a
    // residual that came in lower as positive -- matching
    // Delta Q_K = ... - (R_new - R_lin) losing magnitude precisely when
    // R_new > R_lin.
    BOOST_CHECK_LT(sourceLinearizationDefect(0.0, 0.0, /*rNew=*/5.0, /*rLin=*/2.0), 0.0);
    BOOST_CHECK_GT(sourceLinearizationDefect(0.0, 0.0, /*rNew=*/1.0, /*rLin=*/2.0), 0.0);
}

// ===========================================================================
//  2. Quadratic decay of the Taylor remainder
// ===========================================================================

namespace {
// A smooth, non-trivial synthetic nonlinear "source" (stand-in for a well/NNC
// term's dependence on the primary variable): Q(x) = x^3 - 2x^2 + 5.
// Q'(x)  = 3x^2 - 4x
// Q''(x) = 6x - 4  (nonzero away from x = 2/3, so pick x0 elsewhere)
double Q(double x)  { return x * x * x - 2.0 * x * x + 5.0; }
double dQ(double x) { return 3.0 * x * x - 4.0 * x; }
double d2Q(double x) { return 6.0 * x - 4.0; }
}

BOOST_AUTO_TEST_CASE(QuadraticDecayOfTaylorRemainder)
{
    // Isolate the pure source-nonlinearity remainder: zero accumulation/flux
    // mismatch, R_lin the FIRST-ORDER (Newton-linearized) prediction of the
    // updated residual, R_new the actual nonlinear value at chi^{k-1}+dx.
    // Then sourceLinearizationDefect = -(Q(x0+dx) - Q(x0) - Q'(x0) dx), the
    // exact Taylor remainder -- which for a C^2 function is
    // -0.5 Q''(x0) dx^2 + O(dx^3).
    const double x0 = 1.3; // away from the inflection point 2/3
    const double Q0 = Q(x0);
    const double dQ0 = dQ(x0);

    constexpr int kNumSteps = 8;
    std::vector<double> defects;
    double dx = 0.2;
    for (int n = 0; n < kNumSteps; ++n, dx *= 0.5) {
        const double rLin = Q0 + dQ0 * dx;   // linear-solve-predicted new residual
        const double rNew = Q(x0 + dx);      // actual nonlinear residual at chi^k
        const double defect = sourceLinearizationDefect(0.0, 0.0, rNew, rLin);
        defects.push_back(defect);
    }

    // (a) the remainder shrinks monotonically as dx shrinks.
    for (std::size_t n = 0; n + 1 < defects.size(); ++n) {
        BOOST_CHECK_LT(std::abs(defects[n + 1]), std::abs(defects[n]));
    }

    // (b) quadratic rate: halving dx quarters the remainder, i.e. the ratio
    // of successive |defect| values converges to 4. Since Q is cubic, the
    // exact remainder is 0.5 Q''(x0) dx^2 + Q'''(x0)/6 dx^3, so the ratio's
    // deviation from 4 is O(dx) (~dx/d2Q(x0) here) -- skip the larger early
    // steps, where that O(dx) correction is still a percent or more, and
    // check the asymptotic (small-dx) ratios only.
    for (std::size_t n = 4; n + 1 < defects.size(); ++n) {
        const double ratio = std::abs(defects[n]) / std::abs(defects[n + 1]);
        BOOST_CHECK_CLOSE(ratio, 4.0, 1.0); // within 1% of exact quadratic decay
    }

    // (c) the finest-step remainder matches the analytic leading-order Taylor
    // coefficient -0.5 Q''(x0) dx^2.
    const double dxFinest = 0.2 * std::pow(0.5, kNumSteps - 1);
    const double analytic = -0.5 * d2Q(x0) * dxFinest * dxFinest;
    BOOST_CHECK_CLOSE(defects.back(), analytic, 1.0); // within 1%
}

// ===========================================================================
//  3. Composition with the shared weighted indicator (equilibrationIndicator)
// ===========================================================================

BOOST_AUTO_TEST_CASE(WeightedIndicatorMatchesManualPrefactorFormula)
{
    // completeSourceLinearizationDefect() weights the source defect with the
    // SAME local indicator used for the balance-residual term eta_eq (see
    // equilibrationIndicator's doc comment): the two are algebraically
    // identical, applied to different residual-like quantities. Confirm the
    // composition sourceLinearizationDefect -> equilibrationIndicator
    // reproduces the hand-written prefactor formula
    //   pref = sqrt(tau) eps^{-1/2} c_KK^{-1/2} h_K D_K^{l/2} / sqrt(V_K)
    //   eta = pref * |Delta Q_K|
    // exactly.
    const double linAccum = -0.6, linFlux = 1.4, rNew = 0.05, rLin = -0.32;
    const double dt = 4.0, epsilon = 0.5, cKK = 9.0, hK = 3.0, weightPow = 2.0, volume = 16.0;

    const double defect = sourceLinearizationDefect(linAccum, linFlux, rNew, rLin);
    const double eta = equilibrationIndicator(defect, dt, epsilon, cKK, hK, weightPow, volume);

    const double pref = std::sqrt(dt) / std::sqrt(epsilon) / std::sqrt(cKK)
        * hK * weightPow / std::sqrt(volume);
    const double expected = pref * std::abs(defect);
    BOOST_CHECK_CLOSE(eta, expected, 1e-9);

    // dt = 0 -> no contribution regardless of the defect magnitude (tau^n
    // prefactor collapses eta_lin's source term at a zero-length step).
    BOOST_CHECK_SMALL(equilibrationIndicator(defect, 0.0, epsilon, cKK, hK, weightPow, volume), 1e-12);
}

BOOST_AUTO_TEST_CASE(SourceDecompositionIndependentOfAccumulationAndFlux)
{
    // "Export separately" check: the source term must depend ONLY on the
    // residual pair (R_new, R_lin) once the accumulation/flux remainders are
    // fixed, and vice versa -- the three contributions to eta_lin (NA, flux,
    // source) are structurally independent inputs, not derived from one
    // another. Varying one leaves the others' effect on the identity linear
    // and separable.
    const double rNew = 2.0, rLin = 0.5; // fixed residual pair -> fixed -(rNew-rLin) = -1.5
    const double residualPart = -(rNew - rLin);

    for (double linAccum : {-3.0, 0.0, 1.25, 10.0}) {
        for (double linFlux : {-2.0, 0.0, 0.75}) {
            const double got = sourceLinearizationDefect(linAccum, linFlux, rNew, rLin);
            BOOST_CHECK_CLOSE(got, linAccum - linFlux + residualPart, 1e-12);
        }
    }
}
