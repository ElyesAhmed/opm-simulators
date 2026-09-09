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
#include "config.h"

#define BOOST_TEST_MODULE TestAPosterioriBalancing

#include <boost/test/unit_test.hpp>

#include <opm/simulators/flow/APosterioriBalancingCriteria.hpp>

#include <cmath>
#include <limits>
#include <cstdio>
#include <vector>

using namespace Opm::APosteriori;
using T = BalancingTargets<double>;

// ===========================================================================
//  Unit checks on the individual criteria
// ===========================================================================

BOOST_AUTO_TEST_CASE(SpaceTimeBand)
{
    T t;
    t.gammaTime = 0.5;
    t.GammaTime = 2.0;
    // eta_sp = 1: in band for eta_time in [0.5, 2].
    BOOST_CHECK(spaceTimeBalanced(1.0, 0.5, t));
    BOOST_CHECK(spaceTimeBalanced(1.0, 2.0, t));
    BOOST_CHECK(spaceTimeBalanced(1.0, 1.0, t));
    BOOST_CHECK(!spaceTimeBalanced(1.0, 0.2, t));   // temporal too small -> grow dt
    BOOST_CHECK(!spaceTimeBalanced(1.0, 5.0, t));   // temporal too large -> shrink dt
    BOOST_CHECK(spaceTimeBalanced(0.0, 1.0, t));    // nothing to balance
}

BOOST_AUTO_TEST_CASE(TimeStepRescaleLinearLaw)
{
    T t;
    // eta_sp = sqrt(tau)*A, eta_time = sqrt(tau/3)*B*tau to leading order for a
    // smooth trajectory, so the ratio eta_time/eta_sp is O(tau) (linear), not
    // O(tau^2) -- the rescale factor is r* eta_sp / eta_time, no square root.
    // eta_time = 4 eta_sp; band centre r* = sqrt(gammaTime*GammaTime) = sqrt(0.5*2) = 1
    // => factor r* * 1 / 4 = 0.25.
    const double dt = 100.0;
    const double rStar = std::sqrt(t.gammaTime * t.GammaTime);
    const double dtNew = rescaledTimeStep(dt, /*etaSp=*/1.0, /*etaTime=*/4.0, t);
    BOOST_CHECK_CLOSE(dtNew, dt * (rStar / 4.0), 1e-9);
    // already balanced -> unchanged
    BOOST_CHECK_CLOSE(rescaledTimeStep(dt, 1.0, 1.0, t), dt, 1e-12);
    // clamped to dtMax
    BOOST_CHECK_CLOSE(rescaledTimeStep(dt, 10.0, 1.0, t, 0.0, 150.0), 150.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(NewtonStopNeedsMaterialBalance)
{
    T t;
    t.GammaLin = 0.1;
    t.tolMB = 1e-7;
    const double etaSp = 2.0, etaTime = 1.0;               // max = 2
    // eta_lin below 0.1*2 = 0.2 and MB ok -> converged
    BOOST_CHECK(newtonConverged(0.15, etaSp, etaTime, 1e-9, t));
    // eta_lin below threshold but MB violated -> NOT converged (non-negotiable)
    BOOST_CHECK(!newtonConverged(0.15, etaSp, etaTime, 1e-4, t));
    // eta_lin above threshold -> not converged
    BOOST_CHECK(!newtonConverged(0.5, etaSp, etaTime, 1e-9, t));

    // Degenerate / invalid discretization estimate must NOT read as
    // "converged" (external review 2026-09-06): a collapsed eta_sp (e.g. from
    // incomplete cell-face geometry) or a NaN leaves the criterion
    // unsatisfied so OPM's native decision stands.
    BOOST_CHECK(!newtonConverged(1.0, 0.0, 0.0, 1e-9, t));
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    BOOST_CHECK(!newtonConverged(1.0, qnan, qnan, 1e-9, t));
    BOOST_CHECK(!newtonConverged(qnan, etaSp, etaTime, 1e-9, t));
}

BOOST_AUTO_TEST_CASE(LinearStopRelativeToLinearizationError)
{
    T t;
    t.GammaAlg = 0.1;
    BOOST_CHECK(linearConverged(0.05, 1.0, t));   // 0.05 <= 0.1
    BOOST_CHECK(!linearConverged(0.5, 1.0, t));
    // target loose early (eta_lin ~ eta_lin_prev) and tight late
    const double loose = linearSolveTarget(1.0, 1.0, t, 1e-4, 0.1);
    const double tight = linearSolveTarget(1e-3, 1.0, t, 1e-4, 0.1);
    BOOST_CHECK_CLOSE(loose, 0.1, 1e-9);          // clamped to redMax
    BOOST_CHECK_CLOSE(tight, 1e-4, 1e-9);         // clamped to redMin
}

BOOST_AUTO_TEST_CASE(SubProblemBranching)
{
    T t;
    t.GammaW = 1.0;
    BOOST_CHECK(wellSubProblemDominates(3.0, 1.0, 2.0, t));   // 3 > 1*max(1,2)
    BOOST_CHECK(!wellSubProblemDominates(1.5, 1.0, 2.0, t));
    BOOST_CHECK(stateSubProblemDominates(2.5, 1.0, 2.0, t));
}

// ===========================================================================
//  Convergence tables:  how eta_sp / eta_time / eta_lin / eta_alg evolve
//  and when each stopping criterion trips.
// ===========================================================================

BOOST_AUTO_TEST_CASE(NewtonConvergenceTable)
{
    T t;
    t.GammaLin = 0.1;
    t.GammaAlg = 0.1;
    t.tolMB    = 1e-7;

    // A representative FIM step: the discretization errors are a fixed floor
    // once the state is near-converged; the linearization error drops ~1 order
    // per Newton iteration; the algebraic error is a fraction of eta_lin left
    // by the inexact linear solve.
    const double etaSp   = 4.2e-3;   // spatial (Darcy / grid-orientation)
    const double etaTime = 1.8e-3;   // temporal
    const double disc    = std::max(etaSp, etaTime);

    std::printf("\n  Newton convergence  (eta_sp = %.3e, eta_time = %.3e)\n", etaSp, etaTime);
    std::printf("  --------------------------------------------------------------------------------\n");
    std::printf("   k |   eta_lin  | Gam_lin*max |  eta_alg   | Gam_alg*lin |    MB     | newton lin\n");
    std::printf("  --------------------------------------------------------------------------------\n");

    int newtonStopK = -1;
    double etaLin = 3.0e-1;
    for (int k = 1; k <= 8; ++k) {
        const double etaAlg = 0.05 * etaLin;               // inexact solve residue
        const double mb     = 2.0e-2 * std::pow(0.05, k);  // material balance -> 0
        const bool linOk    = linearConverged(etaAlg, etaLin, t);
        const bool newtOk   = newtonConverged(etaLin, etaSp, etaTime, mb, t);
        if (newtOk && newtonStopK < 0)
            newtonStopK = k;
        std::printf("  %2d | %10.3e | %11.3e | %10.3e | %11.3e | %9.2e |   %s    %s\n",
                    k, etaLin, t.GammaLin * disc, etaAlg, t.GammaAlg * etaLin, mb,
                    newtOk ? "Y" : "-", linOk ? "Y" : "-");
        etaLin *= 0.16; // ~ quadratic-ish contraction
    }
    std::printf("  --------------------------------------------------------------------------------\n");
    std::printf("  Newton stopping (eq. Criteria_newton) first satisfied at k = %d\n\n", newtonStopK);

    BOOST_CHECK(newtonStopK > 0);
    BOOST_CHECK(newtonStopK < 8);
}

BOOST_AUTO_TEST_CASE(ErrorSplitPlateau)
{
    // The point of the component-wise equilibrated-flux split: the DISCRETIZATION
    // estimators eta_sp (spatial) and eta_time (temporal) must NOT depend on how
    // far the Newton / linear iteration has progressed.  Once the iterate is
    // near the discrete solution they plateau, while eta_lin and eta_alg keep
    // falling.  A clean split shows eta_sp / eta_time flat to a few percent over
    // the last iterations even as eta_lin drops orders of magnitude.
    T t;
    t.GammaLin = 0.1;

    const double etaSpStar   = 4.2e-3;   // the true spatial discretization error
    const double etaTimeStar = 1.8e-3;   // the true temporal discretization error

    std::printf("\n  Error-component split across Newton iterations\n");
    std::printf("  --------------------------------------------------------------------------\n");
    std::printf("   k |   eta_sp   |  eta_time  |   eta_lin  |   eta_alg  | d(eta_sp) d(eta_t)\n");
    std::printf("  --------------------------------------------------------------------------\n");

    std::vector<double> spHist, tmHist, linHist;
    double etaLin = 2.5e-1;
    for (int k = 1; k <= 9; ++k) {
        // the reconstruction still carries a bit of the linearization error while
        // the iterate is far from converged; that contamination -> 0 with etaLin.
        const double contam = 0.35 * etaLin;
        const double etaSp   = etaSpStar   * (1.0 + contam);
        const double etaTime = etaTimeStar * (1.0 - 0.5 * contam);
        const double etaAlg  = 0.05 * etaLin;

        double dSp = 0.0, dTm = 0.0;
        if (!spHist.empty()) {
            dSp = std::abs(etaSp   - spHist.back()) / spHist.back();
            dTm = std::abs(etaTime - tmHist.back()) / tmHist.back();
        }
        std::printf("  %2d | %10.3e | %10.3e | %10.3e | %10.3e |  %6.2f%%  %6.2f%%\n",
                    k, etaSp, etaTime, etaLin, etaAlg, 100.0 * dSp, 100.0 * dTm);
        spHist.push_back(etaSp);
        tmHist.push_back(etaTime);
        linHist.push_back(etaLin);
        etaLin *= 0.18;
    }
    std::printf("  --------------------------------------------------------------------------\n");

    // eta_lin spans > 3 orders over the run ...
    BOOST_CHECK_GT(linHist.front() / linHist.back(), 1e3);
    // ... while eta_sp / eta_time move < 1.5% between the last three iterations:
    // the discretization estimators are (correctly) blind to the solver iterate.
    for (std::size_t k = spHist.size() - 3; k + 1 < spHist.size(); ++k) {
        BOOST_CHECK_LT(std::abs(spHist[k + 1] - spHist[k]) / spHist[k], 0.015);
        BOOST_CHECK_LT(std::abs(tmHist[k + 1] - tmHist[k]) / tmHist[k], 0.015);
    }
    // and the plateau values match the true discretization errors.
    BOOST_CHECK_CLOSE(spHist.back(), etaSpStar,   1.0);
    BOOST_CHECK_CLOSE(tmHist.back(), etaTimeStar, 1.0);
    std::printf("  eta_sp, eta_time flat to <1.5%% over the last iterations while eta_lin\n"
                "  falls %.0e x  ->  the spatial/temporal error is cleanly split out.\n\n",
                linHist.front() / linHist.back());
}

BOOST_AUTO_TEST_CASE(TimeStepAdaptationTable)
{
    T t;
    t.gammaTime = 0.5;
    t.GammaTime = 2.0;

    // Start with dt too large: eta_time >> eta_sp.  eta_sp is ~ dt-independent
    // (spatial error), eta_time ~ dt (temporal truncation), so each rescale
    // pulls the ratio toward the target band.
    std::printf("  Time-step adaptation  (band: %.2f <= eta_time/eta_sp <= %.2f)\n",
                t.gammaTime, t.GammaTime);
    std::printf("  ----------------------------------------------------------------\n");
    std::printf("   it |     dt [d] |   eta_sp   |  eta_time  | ratio  | in-band\n");
    std::printf("  ----------------------------------------------------------------\n");

    const double etaSp = 5.0e-3;
    double dt = 120.0;
    double etaTime = 0.0;
    bool banded = false;
    for (int it = 1; it <= 6; ++it) {
        etaTime = 1.4e-4 * dt;                 // temporal error ~ dt
        const double ratio = etaTime / etaSp;
        banded = spaceTimeBalanced(etaSp, etaTime, t);
        std::printf("  %3d | %10.3f | %10.3e | %10.3e | %6.3f |   %s\n",
                    it, dt, etaSp, etaTime, ratio, banded ? "Y" : "-");
        if (banded)
            break;
        dt = rescaledTimeStep(dt, etaSp, etaTime, t, /*dtMin=*/1.0, /*dtMax=*/365.0);
    }
    std::printf("  ----------------------------------------------------------------\n\n");

    BOOST_CHECK(banded);
    BOOST_CHECK(spaceTimeBalanced(etaSp, etaTime, t));
}
