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

#define BOOST_TEST_MODULE TestAdaptiveLinearSolveReduction

#include <boost/test/unit_test.hpp>

#include <opm/simulators/flow/AdaptiveLinearSolveReduction.hpp>

#include <optional>
#include <vector>

using Opm::AdaptiveLinearSolveReduction;

namespace {

constexpr double gamma_ = 0.9;
constexpr double rmin   = 1e-3;   // "--linear-solver-reduction"
constexpr double rmax   = 1e-1;   // loosest permitted

AdaptiveLinearSolveReduction<double> makeEnabled()
{
    AdaptiveLinearSolveReduction<double> a(true, gamma_, rmin, rmax);
    a.reset();
    return a;
}

} // namespace

BOOST_AUTO_TEST_CASE(DisabledReturnsNullopt)
{
    AdaptiveLinearSolveReduction<double> a(false, gamma_, rmin, rmax);
    a.reset();
    BOOST_CHECK(!a.enabled());
    std::vector<std::vector<double>> hist = {{1.0}, {0.1}};
    BOOST_CHECK(a.forcingTerm(hist) == std::nullopt);
}

BOOST_AUTO_TEST_CASE(FirstSolveUsesLoosestTolerance)
{
    auto a = makeEnabled();
    BOOST_CHECK(a.enabled());

    // No history at all.
    auto t0 = a.forcingTerm({});
    BOOST_REQUIRE(t0.has_value());
    BOOST_CHECK_CLOSE(*t0, rmax, 1e-10);

    // Exactly one entry (current iterate only, no previous).
    auto t1 = a.forcingTerm({{1.0}});
    BOOST_REQUIRE(t1.has_value());
    BOOST_CHECK_CLOSE(*t1, rmax, 1e-10);
}

BOOST_AUTO_TEST_CASE(EisenstatWalkerChoice2)
{
    auto a = makeEnabled();

    // Prime with a first solve so prev_target_ == rmax.
    a.forcingTerm({{1.0}});

    // Residual dropped by a factor 10 => ratio 0.1 => gamma*0.1^2 = 9e-3,
    // clamped to [rmin, rmax] leaves 9e-3.
    auto t = a.forcingTerm({{1.0}, {0.1}});
    BOOST_REQUIRE(t.has_value());
    BOOST_CHECK_CLOSE(*t, gamma_ * 0.1 * 0.1, 1e-9);
    BOOST_CHECK_GE(*t, rmin);
    BOOST_CHECK_LE(*t, rmax);
}

BOOST_AUTO_TEST_CASE(ClampToLoosestWhenNoProgress)
{
    auto a = makeEnabled();
    a.forcingTerm({{1.0}});
    // Residual did not shrink => ratio 1 => gamma*1 = 0.9, clamp to rmax.
    auto t = a.forcingTerm({{1.0}, {1.0}});
    BOOST_REQUIRE(t.has_value());
    BOOST_CHECK_CLOSE(*t, rmax, 1e-10);
}

BOOST_AUTO_TEST_CASE(ClampToTightestNearConvergence)
{
    auto a = makeEnabled();
    a.forcingTerm({{1.0}});
    a.forcingTerm({{1.0}, {1e-2}});
    // Very fast drop: ratio 1e-4 => gamma*1e-8, far below rmin, clamp to rmin.
    auto t = a.forcingTerm({{1.0}, {1e-2}, {1e-6}});
    BOOST_REQUIRE(t.has_value());
    BOOST_CHECK_CLOSE(*t, rmin, 1e-10);
}

BOOST_AUTO_TEST_CASE(MaxNormOverComponents)
{
    auto a = makeEnabled();
    a.forcingTerm({{0.4, 1.0, 0.2}});                 // max-norm 1.0
    auto t = a.forcingTerm({{0.4, 1.0, 0.2},
                            {0.05, 0.10, 0.03}});     // max-norm 0.10 => ratio 0.10
    BOOST_REQUIRE(t.has_value());
    BOOST_CHECK_CLOSE(*t, gamma_ * 0.1 * 0.1, 1e-9);
}

BOOST_AUTO_TEST_CASE(NeverTighterThanConfigured)
{
    // The tightest the adaptive tolerance may request equals rmin, i.e. the
    // statically configured --linear-solver-reduction: the linear solve is
    // never made *more* accurate than the user asked for.
    auto a = makeEnabled();
    for (int k = 0; k < 20; ++k) {
        std::vector<std::vector<double>> hist;
        double r = 1.0;
        for (int i = 0; i <= k; ++i) { hist.push_back({r}); r *= 1e-3; }
        auto t = a.forcingTerm(hist);
        BOOST_REQUIRE(t.has_value());
        BOOST_CHECK_GE(*t, rmin);
        BOOST_CHECK_LE(*t, rmax);
    }
}

BOOST_AUTO_TEST_CASE(ResetRestoresInitialState)
{
    auto a = makeEnabled();
    a.forcingTerm({{1.0}, {1e-6}});   // drive prev_target_ towards rmin
    a.reset();
    auto t = a.forcingTerm({{1.0}});  // first solve again
    BOOST_REQUIRE(t.has_value());
    BOOST_CHECK_CLOSE(*t, rmax, 1e-10);
}

BOOST_AUTO_TEST_CASE(MinIterationsGuardDisablesOnCheapSolves)
{
    // min_iterations = 12: engage only if the previous linear solve took >= 12
    // iterations.
    AdaptiveLinearSolveReduction<double> a(true, gamma_, rmin, rmax, /*min_iterations=*/12);
    a.reset();

    // reset() assumes "expensive" so the first post-reset solve is adaptive.
    BOOST_CHECK(a.forcingTerm({{1.0}}).has_value());

    // Cheap previous solve => static tolerance (nullopt), regardless of history.
    a.recordLinearIterations(3);
    BOOST_CHECK(a.forcingTerm({{1.0}, {0.1}}) == std::nullopt);

    // Expensive previous solve => adaptive again.
    a.recordLinearIterations(40);
    BOOST_CHECK(a.forcingTerm({{1.0}, {0.1}, {0.05}}).has_value());
}
