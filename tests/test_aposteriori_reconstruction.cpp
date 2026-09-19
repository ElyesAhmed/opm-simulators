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

#define BOOST_TEST_MODULE TestAPosterioriReconstruction

#include <boost/test/unit_test.hpp>

#include <opm/simulators/flow/APosterioriReconstruction.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

using namespace Opm::APosteriori;
using V3 = Dune::FieldVector<double, 3>;

namespace {
constexpr double tol = 1e-10;
}

BOOST_AUTO_TEST_CASE(EquilibrationIndicatorHasExpectedScaling)
{
    const double eta = equilibrationIndicator(
        /*integratedResidual=*/12.0,
        /*dt=*/4.0,
        /*epsilon=*/0.25,
        /*cKK=*/9.0,
        /*hK=*/3.0,
        /*dLambdaPow=*/5.0,
        /*volume=*/16.0);
    BOOST_CHECK_CLOSE(eta, 60.0, 1e-12);
    BOOST_CHECK_SMALL(equilibrationIndicator(
        0.0, 4.0, 0.25, 9.0, 3.0, 5.0, 16.0), tol);
}

BOOST_AUTO_TEST_CASE(NeumannMeanPolicySeparatesCompatibilityDefect)
{
    constexpr std::array<double, 2> darcy{3.0, 0.0};
    constexpr std::array<double, 2> orthogonalEq{0.0, 4.0};
    constexpr std::array<double, 2> alignedEq{4.0, 0.0};
    BOOST_CHECK_EQUAL(
        componentwiseCombinedIndicator(darcy, orthogonalEq, false), 5.0);
    BOOST_CHECK_EQUAL(
        componentwiseCombinedIndicator(darcy, alignedEq, false), 7.0);
    BOOST_CHECK_EQUAL(
        componentwiseCombinedIndicator(darcy, orthogonalEq, true), 3.0);
}

// ---------------------------------------------------------------------------
//  Least-squares gradient
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(LSGradientExactForAffineField)
{
    // p(x) = p0 + a . (x - x_cell), recover a exactly from any >= 3 offsets.
    const V3 a{2.0, -3.5, 0.75};
    const double p0 = 10.0;

    std::vector<V3> d = {
        { 1.0,  0.0,  0.0}, {-1.0,  0.0,  0.0},
        { 0.0,  2.0,  0.0}, { 0.0, -0.5,  0.0},
        { 0.0,  0.0,  1.5}, { 0.3,  0.4, -0.9},
    };
    std::vector<double> uNb;
    for (const auto& dc : d) {
        uNb.push_back(p0 + (a * dc));
    }

    auto g = leastSquaresGradient<double, 3>(p0, uNb, d);
    for (int i = 0; i < 3; ++i) {
        BOOST_CHECK_CLOSE(g[i], a[i], 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(LSGradientRankDeficientIsFinite)
{
    // Only two offsets, both along x: the y/z gradient is undetermined but the
    // routine must return a finite, non-NaN vector (regularised).
    std::vector<V3> d = {{1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}};
    std::vector<double> uNb = {5.0, 3.0};
    auto g = leastSquaresGradient<double, 3>(4.0, uNb, d);
    for (int i = 0; i < 3; ++i) {
        BOOST_CHECK(std::isfinite(g[i]));
    }
    BOOST_CHECK_CLOSE(g[0], 1.0, 1e-6); // (5-3)/2
}

BOOST_AUTO_TEST_CASE(LSGradientWeightedStillExactForAffine)
{
    // Transmissibility-weighted fit (the a posteriori framework's choice): the
    // weights must not bias the recovery of an affine field, and non-positive
    // weights must be ignored.
    const V3 a{1.5, -2.0, 0.25};
    const double p0 = 7.0;
    std::vector<V3> d = {
        { 2.0,  0.0,  0.0}, {-1.0,  0.0,  0.0},
        { 0.0,  3.0,  0.0}, { 0.0, -1.0,  0.0},
        { 0.0,  0.0,  1.0}, { 0.0,  0.0, -2.0},
        { 0.5,  0.5,  0.5},
    };
    std::vector<double> uNb;
    for (const auto& dc : d) uNb.push_back(p0 + (a * dc));
    std::vector<double> w = {100.0, 5.0, 42.0, 7.0, 0.0 /*ignored*/, 3.0, 1.0};

    auto g = leastSquaresGradient<double, 3>(p0, uNb, d, w);
    for (int i = 0; i < 3; ++i)
        BOOST_CHECK_CLOSE(g[i], a[i], 1e-7);
}

BOOST_AUTO_TEST_CASE(TaylorVertexAverageReproducesAffineField)
{
    const V3 gradient{1.5, -2.0, 0.25};
    const V3 vertex{3.0, 4.0, -1.0};
    const std::vector<V3> centres = {
        {0.0, 0.0, 0.0}, {2.0, 1.0, -2.0}, {4.0, 5.0, 1.0},
    };
    const double intercept = 7.0;
    double vertexAverage = 0.0;
    for (const auto& centre : centres) {
        const double cellValue = intercept + gradient * centre;
        V3 offset = vertex;
        offset -= centre;
        vertexAverage += taylorExtrapolate<double, 3>(
            cellValue, gradient, offset);
    }

    vertexAverage /= static_cast<double>(centres.size());
    BOOST_CHECK_CLOSE(vertexAverage, intercept + gradient * vertex, 1e-10);
}

BOOST_AUTO_TEST_CASE(GradientLimiterPreservesAdmissibleAffineGradient)
{
    const V3 gradient{2.0, -1.0, 0.5};
    const std::vector<V3> offsets = {
        {0.5, 0.5, 0.5}, {-0.5, -0.5, -0.5}
    };
    const auto limited = limitGradientToBounds<double, 3>(
        10.0, gradient, offsets, 8.0, 12.0);
    for (int i = 0; i < 3; ++i) {
        BOOST_CHECK_CLOSE(limited[i], gradient[i], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(GradientLimiterClipsTaylorOvershoot)
{
    const V3 gradient{20.0, 0.0, 0.0};
    const std::vector<V3> offsets = {
        {0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}
    };
    const auto limited = limitGradientToBounds<double, 3>(
        10.0, gradient, offsets, 8.0, 12.0);
    BOOST_CHECK_CLOSE(limited[0], 4.0, 1e-12);
    for (const auto& offset : offsets) {
        const double value = taylorExtrapolate<double, 3>(
            10.0, limited, offset);
        BOOST_CHECK(value >= 8.0);
        BOOST_CHECK(value <= 12.0);
    }
}

BOOST_AUTO_TEST_CASE(LSGradientIgnoresNonFiniteConnections)
{
    const V3 expected{3.0, -2.0, 0.0};
    const std::vector<V3> offsets = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}
    };
    const std::vector<double> values = {
        8.0, 3.0, std::numeric_limits<double>::quiet_NaN()
    };
    const auto gradient = leastSquaresGradient<double, 3>(
        5.0, values, offsets);
    BOOST_CHECK_CLOSE(gradient[0], expected[0], 1e-7);
    BOOST_CHECK_CLOSE(gradient[1], expected[1], 1e-7);
    BOOST_CHECK_SMALL(gradient[2], tol);
}

// ---------------------------------------------------------------------------
//  Pi0 of an RT_0 field from face fluxes
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(PiZeroConstantFieldOnUnitCube)
{
    // Unit cube [-.5,.5]^3, faces at +-0.5 along each axis, centroids dc.
    // A constant field w = (wx, wy, wz): integrated flux across face with
    // outward normal e_i is w_i * area (area = 1).  Pi0 must return w.
    const V3 w{0.7, -1.3, 4.2};
    std::vector<V3> dc = {
        { 0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0},
        { 0.0, 0.5, 0.0}, { 0.0,-0.5, 0.0},
        { 0.0, 0.0, 0.5}, { 0.0, 0.0,-0.5},
    };
    std::vector<V3> n = {
        { 1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
        { 0.0, 1.0, 0.0}, { 0.0,-1.0, 0.0},
        { 0.0, 0.0, 1.0}, { 0.0, 0.0,-1.0},
    };
    std::vector<double> F;
    for (const auto& nc : n) {
        F.push_back(w * nc); // area = 1
    }
    auto v = piZeroFromFaceFluxes<double, 3>(F, dc, /*volume=*/1.0);
    for (int i = 0; i < 3; ++i) {
        BOOST_CHECK_CLOSE(v[i], w[i], 1e-9);
    }
}

BOOST_AUTO_TEST_CASE(PiZeroZeroVolumeIsZero)
{
    std::vector<V3> dc = {{1, 0, 0}};
    std::vector<double> F = {3.0};
    auto v = piZeroFromFaceFluxes<double, 3>(F, dc, 0.0);
    BOOST_CHECK_SMALL(v.two_norm(), tol);
}

// ---------------------------------------------------------------------------
//  Consistency: K-orthogonal grid => TPFA flux == reconstructed Darcy flux
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(KOrthogonalTpfaMatchesConsistentFlux)
{
    // Cartesian cell, isotropic K = perm.  Neighbour pressures from an affine
    // field.  The TPFA connection flux  T_c (p_i - p_j)  and the consistent
    // flux  -area_c n_c . K grad(p)  must agree => eta_sp ~ 0.
    const double perm = 100.0;
    const V3 gradp{3.0, -1.0, 0.5};
    const double pC = 50.0;
    const double dx = 10.0, area = dx * dx, half = dx / 2.0;
    const double T = perm * area / dx; // two-point transmissibility (2 * half-trans)

    std::vector<V3> d = {
        { dx, 0, 0}, {-dx, 0, 0}, {0,  dx, 0}, {0, -dx, 0}, {0, 0, dx}, {0, 0, -dx},
    };
    std::vector<V3> n = {
        { 1, 0, 0}, {-1, 0, 0}, {0,  1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    };
    std::vector<double> uNb;
    for (const auto& dc : d) {
        uNb.push_back(pC + (gradp * dc));
    }

    auto g = leastSquaresGradient<double, 3>(pC, uNb, d);
    for (int i = 0; i < 3; ++i) {
        BOOST_CHECK_CLOSE(g[i], gradp[i], 1e-7);
    }

    double maxDiff = 0.0;
    for (std::size_t c = 0; c < d.size(); ++c) {
        const double tpfa = T * (pC - uNb[c]);          // = -T (p_nb - p_i)
        double consistent = 0.0;                        // -area n . K grad
        for (int i = 0; i < 3; ++i) {
            consistent += -area * n[c][i] * perm * g[i];
        }
        maxDiff = std::max(maxDiff, std::abs(tpfa - consistent));
    }
    BOOST_CHECK_SMALL(maxDiff, 1e-6 * perm * area);
}

// ---------------------------------------------------------------------------
//  MVEM local flux mass matrix (Vohralik & Yousef, CMAME 2018)
// ---------------------------------------------------------------------------

namespace {
// Same axis-aligned unit-cube geometry as KOrthogonalTpfaMatchesConsistentFlux:
// N[f] = area*n[f], C[f] = (dx/2)*n[f] (this file's dcFace convention, i.e. the
// face is approximated by the cell-to-neighbour midpoint).
struct CubeGeom
{
    double dx, area, volume;
    std::vector<V3> N, C;
};

CubeGeom makeCube(double dx)
{
    CubeGeom cg;
    cg.dx = dx;
    cg.area = dx * dx;
    cg.volume = dx * dx * dx;
    std::vector<V3> n = {
        { 1, 0, 0}, {-1, 0, 0}, {0,  1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    };
    for (const auto& ni : n) {
        V3 Nf = ni; Nf *= cg.area;
        V3 Cf = ni; Cf *= (dx / 2.0);
        cg.N.push_back(Nf);
        cg.C.push_back(Cf);
    }
    return cg;
}
}

BOOST_AUTO_TEST_CASE(MvemConsistencyNTransposeCOverVolumeIsIdentity)
{
    // The paper's own stated validity check: N^T C / |K| ~= I; otherwise the
    // geometry is unsuitable without face subdivision.
    const auto cg = makeCube(10.0);
    Dune::FieldMatrix<double, 3, 3> NtC(0.0);
    for (std::size_t f = 0; f < cg.N.size(); ++f)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                NtC[a][b] += cg.N[f][a] * cg.C[f][b];
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            const double val = NtC[a][b] / cg.volume;
            if (a == b)
                BOOST_CHECK_CLOSE(val, 1.0, 1e-9);
            else
                BOOST_CHECK_SMALL(val, 1e-12);
        }
    }
}

BOOST_AUTO_TEST_CASE(MvemConstantFluxGivesZeroEnergy)
{
    // Essential test (constant-flux reproduction): if F is exactly the face
    // interpolation of a constant field u_alpha (F = N.u_alpha), the defect
    // z = F - N.u_alpha is exactly zero, so z^T M_K z = 0 regardless of K.
    const auto cg = makeCube(10.0);
    Dune::FieldMatrix<double, 3, 3> K(0.0);
    K[0][0] = 100.0; K[1][1] = 40.0; K[2][2] = 5.0;  // anisotropic, not K-orthogonal-special-cased
    const auto mvem = mvemFluxMassMatrix<double, 3>(cg.N, cg.C, K, cg.volume);

    const V3 uAlpha{2.0, -3.0, 1.5};
    std::vector<double> F(cg.N.size()), z(cg.N.size());
    for (std::size_t f = 0; f < cg.N.size(); ++f)
        F[f] = cg.N[f] * uAlpha;  // exact face interpolation of the constant field
    for (std::size_t f = 0; f < cg.N.size(); ++f)
        z[f] = F[f] - (cg.N[f] * uAlpha);

    std::vector<double> Mtot(mvem.nf * mvem.nf);
    for (std::size_t k = 0; k < Mtot.size(); ++k)
        Mtot[k] = mvem.Mc[k] + mvem.Ms[k];
    BOOST_CHECK_SMALL(mvemQuadForm(Mtot, z), 1e-12);
}

BOOST_AUTO_TEST_CASE(MvemConsistencyTermMatchesPi0Formula)
{
    // z^T M^c z must equal |K| (Pi0_K(F) - u_alpha)^T K^{-1} (Pi0_K(F) - u_alpha)
    // exactly (the decomposition's first term), for an ARBITRARY (nonzero
    // defect) F, checked here by direct evaluation via Pi0_K = P F, P's
    // columns C[f]/volume, independent of mvemFluxMassMatrix's own internals.
    const auto cg = makeCube(10.0);
    Dune::FieldMatrix<double, 3, 3> K(0.0);
    K[0][0] = 100.0; K[1][1] = 40.0; K[2][2] = 5.0;
    const auto mvem = mvemFluxMassMatrix<double, 3>(cg.N, cg.C, K, cg.volume);

    // An arbitrary, non-P0-consistent flux vector F.
    std::vector<double> F = {120.0, -80.0, 45.0, -60.0, 15.0, -25.0};
    const V3 uAlpha{1.0, 0.5, -0.3};
    std::vector<double> z(F.size());
    for (std::size_t f = 0; f < F.size(); ++f) {
        const double NuAlpha = cg.N[f] * uAlpha;
        z[f] = F[f] - NuAlpha;
    }

    // Pi0_K(F) = P F = sum_f C[f]/volume * F[f]
    V3 pi0(0.0);
    for (std::size_t f = 0; f < F.size(); ++f) {
        V3 term = cg.C[f];
        term *= (F[f] / cg.volume);
        pi0 += term;
    }
    V3 diff = pi0; diff -= uAlpha;
    Dune::FieldVector<double, 3> KinvDiff(0.0);
    Dune::FieldMatrix<double, 3, 3> Kfac(K);
    Kfac.solve(KinvDiff, diff);  // K^{-1} diff, per the decomposition (K^{-1}, not K)
    const double expected = cg.volume * (diff * KinvDiff);

    BOOST_CHECK_CLOSE(mvemQuadForm(mvem.Mc, z), expected, 1e-6);
}

BOOST_AUTO_TEST_CASE(FullTensorStarNormReducesToDiagonalAndMatchesMvemP0)
{
    // (a) For a diagonal K it must equal weightedStarNorm(applyInvSqrtPerm)^2.
    const auto cg = makeCube(10.0);
    Dune::FieldMatrix<double, 3, 3> Kd(0.0);
    Kd[0][0] = 100.0; Kd[1][1] = 40.0; Kd[2][2] = 5.0;
    const V3 v{2.0, -3.0, 1.5};
    const double w = 1.3;  // D_K^{l/2}

    const auto s = applyInvSqrtPerm<double, 3>(v, {Kd[0][0], Kd[1][1], Kd[2][2]});
    const double diagN = weightedStarNorm<double, 3>(s, w, cg.volume);
    const double full  = fullTensorStarNorm2<double, 3>(v, Kd, w, cg.volume);
    BOOST_CHECK_CLOSE(full, diagN * diagN, 1e-9);

    // (b) For a P0 field v, fullTensorStarNorm2 (with w=1) must equal
    // z^T M_K z on z = N v exactly (the mimetic consistency property; the
    // stability term annihilates N v).
    Dune::FieldMatrix<double, 3, 3> K(0.0);
    K[0][0] = 100.0; K[1][1] = 40.0; K[2][2] = 5.0;
    K[0][1] = K[1][0] = 12.0;  // off-diagonal: exercises the full tensor
    const auto mvem = mvemFluxMassMatrix<double, 3>(cg.N, cg.C, K, cg.volume);
    std::vector<double> Mtot(mvem.nf * mvem.nf);
    for (std::size_t k = 0; k < Mtot.size(); ++k)
        Mtot[k] = mvem.Mc[k] + mvem.Ms[k];
    std::vector<double> z(cg.N.size());
    for (std::size_t f = 0; f < cg.N.size(); ++f)
        z[f] = cg.N[f] * v;  // exact face interpolation of the P0 field
    const double viaMvem = mvemQuadForm(Mtot, z);
    const double viaFull = fullTensorStarNorm2<double, 3>(v, K, 1.0, cg.volume);
    BOOST_CHECK_CLOSE(viaMvem, viaFull, 1e-6);
}

// ---------------------------------------------------------------------------
//  Weighted norm scaling
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(WeightedStarNormScaling)
{
    const V3 v{3.0, 4.0, 0.0};             // |v| = 5
    const V3 perm{4.0, 4.0, 4.0};          // K^{-1/2} = 1/2
    auto iv = applyInvSqrtPerm<double, 3>(v, perm);
    BOOST_CHECK_CLOSE(iv.two_norm(), 2.5, tol);

    const double DkPow = std::pow(3.0, 0.5); // D_K^{l/2}, l=1, D_K=3
    const double vol = 8.0;
    const double got = weightedStarNorm<double, 3>(iv, DkPow, vol);
    BOOST_CHECK_CLOSE(got, DkPow * std::sqrt(vol) * 2.5, tol);
}

BOOST_AUTO_TEST_CASE(ComponentMobilityIncludesBlackOilMixing)
{
    // Two conserved components (oil, gas), three canonical phase slots
    // (water is inactive).  C includes b_o, R_v b_g, R_s b_o, b_g.
    const std::array<std::array<double, 3>, 2> C{{
        {{0.0, 2.0, 0.5}},
        {{0.0, 3.0, 4.0}}
    }};
    const std::array<double, 3> mobility{{0.0, 5.0, 7.0}};
    const auto L = componentMobilityMatrix<double, 2, 3>(C, mobility, 0.0);

    BOOST_CHECK_CLOSE(L[0][0], 21.75, tol);
    BOOST_CHECK_CLOSE(L[0][1], 44.0, tol);
    BOOST_CHECK_CLOSE(L[1][0], 44.0, tol);
    BOOST_CHECK_CLOSE(L[1][1], 157.0, tol);
}

BOOST_AUTO_TEST_CASE(ComponentMobilityCholeskyProducesInverseEnergy)
{
    const std::array<std::array<double, 2>, 2> C{{
        {{1.0, 0.0}},
        {{0.0, 1.0}}
    }};
    const std::array<double, 2> mobility{{3.0, 4.0}};
    const auto L = componentMobilityMatrix<double, 2, 2>(C, mobility, 0.0);
    const auto lower = choleskyLower<double, 2>(L);
    const auto transformed =
        solveLower<double, 2>(lower, std::array<double, 2>{{6.0, 8.0}});

    // z^T L^{-1} z = 6^2/3 + 8^2/4 = 28.
    BOOST_CHECK_CLOSE(transformed[0] * transformed[0]
                      + transformed[1] * transformed[1],
                      28.0, tol);
}

BOOST_AUTO_TEST_CASE(ComponentMobilityCorrelatedInverseEnergy)
{
    const std::array<std::array<double, 2>, 2> C{{
        {{1.0, 1.0}},
        {{0.0, 1.0}}
    }};
    const std::array<double, 2> mobility{{2.0, 3.0}};
    const auto L = componentMobilityMatrix<double, 2, 2>(C, mobility, 0.0);
    const auto lower = choleskyLower<double, 2>(L);
    const std::array<double, 2> defect{{4.0, -1.0}};
    const auto transformed = solveLower<double, 2>(lower, defect);

    // L = [5 3; 3 3], L^-1 = [0.5 -0.5; -0.5 5/6].
    const double expected = 0.5 * defect[0] * defect[0]
        - defect[0] * defect[1]
        + (5.0 / 6.0) * defect[1] * defect[1];
    BOOST_CHECK_CLOSE(transformed[0] * transformed[0]
                      + transformed[1] * transformed[1],
                      expected, tol);
}

BOOST_AUTO_TEST_CASE(ComponentMobilityRegularisationIsScaleInvariant)
{
    const std::array<std::array<double, 2>, 2> C{{
        {{1.0, 0.4}},
        {{0.3, 2.0}}
    }};
    const std::array<double, 2> mobility{{1e-10, 4.0}};
    const std::array<double, 2> defect{{2.0, -3.0}};
    constexpr double floorFraction = 1e-3;

    const auto L = componentMobilityMatrix<double, 2, 2>(
        C, mobility, floorFraction);
    const auto transformed = solveLower<double, 2>(
        choleskyLower<double, 2>(L), defect);
    const double energy = transformed[0] * transformed[0]
        + transformed[1] * transformed[1];

    constexpr std::array<double, 2> scale{{1000.0, 0.01}};
    auto scaledC = C;
    auto scaledDefect = defect;
    for (std::size_t component = 0; component < scale.size(); ++component) {
        scaledDefect[component] *= scale[component];
        for (double& value : scaledC[component]) {
            value *= scale[component];
        }
    }
    const auto scaledL = componentMobilityMatrix<double, 2, 2>(
        scaledC, mobility, floorFraction);
    const auto scaledTransformed = solveLower<double, 2>(
        choleskyLower<double, 2>(scaledL), scaledDefect);
    const double scaledEnergy =
        scaledTransformed[0] * scaledTransformed[0]
        + scaledTransformed[1] * scaledTransformed[1];

    BOOST_CHECK_CLOSE(scaledEnergy, energy, 1e-8);
}

BOOST_AUTO_TEST_CASE(ComponentMobilityFloorControlsPhaseDegeneracy)
{
    const std::array<std::array<double, 2>, 2> C{{
        {{1.0, 0.0}},
        {{0.0, 1.0}}
    }};
    const std::array<double, 2> mobility{{0.0, 2.0}};
    const auto L = componentMobilityMatrix<double, 2, 2>(C, mobility, 1e-3);
    const auto transformed = solveLower<double, 2>(
        choleskyLower<double, 2>(L), std::array<double, 2>{{1.0, 1.0}});

    BOOST_CHECK(std::isfinite(transformed[0]));
    BOOST_CHECK(std::isfinite(transformed[1]));
    BOOST_CHECK_CLOSE(L[0][0], 2e-3, 1e-10);
    BOOST_CHECK_CLOSE(L[1][1], 2.002, 1e-10);
}

BOOST_AUTO_TEST_CASE(ComponentMobilityActiveThreePhaseInverseEnergy)
{
    const std::array<std::array<double, 3>, 3> C{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}
    }};
    const std::array<double, 3> mobility{{2.0, 3.0, 5.0}};
    const std::array<double, 3> defect{{2.0, 3.0, 5.0}};
    const auto L = componentMobilityMatrix<double, 3, 3>(C, mobility, 0.0);
    const auto transformed = solveLower<double, 3>(
        choleskyLower<double, 3>(L), defect);
    const double energy = transformed[0] * transformed[0]
        + transformed[1] * transformed[1]
        + transformed[2] * transformed[2];

    // z^T L^-1 z = 2^2/2 + 3^2/3 + 5^2/5 = 10.
    BOOST_CHECK_CLOSE(energy, 10.0, tol);
}

BOOST_AUTO_TEST_CASE(ComponentMobilityInRangeDefectIsFloorStable)
{
    // The gas phase has disappeared. The two active columns span e_w and
    // (0,1,2), and the chosen defect lies in that physical flux range.
    const std::array<std::array<double, 3>, 3> C{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 2.0, 1.0}}
    }};
    const std::array<double, 3> mobility{{4.0, 9.0, 0.0}};
    const std::array<double, 3> defect{{1.0, 1.0, 2.0}};
    auto energy = [&](double floorFraction) {
        const auto L = componentMobilityMatrix<double, 3, 3>(
            C, mobility, floorFraction);
        const auto transformed = solveLower<double, 3>(
            choleskyLower<double, 3>(L), defect);
        return transformed[0] * transformed[0]
            + transformed[1] * transformed[1]
            + transformed[2] * transformed[2];
    };

    const double coarseFloor = energy(1e-6);
    const double fineFloor = energy(1e-10);
    const double limitingEnergy = 1.0 / 4.0 + 1.0 / 9.0;
    BOOST_CHECK_CLOSE(fineFloor, limitingEnergy, 1e-5);
    BOOST_CHECK_CLOSE(coarseFloor, fineFloor, 1e-3);
}

BOOST_AUTO_TEST_CASE(ComponentMobilityOutOfRangeDefectIsFloorDominated)
{
    // With the gas phase absent, (0,-2,1) is orthogonal to both active phase
    // columns. Its regularised inverse energy must diverge as the artificial
    // mobility floor is removed. This is the failure mode a componentwise
    // reconstruction can trigger when its defect does not remain in range(C).
    const std::array<std::array<double, 3>, 3> C{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 2.0, 1.0}}
    }};
    const std::array<double, 3> mobility{{4.0, 9.0, 0.0}};
    const std::array<double, 3> defect{{0.0, -2.0, 1.0}};
    auto energy = [&](double floorFraction) {
        const auto L = componentMobilityMatrix<double, 3, 3>(
            C, mobility, floorFraction);
        const auto transformed = solveLower<double, 3>(
            choleskyLower<double, 3>(L), defect);
        return transformed[0] * transformed[0]
            + transformed[1] * transformed[1]
            + transformed[2] * transformed[2];
    };

    const double coarseFloor = energy(1e-6);
    const double fineFloor = energy(1e-10);
    BOOST_CHECK(std::isfinite(coarseFloor));
    BOOST_CHECK(std::isfinite(fineFloor));
    BOOST_CHECK_GT(fineFloor, 1000.0 * coarseFloor);
}

BOOST_AUTO_TEST_CASE(ComponentMobilityZeroStateAcceptsOnlyZeroDefect)
{
    const std::array<std::array<double, 3>, 3> C{{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}
    }};
    const std::array<double, 3> mobility{{0.0, 0.0, 0.0}};
    const auto L = componentMobilityMatrix<double, 3, 3>(C, mobility, 1e-6);
    const auto transformed = solveLower<double, 3>(
        choleskyLower<double, 3>(L), std::array<double, 3>{{0.0, 0.0, 0.0}});

    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_CHECK_EQUAL(L[i][i], 0.0);
        BOOST_CHECK(std::isfinite(transformed[i]));
        BOOST_CHECK_EQUAL(transformed[i], 0.0);
    }
}
