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
 * \brief Local reconstructions for the a posteriori spatial/temporal error estimators.
 *
 * These are the low-level, geometry-only ingredients of the estimators
 * eta_sp (Darcy / grid-orientation) and eta_time from the a posteriori
 * framework of Ahmed et al.  They are deliberately free functions operating on
 * plain per-cell / per-connection data so they can be unit tested without a
 * simulator.
 */
#ifndef OPM_APOSTERIORI_RECONSTRUCTION_HPP
#define OPM_APOSTERIORI_RECONSTRUCTION_HPP

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace Opm::APosteriori {

/*!
 * \brief Weighted local equilibration indicator from an integrated balance residual.
 *
 * For a cellwise-constant balance residual r_K = R_K/|K|,
 *
 *   eta_eq,K = sqrt(tau) epsilon^{-1/2} c_K^{-1/2} h_K D_K^{ell/2}
 *              ||r_K||_K
 *            = sqrt(tau) epsilon^{-1/2} c_K^{-1/2} h_K D_K^{ell/2}
 *              |R_K| / sqrt(|K|).
 */
template<class Scalar>
Scalar
equilibrationIndicator(Scalar integratedResidual,
                       Scalar dt,
                       Scalar epsilon,
                       Scalar cKK,
                       Scalar hK,
                       Scalar dLambdaPow,
                       Scalar volume)
{
    if (!(dt >= Scalar{0}) || !(epsilon > Scalar{0})
        || !(cKK > Scalar{0}) || !(volume > Scalar{0})) {
        return std::numeric_limits<Scalar>::quiet_NaN();
    }
    return std::sqrt(dt / (epsilon * cKK * volume))
        * hK * dLambdaPow * std::abs(integratedResidual);
}

/*!
 * \brief Cell-centred least-squares gradient.
 *
 * Given a cell value \p uCell at the cell centre and, for each connection
 * \c c, the neighbour value \p uNb[c] and the offset \p d[c] = x_nb - x_cell,
 * returns the gradient g minimising
 *
 *     sum_c w_c ( g . d[c] - (uNb[c] - uCell) )^2 .
 *
 * The weights \p w are supplied explicitly (the a posteriori framework uses the
 * connection transmissibilities, so that -K g reproduces the two-point flux on
 * a K-orthogonal grid and the residual -K g - flux is exactly the
 * grid-orientation error eta_sp measures); if \p w is empty the geometric
 * default w_c = 1 / |d[c]|^2 is used.  Exact for affine fields.
 *
 * \tparam dim   world dimension (2 or 3).
 */
template<class Scalar, int dim>
Dune::FieldVector<Scalar, dim>
leastSquaresGradient(Scalar uCell,
                     std::span<const Scalar> uNb,
                     std::span<const Dune::FieldVector<Scalar, dim>> d,
                     std::span<const Scalar> w = {})
{
    Dune::FieldMatrix<Scalar, dim, dim> M(Scalar{0});
    Dune::FieldVector<Scalar, dim> rhs(Scalar{0});

    const std::size_t n = uNb.size();
    for (std::size_t c = 0; c < n; ++c) {
        const auto& dc = d[c];
        const Scalar len2 = dc.two_norm2();
        if (len2 <= Scalar{0}) {
            continue;
        }
        const Scalar wc = w.empty() ? (Scalar{1} / len2) : w[c];
        if (wc <= Scalar{0}) {
            continue;
        }
        const Scalar du = uNb[c] - uCell;
        for (int i = 0; i < dim; ++i) {
            rhs[i] += wc * dc[i] * du;
            for (int j = 0; j < dim; ++j) {
                M[i][j] += wc * dc[i] * dc[j];
            }
        }
    }

    Dune::FieldVector<Scalar, dim> g(Scalar{0});
    // Regularise a rank-deficient system (e.g. a cell with < dim connections).
    Scalar tr{0};
    for (int i = 0; i < dim; ++i) {
        tr += M[i][i];
    }
    if (tr > Scalar{0}) {
        const Scalar eps = std::numeric_limits<Scalar>::epsilon() * tr;
        for (int i = 0; i < dim; ++i) {
            M[i][i] += eps;
        }
        try {
            M.solve(g, rhs);
        }
        catch (...) {
            g = Scalar{0};
        }
    }
    return g;
}

/*!
 * \brief Taylor extrapolation of a cell-centred value to a vertex.
 *
 * Given u_K, a reconstructed cell gradient g_K, and x_a-x_K, returns
 *
 *     u_{K->a} = u_K + g_K . (x_a-x_K).
 *
 * Averaging these values over all cells sharing a vertex gives a single
 * vertex value and therefore an H1-conforming nodal reconstruction.
 */
template<class Scalar, int dim>
Scalar
taylorExtrapolate(Scalar uCell,
                  const Dune::FieldVector<Scalar, dim>& gradient,
                  const Dune::FieldVector<Scalar, dim>& vertexOffset)
{
    return uCell + gradient * vertexOffset;
}

/*!
 * \brief Constant part of a lowest-order Raviart--Thomas field from its face fluxes.
 *
 * For w in RT_0(K) with prescribed \e integrated normal fluxes
 * F_c = \int_c w . n_{K,c}, the L^2 projection onto constant vector fields is
 *
 *     Pi0_K w = (1/|K|) sum_c F_c ( x_c - x_K ) ,
 *
 * with x_c the face (or NNC-overlap) centroid.  Exact for RT_0.
 *
 * \param F   integrated component flux per connection, oriented out of the cell.
 * \param dc  x_c - x_K per connection.
 * \param volume  |K|.
 */
template<class Scalar, int dim>
Dune::FieldVector<Scalar, dim>
piZeroFromFaceFluxes(std::span<const Scalar> F,
                     std::span<const Dune::FieldVector<Scalar, dim>> dc,
                     Scalar volume)
{
    Dune::FieldVector<Scalar, dim> v(Scalar{0});
    if (volume <= Scalar{0}) {
        return v;
    }
    const std::size_t n = F.size();
    for (std::size_t c = 0; c < n; ++c) {
        v.axpy(F[c], dc[c]);
    }
    v /= volume;
    return v;
}

/*!
 * \brief Weighted energy (dual) norm of a constant flux vector on a cell.
 *
 *     || v ||_{*,K} = || d_Lambda^{l/2} K^{-1/2} v ||_K
 *                   \approx  D_K^{l/2} * sqrt(|K|) * |K^{-1/2} v| ,
 *
 * bounding the near-well weight d_Lambda^l by D_K^l (guaranteed, see the
 * appendix).  \p invKsqrtV is K^{-1/2} v (already applied by the caller), and
 * \p dLambdaPow is D_K^{l/2}.
 */
template<class Scalar, int dim>
Scalar
weightedStarNorm(const Dune::FieldVector<Scalar, dim>& invKsqrtV,
                 Scalar dLambdaPow,
                 Scalar volume)
{
    return dLambdaPow * std::sqrt(std::max(volume, Scalar{0})) * invKsqrtV.two_norm();
}

/*!
 * \brief Weighted energy norm SQUARED of a constant (P0) flux vector on a
 *        cell, using the FULL permeability tensor:
 *
 *     || v ||^2_{*,K}  =  D_K^l * |K| * (v . K^{-1} v).
 *
 * This is the tensor-consistent companion to weightedStarNorm above (which
 * uses only the diagonal of K via applyInvSqrtPerm). For a P0 field it
 * coincides exactly with the mimetic form z^T M_K z evaluated on z = N v:
 * the consistency part gives |K| v.K^{-1}v exactly and the stability part
 * annihilates N v (M^s N = 0 by construction). K^{-1} is applied by an SPD
 * solve, not formed. A singular K (solve throws) returns NaN so the caller
 * can propagate "invalid" rather than mistake it for a small estimate.
 */
template<class Scalar, int dim>
Scalar
fullTensorStarNorm2(const Dune::FieldVector<Scalar, dim>& v,
                    const Dune::FieldMatrix<Scalar, dim, dim>& K,
                    Scalar dLambdaPow,
                    Scalar volume)
{
    try {
        Dune::FieldMatrix<Scalar, dim, dim> Kfac(K);
        Dune::FieldVector<Scalar, dim> KinvV(Scalar{0});
        Kfac.solve(KinvV, v);
        const Scalar q = v * KinvV;
        return dLambdaPow * dLambdaPow * std::max(volume, Scalar{0}) * std::max(q, Scalar{0});
    }
    catch (...) {
        return std::numeric_limits<Scalar>::quiet_NaN();
    }
}

/*!
 * \brief Local MVEM (mimetic/virtual-element) flux mass matrix M_K and its
 *        consistency/stability split, following Vohralik & Yousef, "A simple
 *        a posteriori estimate on general polytopal meshes with applications
 *        to complex porous media flows", CMAME 331 (2018), Sections 3 and 6
 *        (Definition 3.3, Lemmas 3.5-3.7, Remark 3.15) -- the generic MFD
 *        consistency+stability decomposition (Brezzi-Lipnikov-Simoncini
 *        family), not the exact RT Schur complement of Lemma 3.5, which is
 *        the "certified" but much more involved alternative (needs a
 *        simplicial sub-tessellation and a local mixed FE solve).
 *
 * Given the nf GEOMETRIC faces of a polyhedral cell K (NNCs are excluded --
 * see the OPM-specific notes at the call site), \p N (row f = A_f n_{K,f},
 * the area-weighted outward normal) and \p C (row f = x_f-x_K, the face
 * centroid offset from the cell centre), the cell's full permeability tensor
 * \p K and volume \p volume, builds:
 *
 *   P    = C^T / |K|
 *   M^c  = (1/|K|) C K^{-1} C^T                      (consistency)
 *   PiF  = N P                                        (face-DOF projector)
 *   D_K  = diag_f max{ C_f.(K^{-1}C_f)/|K|, eps*trace(M^c)/nf }
 *   M^s  = (I-PiF)^T D_K (I-PiF)                       (stability)
 *   M_K  = M^c + M^s
 *
 * Returns M^c and M^s separately (both nf x nf, row-major flattened) so the
 * caller can evaluate z^T M_K z = z^T M^c z + z^T M^s z (mvemQuadForm below);
 * z^T M^c z reproduces the existing Pi0-moment (T1^2) construction exactly
 * (Lemma 3.5's consistency property: exact for any P0-on-K field), while
 * z^T M^s z replaces the old c_KK^{-1/2}*|div| surrogate (T3) with the
 * stability term's unresolved-face-flux-mode measure.
 *
 * K^{-1} is never formed explicitly -- each column of K^{-1}C^T is obtained
 * by solving K x = C_f (K is SPD, Dune::FieldMatrix::solve uses its own LU
 * factorization). Degenerate cells (volume <= 0, or K singular so the solve
 * throws) return zero matrices.
 */
template<class Scalar, int dim>
struct MvemMatrices
{
    std::size_t nf {0};
    std::vector<Scalar> Mc;  // nf*nf, row-major
    std::vector<Scalar> Ms;  // nf*nf, row-major
};

template<class Scalar, int dim>
MvemMatrices<Scalar, dim>
mvemFluxMassMatrix(const std::vector<Dune::FieldVector<Scalar, dim>>& N,
                   const std::vector<Dune::FieldVector<Scalar, dim>>& C,
                   const Dune::FieldMatrix<Scalar, dim, dim>& K,
                   Scalar volume,
                   Scalar epsilon = Scalar{1e-2})
{
    const std::size_t nf = N.size();
    MvemMatrices<Scalar, dim> out;
    out.nf = nf;
    out.Mc.assign(nf * nf, Scalar{0});
    out.Ms.assign(nf * nf, Scalar{0});
    if (nf == 0 || !(volume > Scalar{0}))
        return out;

    // KinvC[f] = K^{-1} C[f]  (solve, never invert K explicitly)
    std::vector<Dune::FieldVector<Scalar, dim>> KinvC(nf);
    for (std::size_t f = 0; f < nf; ++f) {
        try {
            Dune::FieldMatrix<Scalar, dim, dim> Kfac(K);
            Dune::FieldVector<Scalar, dim> x(Scalar{0});
            Kfac.solve(x, C[f]);
            KinvC[f] = x;
        }
        catch (...) {
            return out;  // singular K: leave Mc/Ms zero rather than propagate garbage
        }
    }

    // M^c_{f,f'} = C[f] . KinvC[f'] / volume  (symmetric since K, hence K^{-1}, is symmetric)
    for (std::size_t f = 0; f < nf; ++f)
        for (std::size_t fp = 0; fp < nf; ++fp)
            out.Mc[f * nf + fp] = (C[f] * KinvC[fp]) / volume;

    Scalar traceMc = 0;
    for (std::size_t f = 0; f < nf; ++f)
        traceMc += out.Mc[f * nf + f];

    // D_K[f] = max{ C[f].KinvC[f]/volume, epsilon*trace(Mc)/nf }
    std::vector<Scalar> DK(nf);
    for (std::size_t f = 0; f < nf; ++f) {
        const Scalar consistPart = (C[f] * KinvC[f]) / volume;
        DK[f] = std::max(consistPart, epsilon * traceMc / static_cast<Scalar>(nf));
    }

    // PiF = N P, P's columns are C[f]/volume, so PiF_{f,f'} = (N[f] . C[f'])/volume
    std::vector<Scalar> PiF(nf * nf);
    for (std::size_t f = 0; f < nf; ++f)
        for (std::size_t fp = 0; fp < nf; ++fp)
            PiF[f * nf + fp] = (N[f] * C[fp]) / volume;

    // M^s = (I-PiF)^T D_K (I-PiF), D_K diagonal:
    //   M^s_{f,f'} = sum_g (I-PiF)_{g,f} DK[g] (I-PiF)_{g,f'}
    for (std::size_t f = 0; f < nf; ++f) {
        for (std::size_t fp = 0; fp < nf; ++fp) {
            Scalar sum = 0;
            for (std::size_t g = 0; g < nf; ++g) {
                const Scalar Igf  = (g == f)  ? (Scalar{1} - PiF[g * nf + f])  : (Scalar{0} - PiF[g * nf + f]);
                const Scalar Igfp = (g == fp) ? (Scalar{1} - PiF[g * nf + fp]) : (Scalar{0} - PiF[g * nf + fp]);
                sum += Igf * DK[g] * Igfp;
            }
            out.Ms[f * nf + fp] = sum;
        }
    }
    return out;
}

//! z^T M z for a flat row-major nf x nf matrix M and nf-vector z.
template<class Scalar>
Scalar mvemQuadForm(const std::vector<Scalar>& M, const std::vector<Scalar>& z)
{
    const std::size_t nf = z.size();
    Scalar sum = 0;
    for (std::size_t f = 0; f < nf; ++f) {
        Scalar row = 0;
        for (std::size_t fp = 0; fp < nf; ++fp)
            row += M[f * nf + fp] * z[fp];
        sum += z[f] * row;
    }
    return sum;
}

/*!
 * \brief Apply K^{-1/2} to a vector for a diagonal (or scalar) permeability.
 *
 * For anisotropic full-tensor K the caller should pass the eigenbasis; here we
 * use the principal (diagonal) values \p perm which is what OPM stores per cell.
 */
template<class Scalar, int dim>
Dune::FieldVector<Scalar, dim>
applyInvSqrtPerm(const Dune::FieldVector<Scalar, dim>& v,
                 const Dune::FieldVector<Scalar, dim>& perm)
{
    Dune::FieldVector<Scalar, dim> r(Scalar{0});
    for (int i = 0; i < dim; ++i) {
        r[i] = (perm[i] > Scalar{0}) ? v[i] / std::sqrt(perm[i]) : Scalar{0};
    }
    return r;
}

} // namespace Opm::APosteriori

#endif // OPM_APOSTERIORI_RECONSTRUCTION_HPP
