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
 * \brief A posteriori spatial (\c eta_sp) and temporal (\c eta_time) error
 *        estimators for the fully implicit black-oil scheme.
 *
 * These are the two "discretization" estimators of the energy-type a posteriori
 * framework of Ahmed et al.:
 *
 *   eta_sp,K,a   = sqrt(tau^n)     * || Pi0_K W_a - Pi0_K u_a ||_{*,K}
 *   eta_time,K,a = sqrt(tau^n / 3) * || u_a^{n} - u_a^{n-1}  ||_{*,K}
 *
 * with, following the paper's reconstructions (Section "Flux reconstructions"):
 *   - W_a  the "discrete constitutive" flux: the scheme's own TPFA component
 *          flux (eq. Main_problem_model_flux) at the current potentials, read
 *          from LocalResidual::computeFlux, and used through its moments
 *          Pi0_K W_a = V_K^{-1} sum_c F_{a,ic}(x_c - x_K)  and
 *          div W_a|_K = V_K^{-1} sum_c F_{a,ic};
 *   - u_a  the "continuous constitutive" flux, eq. (constitutive_flux):
 *          v_hat_beta = -lambda_beta(s_hat) K (grad p_hat_beta - rho_beta g grad z),
 *          u_w = b_w v_w, u_o = b_o v_o + r_v b_g v_g, u_g = b_g v_g + r_s b_o v_o,
 *          in surface-volume units.  grad p_hat_beta is, by default
 *          (PressureRecon::PatchAverageLift), the gradient of the H1-conforming
 *          lift of eq. (eq:averaging): vertex patch averages of the phase
 *          pressures, extended by mean-value coordinates (which reproduce affine
 *          fields, so grad p_hat|_K is the least-squares fit of the nodal values;
 *          the bubble of eq. (eq:averaging_bubble) has zero cell-mean gradient
 *          and does not affect the piecewise-constant u_a).  Capillary gradients
 *          are retained by lifting each phase pressure.  PressureRecon::ConnectionLS
 *          selects instead a transmissibility-weighted LS fit of the raw
 *          connection pressure drops, for comparison;
 *   - lambda_beta(s_hat) and grad p_hat_beta are BOTH evaluated from a single
 *     lifted state per cell -- point saturations s_hat(x_K) (vertex patch
 *     average, optionally bubble-corrected to the FV cell mean) fed through the
 *     real MaterialLaw::relativePermeabilities -- so u_a mimics the continuous
 *     flux consistently rather than mixing a reconstructed gradient with
 *     discrete-state coefficients (setUseLiftedRelperm(), on by default).  b_beta,
 *     rho_beta, R_s, R_v and the viscosities remain at the FV cell state (Tier A;
 *     see the Scope note below);
 *   - || . ||_{*,K}  the weighted dual norm  || d_Lambda^{l/2} K^{-1/2} . ||_K,
 *          the near-well weight d_Lambda^l bounded by D_K^l away from wells.
 *          l = 0 (default) => plain energy norm; setWeightExponent()/setWellCells()
 *          activate the near-well scaling.
 *
 * eta_D is evaluated through the (paper eq. flux_est_cp) split, with W_a in
 * place of Theta_a:
 *     ||W_a - u_a||_{*,K}  <=  ||Pi0_K(W_a - u_a)||_{*,K}          (T1)
 *                            + ||(I - Pi0_K) u_a||_{*,K}          (T2 = 0, u_a in P0(K)^d)
 *                            + (1/pi) h_K c_KK^{-1/2} D_K^{l/2} ||div W_a||_K   (T3)
 * and eta_sp := eta_D.  As the paper's Section "Evaluation of the estimators on
 * corner-point grids" (iii) makes explicit: T1 is a genuine computable quantity
 * (exact moments of a specified constant-per-face RTN/VEM field), but T3 is a
 * *practical indicator*, not a certified bound -- "divergence alone cannot
 * control (I-Pi0_K)Theta" without an assembled, stability-certified RT/VEM
 * lifting, which is not built here.  eta_sp below is therefore reported as a
 * diagnostic, matching the paper's own caveat, except on the affine
 * (Cartesian / K-orthogonal) cells identified in Remark rem:assemble, where T1
 * alone already reproduces ||W_a-u_a||_{*,K} to machine precision (see
 * KOrthogonalTpfaMatchesConsistentFlux in the unit tests).  The per-cell
 * numerics are delegated to the unit-tested free functions in
 * APosterioriReconstruction.hpp.
 *
 * T1/T3 split, and why T1-only drives control (2026-09-04): a full-run SPE9
 * cell trace confirmed T3 is not merely "loose" but genuinely uncertified as
 * implemented -- it uses a plain scalar Poincare constant (1/pi) applied to
 * the divergence of the raw, unequilibrated TPFA flux, with no actual RT/VEM
 * lifting of the face-flux moments behind that constant, and (since
 * divergence is scalar) no valid way to make c_KK^{-1/2} direction-dependent
 * either. In cells with near-degenerate permeability (SPE9's low-permz
 * layers, c_KK ~ 1e-16-1e-18) this surrogate exceeds T1 by 5-800x and is the
 * entire source of the estimator's observed ~1e9-1e10 aggregate magnitude --
 * independent of which component it shows up in (confirmed both before and
 * after the reference-density mass scaling below, which only relabels which
 * component's T3 dominates, not whether T3 dominates).
 *
 * Mimetic (MVEM) replacement (2026-09-05): rather than the ad hoc T3,
 * etaSpatialMimetic() implements the generic MFD consistency+stability
 * decomposition of Vohralik & Yousef, "A simple a posteriori estimate on
 * general polytopal meshes with applications to complex porous media flows",
 * CMAME 331 (2018), Sections 3 and 6 (Definition 3.3, Lemmas 3.5-3.7,
 * Remark 3.15): for a face-flux defect z (either W_a-u_a's own face-normal
 * moments, or eta_lin's Theta_lin moments directly), z^T M_K z, with M_K =
 * M^c + M^s built from the cell's geometry and FULL permeability tensor --
 * see mvemFluxMassMatrix's doc comment for the exact construction. M^c (the
 * consistency term) reproduces T1^2 exactly; M^s (the stability term)
 * replaces T3, measuring the unresolved face-flux modes via a proper
 * face-DOF-projector kernel rather than multiplying the raw divergence by
 * c_KK^{-1/2}. This holds with mesh-size-independent equivalence (Lemma 3.7),
 * not an ad hoc heuristic, though NNCs are still excluded (no face
 * normal/centroid to build N/C from -- treated as nonlocal sources, same as
 * T1's Pi0 moment). An earlier attempt (dividing the raw defect by OPM's own
 * per-connection transmissibility directly) was tried and discarded: it
 * conflated the consistency and kernel parts under one weight and used a
 * transmissibility that carries deck multipliers (NTG etc.) inconsistent
 * with T1/T3's raw-permeability basis, producing an aggregate LARGER than
 * even the discredited T1+T3. The corrected MVEM construction instead
 * empirically tracks 10-20% above T1 (never a blowup) and 3-4x below T1+T3
 * across SPE9 steps, matching theoretical expectations. etaSpatialMimetic()
 * (not etaSpatialT1()) is now what Criteria_space_time_balance and
 * --enable-aposteriori-timestep-control actually use; etaSpatialT1() and
 * etaSpatial() (T1+T3) remain for research/reporting comparison, all three
 * printed in the per-iteration table. The SAME construction replaces
 * eta_lin's flux term (see its assembly site).
 *
 * This covers ONLY the flux-energy term of the paper's full Theorem 3.12
 * three-term estimate (Prager-Synge duality: flux-energy + a cross term
 * (uh,grad sh)_K + a potential-reconstruction stiffness term S_K^t Ŝ_FE,K S_K,
 * eq. 3.12-3.26). The cross term needs u_alpha related to an H1-conforming
 * potential reconstruction s_h (Definition 3.8 -- close to, but not
 * identical to, the vertex-patch-averaged potentials already used to build
 * u_alpha here); the stiffness term needs a P1-FEM stiffness/mass assembly
 * on a simplicial sub-tessellation of each polytopal cell (eq. 3.16-3.21,
 * Remark 3.9 -- closed-form from cell/vertex geometry alone, no physical
 * submesh construction needed, but genuine new machinery). Neither is
 * implemented; tracked as the next major piece of this work.
 *
 * Scope (documented simplifications, matching the paper's remarks):
 *   - isothermal, no molecular diffusion / dispersion;
 *   - the H1 lift uses vertex patch averages + an affine (least-squares) fit in
 *     place of the full mean-value-coordinate interpolant + bubble; equivalently
 *     u_a is taken piecewise constant on K (lowest order) => T2 = 0.  The
 *     sub-tessellation S_K (for the T2 oscillation and the exact *-norm
 *     quadrature) is not yet built;
 *   - relperm is re-evaluated at the lifted saturation (see above); b_beta,
 *     rho_beta, R_s, R_v stay at the FV cell state -- recomputing them from PVT
 *     at the lifted p_hat_o (with the primary-variable-switching-like logic the
 *     paper's postprocessing section uses for R_s=r_s(p_hat_o), R_v=r_v(p_hat_o))
 *     is Tier B, not yet implemented;
 *   - C_PW,K = 1/pi is used (Payne-Weinberger; per the paper, certified for
 *     convex cells but T3 itself is only a practical indicator, see above);
 *     the polytopal eigenproblem (eq. poincare_cp) gives a lower estimate, not a
 *     certified constant, per the paper's own caveat -- a hook, not used;
 *   - K^{-1/2} in the star norm uses the diagonal of K; c_KK is the true
 *     smallest eigenvalue;
 *   - the connection-face centroid is approximated by the cell-centre midpoint;
 *   - NNC connections contribute to div W_a (the FV balance) but are excluded
 *     from the Pi0_K moment sum -- they are not assigned an artificial face
 *     normal/centroid, per the paper's flux-reconstruction and corner-point-grid
 *     remarks ("An NNC is treated as an equal-and-opposite nonlocal cell
 *     source... it is not assigned an artificial face normal or centroid").
 */
#ifndef OPM_APOSTERIORI_SPATIAL_TEMPORAL_ESTIMATOR_HPP
#define OPM_APOSTERIORI_SPATIAL_TEMPORAL_ESTIMATOR_HPP

#include <dune/common/fmatrix.hh>
#include <dune/common/fmatrixev.hh>
#include <dune/common/fvector.hh>
#include <dune/grid/common/gridenums.hh>
#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/common/partitionset.hh>
#include <dune/grid/common/rangegenerators.hh>

#include <opm/input/eclipse/EclipseState/Grid/FaceDir.hpp>

#include <opm/material/common/MathToolbox.hpp>
#include <opm/material/fluidstates/SaturationOverlayFluidState.hpp>

#include <opm/models/discretization/common/fvbaseproperties.hh>
#include <opm/models/utils/basicproperties.hh>
#include <opm/models/utils/propertysystem.hh>
#include <opm/models/parallel/threadmanager.hpp>

#include <opm/grid/utility/ElementChunks.hpp>

#include <opm/simulators/flow/APosterioriReconstruction.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Opm {

/*!
 * \brief Evaluates eta_sp and eta_time on a converged fully implicit step.
 */
template<class TypeTag>
class APosterioriSpatialTemporalEstimator
{
    using Scalar          = GetPropType<TypeTag, Properties::Scalar>;
    using Simulator       = GetPropType<TypeTag, Properties::Simulator>;
    using GridView        = GetPropType<TypeTag, Properties::GridView>;
    using ElementMapper   = GetPropType<TypeTag, Properties::ElementMapper>;
    using VertexMapper    = GetPropType<TypeTag, Properties::VertexMapper>;
    using ElementContext  = GetPropType<TypeTag, Properties::ElementContext>;
    using IntensiveQuantities = GetPropType<TypeTag, Properties::IntensiveQuantities>;
    using FluidSystem     = GetPropType<TypeTag, Properties::FluidSystem>;
    using Indices         = GetPropType<TypeTag, Properties::Indices>;
    using RateVector      = GetPropType<TypeTag, Properties::RateVector>;
    using MaterialLaw     = GetPropType<TypeTag, Properties::MaterialLaw>;
    using FluidState      = typename IntensiveQuantities::FluidState;
    using Evaluation      = GetPropType<TypeTag, Properties::Evaluation>;
    using LocalResidual   = GetPropType<TypeTag, Properties::LocalResidual>;
    using BVector         = GetPropType<TypeTag, Properties::GlobalEqVector>;

    using Element   = typename GridView::template Codim<0>::Entity;

    static constexpr int dim = GridView::dimensionworld;
    static constexpr unsigned numComponents = FluidSystem::numComponents;
    static constexpr unsigned conti0EqIdx = Indices::conti0EqIdx;
    static constexpr unsigned numEq = Indices::numEq;

    using DimVector = Dune::FieldVector<Scalar, dim>;
    using CompFlux  = std::array<DimVector, numComponents>;
    using CompFlux0 = std::array<Scalar, numComponents>;      //!< per-component scalar (e.g. one connection's flux)
    using CompEval  = std::array<Evaluation, numComponents>;  //!< per-component AD value+local-Jacobian

    //! Frozen per-cell geometry (built once, or whenever the grid / wells change).
    struct CellGeom
    {
        Scalar    volume    {0};
        DimVector center     = DimVector(Scalar{0});
        Scalar    hK        {0};   //!< cell diameter  h_K
        Scalar    DK        {0};   //!< bound of max_{x in K} d_Lambda(x)
        Scalar    weightPow {1};   //!< D_K^{l/2}: bound of the near-well weight on K
        DimVector permDiag   = DimVector(Scalar{0});
        Dune::FieldMatrix<Scalar, dim, dim> permTensor {};  //!< full K|_K, for the MVEM matrix
        Scalar    cKK       {0};   //!< smallest eigenvalue of K|_K
        Scalar    phiRef    {1};   //!< reference porosity (eq. est_NA's Phiref factor)
        bool      interior  {false};
        std::vector<int>       vtx;      //!< global vertex indices of K
        std::vector<DimVector> vtxOff;   //!< x_a - x_K per vertex
    };

public:
    //! How grad p_hat entering u_alpha is reconstructed.
    enum class PressureRecon
    {
        PatchAverageLift,  //!< paper: grad of the vertex patch-average lift (eq. eq:averaging)
        ConnectionLS       //!< transmissibility-weighted LS of the raw connection pressure drops
    };

    explicit APosterioriSpatialTemporalEstimator(Simulator& simulator)
        : simulator_(simulator)
    {}

    void setPressureReconstruction(PressureRecon r) { pressureRecon_ = r; }
    PressureRecon pressureReconstruction() const { return pressureRecon_; }

    //! u_a should mimic the continuous flux -Kb: evaluate lambda_beta at the
    //! lifted saturation s_hat(x_K) (via the real MaterialLaw) rather than the
    //! FV cell state.  On by default; only used with PressureRecon::PatchAverageLift
    //! (where a saturation lift is available).  One extra relativePermeabilities
    //! call per cell per compute() -- turn off for cheaper, less rigorous evals.
    void setUseLiftedRelperm(bool on) { useLiftedRelperm_ = on; }
    bool useLiftedRelperm() const { return useLiftedRelperm_; }

    //! Restore the FV cell mean exactly in the point value used for the lifted
    //! saturation / pressure (paper eq. eq:averaging_bubble, point-value form
    //! only -- no sub-tessellation).  On by default; cheap (one scalar shift).
    void setUseBubbleCorrection(bool on) { useBubbleCorrection_ = on; }
    bool useBubbleCorrection() const { return useBubbleCorrection_; }

    //! l in the weighted space; the near-well weight scales like d_Lambda^l,
    //! l in (d-2, 2).  l = 0 (default) reproduces the plain energy norm.
    void setWeightExponent(Scalar ell) { ell_ = ell; geomValid_ = false; }

    //! epsilon > 0, the Neumann-scaling parameter of eq. eq:eps_norm; used in
    //! the accumulation-defect (NA) term of eta_lin.  Paper recommendation: 1.
    void setEpsilon(Scalar eps) { epsilon_ = eps; mvemMtotCache_.clear(); mvemCacheBuilt_ = false; }

    //! When false (default), ALL the *,K-energy-norm terms are evaluated on
    //! the same footing: eta_D and eta_lin's flux term via the full mimetic
    //! matrix z^T M_K z (full tensor K, stability term), eta_time and the
    //! eta_lin iterate-diff proxy via the full-tensor P0 form
    //! D_K^l |K| (v.K^{-1}v). When true, all of them drop to the cheap
    //! approximation: eta_D/eta_lin-flux to the T1-only Pi0 moment (diagonal
    //! K, no stability), eta_time/proxy to the diagonal |K_diag^{-1/2} v|.
    //! The eta_lin accumulation-defect term follows the paper's own
    //! c_KK^{-1/2} formula in both modes (it is a scalar L2 norm, not a flux
    //! energy norm -- no mimetic upgrade applies).
    void setCheapNorms(bool on) { cheapNorms_ = on; mvemMtotCache_.clear(); mvemCacheBuilt_ = false; }

    //! Compressed indices of the cells carrying a well connection; used to bound
    //! the near-well weight d_Lambda(x_K) <= D_K by  dist(x_K, wells) + h_K.
    //! Must be sorted ascending.
    void setWellCells(std::vector<int> cells)
    {
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        if (cells == wellCells_ && geomValid_)
            return;   // identical active-well set and geometry current: nothing to do
        wellCells_ = std::move(cells);
        // The near-well set changed (or this is the first call). The *static*
        // geometry (volumes, centroids, vertices, permeability tensor/eigenvalue,
        // h_K, M_K) does not depend on it -- only the D_K^{l/2} weight does. If
        // the static cache is already built and sized to the current grid, just
        // refresh the weights; otherwise force a full rebuild.
        if (geomValid_ && geom_.size() == simulator_.model().numGridDof())
            updateWellWeights_();
        else
            geomValid_ = false;
    }

    //! (Re)build the frozen geometry.
    void updateGeometry()
    {
        const auto& gridView = simulator_.gridView();
        const auto& model    = simulator_.model();
        const auto& problem  = simulator_.problem();
        ElementMapper mapper(gridView, Dune::mcmgElementLayout());
        const auto& vertexMapper = model.vertexMapper();

        const std::size_t nc = model.numGridDof();
        geom_.assign(nc, CellGeom{});

        for (const auto& elem : elements(gridView)) {
            const unsigned i = mapper.index(elem);
            auto& g = geom_[i];
            const auto igeom = elem.geometry();
            g.interior = (elem.partitionType() == Dune::InteriorEntity);
            g.volume = model.dofTotalVolume(i);
            const auto cc = igeom.center();
            for (int d = 0; d < dim; ++d)
                g.center[d] = cc[d];

            const int nc2 = igeom.corners();
            g.vtx.resize(nc2);
            g.vtxOff.resize(nc2);
            for (int k = 0; k < nc2; ++k) {
                g.vtx[k] = static_cast<int>(vertexMapper.subIndex(elem, k, dim));
                const auto xa = igeom.corner(k);
                for (int d = 0; d < dim; ++d)
                    g.vtxOff[k][d] = xa[d] - g.center[d];
            }

            Scalar h2 = 0.0;
            const int nCorners = igeom.corners();
            for (int a = 0; a < nCorners; ++a)
                for (int b = a + 1; b < nCorners; ++b) {
                    auto diff = igeom.corner(a);
                    diff -= igeom.corner(b);
                    h2 = std::max(h2, static_cast<Scalar>(diff.two_norm2()));
                }
            g.hK = std::sqrt(h2);

            const auto& K = problem.intrinsicPermeability(i);
            for (int d = 0; d < dim; ++d)
                g.permDiag[d] = K[d][d];
            g.permTensor = K;
            g.cKK = smallestEigenvalue_(K);
            g.phiRef = problem.referencePorosity(i, /*timeIdx=*/0);
        }

        updateWellWeights_();

        // History (havePrev_/prevU_, havePrevIter_/prevIterU_) must survive a
        // geometry rebuild: updateGeometry() now runs only on a genuine first
        // build or an actual cell-count change (setWellCells() refreshes just
        // the D_K^{l/2} weights in place -- see updateWellWeights_()). Still,
        // wiping the cross-timestep/iterate history whenever it does run would
        // make eta_time unavailable, so only reset the history buffers when the
        // cell count itself changes.
        if (prevU_.size() != nc) {
            prevU_.assign(nc, CompFlux{});
            havePrev_ = false;
        }
        if (prevIterU_.size() != nc) {
            prevIterU_.assign(nc, CompFlux{});
            havePrevIter_ = false;
        }
        // The MVEM matrix M_K depends only on cell geometry, permeability and
        // epsilon -- NOT on the near-well weight (applied outside M_K) or the
        // Newton iterate. updateGeometry() runs every iteration (setWellCells
        // invalidates geomValid_), so only drop the M_K cache when the cell
        // count actually changes; setEpsilon()/setCheapNorms() drop it too.
        if (mvemMtotCache_.size() != nc) {
            mvemMtotCache_.assign(nc, {});
            mvemCacheBuilt_ = false;
        }
        geomValid_ = true;
    }

    //! Refresh only the near-well weight D_K^{l/2} from the current wellCells_,
    //! reusing the cached static geometry (centres, h_K). Cheap: one pass over
    //! the grid, and a complete no-op when l = 0 (weightPow == 1 everywhere).
    void updateWellWeights_()
    {
        if (ell_ == Scalar{0}) {
            for (auto& g : geom_) {
                g.DK = g.hK;
                g.weightPow = Scalar{1};
            }
            return;
        }
        std::vector<DimVector> wellCenters;
        wellCenters.reserve(wellCells_.size());
        for (std::size_t i = 0; i < geom_.size(); ++i)
            if (std::binary_search(wellCells_.begin(), wellCells_.end(),
                                   static_cast<int>(i)))
                wellCenters.push_back(geom_[i].center);
        for (auto& g : geom_) {
            Scalar dLambda = 0.0;
            if (!wellCenters.empty()) {
                dLambda = std::numeric_limits<Scalar>::max();
                for (const auto& w : wellCenters) {
                    auto diff = g.center;
                    diff -= w;
                    dLambda = std::min(dLambda, static_cast<Scalar>(diff.two_norm()));
                }
            }
            // max_{x in K} d_Lambda(x) <= d_Lambda(x_K) + h_K  (triangle ineq.)
            g.DK = dLambda + g.hK;
            g.weightPow = std::pow(std::max(g.DK, Scalar{0}), ell_ / Scalar{2});
        }
    }

    /*!
     * \brief Capture the Newton-linearized flux prediction F^{k,n}_{alpha,ic}
     *        (eq. Newton_it_flux) and the linearized accumulation increment
     *        L^{k,n}_{alpha,K} (eq. lin_fv_balance), for the two terms of
     *        eta_lin (eq. Disc_estimator_eq_lin_alpha).
     *
     * MUST be called after the linear solve for this Newton step but BEFORE
     * the primary variables are updated (i.e. while model.intensiveQuantities()
     * still reflects chi^{k-1,n}): the whole point is to capture the exact
     * local flux/accumulation Jacobians OPM's own assembly used to build the
     * linear system, at the state that Jacobian was built at, then dot them
     * with the solved (and, if enabled, oscillation-relaxed) Newton increment.
     *
     * \param dx  the solved increment, in OPM's own sign convention:
     *            chi^{k,n} = chi^{k-1,n} - dx (blackoilnewtonmethod.hpp
     *            applies nextValue = currentValue - delta), i.e. the paper's
     *            delta_chi^{k,n} = -dx.  Whatever additional per-variable
     *            safety clamping OPM's updatePrimaryVariables_ performs on
     *            top of dx (dPMax, dSMax, ...) is NOT separately captured
     *            here -- a documented, narrow approximation (only relevant in
     *            iterations where those clamps actually engage).
     *
     * KNOWN LIMITATIONS (not yet fixed):
     *  - Pass 2's mirror-connection lookup matches (i,j) to (j,i) by
     *    neighbour-cell occurrence order within each row, not by a genuine
     *    face/NNC id. It is exact for the common case of a single connection
     *    between a cell pair and relies on OPM enumerating both directions of
     *    a shared connection list in the same relative order otherwise; not
     *    a proven invariant of the NeighborInfo table.
     *  - Parallel (MPI) runs: Pass 1 only populates ownFlux_/ownAccumVal_ for
     *    Dune::Partitions::interior elements. A neighbour j that is a
     *    ghost/overlap (non-owned) cell on this rank has no valid ownFlux_[j]
     *    entry, so Pass 2 cannot use j's own derivatives for such a
     *    connection; it falls back to the one-sided (i-only) contribution,
     *    silently omitting the exterior-side Jacobian term across partition
     *    boundaries. This mirrors the already-documented serial-exact-only
     *    caveat on the vertex-patch reconstruction.
     */
    //! Algebraic estimator eta_alg, eq. est_alg: per cell/component
    //!   eta_alg,a,K = sqrt(tau) eps^{-1/2} c_KK^{-1/2} h_K D_K^{l/2}
    //!                 |R_lin,a,K| / sqrt(V_K),
    //! with R_lin the algebraic residual (A dx - b) of the Newton linear
    //! solve, in reference-mass units (same scaling as the other terms).
    //! The first term of eta_lin vanishes at Newton convergence; eta_alg
    //! vanishes at linear-solve convergence. Result in etaAlgebraic().
    //! Returns eta_alg for the given residual vector and also stores it in
    //! etaAlgebraic(). Pass the un-reduced residual b (rAlg = b, i.e. x = 0)
    //! before a solve to get the initial algebraic-error scale eta_alg^{(0)}.
    Scalar computeAlgebraicEstimator(const BVector& rAlg, Scalar dt)
    {
        if (!geomValid_)
            updateGeometry();
        const auto& gridView = simulator_.gridView();
        auto& model          = simulator_.model();
        ElementMapper mapper(gridView, Dune::mcmgElementLayout());
        const int watPh = FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)
            ? FluidSystem::waterPhaseIdx : -1;
        const int oilPh = FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)
            ? FluidSystem::oilPhaseIdx : -1;
        const int gasPh = FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)
            ? FluidSystem::gasPhaseIdx : -1;
        const Scalar tau = std::max(dt, Scalar{0});
        const Scalar epsInv = (epsilon_ > Scalar{0}) ? Scalar{1} / std::sqrt(epsilon_) : Scalar{0};
        Scalar sum2 = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum2)
#endif
        for (const auto& chunk : ElementChunks(gridView, Dune::Partitions::interior,
                                               ThreadManager::maxThreads())) {
          for (const auto& elem : chunk) {
            const unsigned i = mapper.index(elem);
            if (i >= geom_.size()) continue;
            const auto& g = geom_[i];
            if (!(g.volume > Scalar{0}) || !(g.cKK > Scalar{0})) continue;
            const auto& iq = model.intensiveQuantities(i, /*timeIdx=*/0);
            const auto rhoRef = componentRefDensities_(watPh, oilPh, gasPh, iq.pvtRegionIndex());
            const Scalar pref = std::sqrt(tau) * epsInv / std::sqrt(g.cKK)
                * g.hK * g.weightPow / std::sqrt(g.volume);
            Scalar cAlg2 = 0;
            for (unsigned c = 0; c < numComponents; ++c) {
                const Scalar rc = rhoRef[c] * getValue(rAlg[i][conti0EqIdx + c]);
                const Scalar e = pref * std::abs(rc);
                sum2 += e * e; cAlg2 += e * e;
            }
            if (i < cellEta_.size())
                cellEta_[i][3] = std::sqrt(std::max(cAlg2, Scalar{0}));
          }
        }
        sum2 = gridView.comm().sum(sum2);
        etaAlg_ = std::sqrt(std::max(sum2, Scalar{0}));
        return etaAlg_;
    }

    Scalar etaAlgebraic() const { return etaAlg_; }

    //! Write the per-cell estimator distribution to a CSV for 3-D plotting:
    //! columns cell, x, y, z, eta_sp(mim), eta_time, eta_lin, eta_alg.
    void dumpCellEstimators(const std::string& path) const
    {
        std::ofstream os(path);
        if (!os) return;
        os << "cell,x,y,z,eta_sp,eta_time,eta_lin,eta_alg\n";
        const std::size_t n = std::min(cellEta_.size(), geom_.size());
        for (std::size_t i = 0; i < n; ++i) {
            const auto& g = geom_[i];
            os << i;
            for (int d = 0; d < 3; ++d)
                os << ',' << (d < dim ? g.center[std::min(d, dim - 1)] : Scalar{0});
            os << ',' << cellEta_[i][0] << ',' << cellEta_[i][1] << ','
               << cellEta_[i][2] << ',' << cellEta_[i][3] << '\n';
        }
    }

    //! Algorithm 6.1 spatial marking. Per-cell mark m_K in
    //!   {+1 refine  : eta_sp,K >= zetaRef  * max_K eta_sp,K,
    //!     0 keep,
    //!    -1 derefine: eta_sp,K <= zetaDeref * max_K eta_sp,K}.
    //! Uses the mimetic per-cell spatial estimator cellEta_[.][0].
    //! Well cells are excluded entirely -- neither marked nor counted towards
    //! max_K -- because the spatial estimator over-reads near wells (near-well
    //! weight, omitted well-source Taylor defect), so h-refining there chases
    //! an indicator artefact, not a real discretisation error.
    //! Returns {nRefine, nDerefine}; fills \p marks when non-null.
    std::pair<int, int> refinementMarks(Scalar zetaRef, Scalar zetaDeref,
                                        std::vector<int>* marks = nullptr) const
    {
        const std::size_t nc = cellEta_.size();
        if (marks) marks->assign(nc, 0);
        auto isWellCell = [this](std::size_t i) {
            return std::binary_search(wellCells_.begin(), wellCells_.end(),
                                      static_cast<int>(i));
        };
        double emax = 0.0;
        for (std::size_t i = 0; i < nc; ++i) {
            if (isWellCell(i)) continue;
            emax = std::max(emax, cellEta_[i][0]);
        }
        if (!(emax > 0.0)) return {0, 0};

        // Bulk (cumulative Dörfler) marking when env OPM_APOST_BULK_THETA is set:
        // REFINE the smallest cell set whose sum of eta_sp,K^2 reaches
        // theta * sum_K eta_sp,K^2. Falls back to the max-threshold rule below
        // when theta is unset/invalid. Derefine still uses zetaDeref*max.
        double theta = -1.0;
        if (const char* s = std::getenv("OPM_APOST_BULK_THETA")) theta = std::atof(s);
        if (theta > 0.0 && theta < 1.0) {
            double total = 0.0;
            std::vector<std::pair<double, std::size_t>> ord;
            ord.reserve(nc);
            for (std::size_t i = 0; i < nc; ++i) {
                if (isWellCell(i)) continue;
                const double e2 = cellEta_[i][0] * cellEta_[i][0];
                total += e2;
                ord.emplace_back(e2, i);
            }
            std::sort(ord.begin(), ord.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            const double tD = static_cast<double>(zetaDeref) * emax;
            double acc = 0.0;
            int nR = 0, nD = 0;
            const double target = theta * total;
            for (const auto& [e2, i] : ord) {
                if (acc < target) { acc += e2; if (marks) (*marks)[i] = 1; ++nR; }
                else if (std::sqrt(e2) <= tD) { if (marks) (*marks)[i] = -1; ++nD; }
            }
            return {nR, nD};
        }

        const double tR = static_cast<double>(zetaRef) * emax;
        const double tD = static_cast<double>(zetaDeref) * emax;
        int nR = 0, nD = 0;
        for (std::size_t i = 0; i < nc; ++i) {
            if (isWellCell(i)) continue;
            const double e = cellEta_[i][0];
            int m = 0;
            if (e >= tR) { m = 1; ++nR; }
            else if (e <= tD) { m = -1; ++nD; }
            if (marks) (*marks)[i] = m;
        }
        return {nR, nD};
    }

    //! CSV of the Algorithm 6.1 spatial marks: cell, x, y, z, eta_sp_K, mark.
    void dumpRefinementMarks(const std::string& path, Scalar zetaRef, Scalar zetaDeref) const
    {
        std::vector<int> marks;
        refinementMarks(zetaRef, zetaDeref, &marks);
        std::ofstream os(path);
        if (!os) return;
        os << "cell,x,y,z,eta_sp_K,mark\n";
        const std::size_t n = std::min(cellEta_.size(), geom_.size());
        for (std::size_t i = 0; i < n; ++i) {
            const auto& g = geom_[i];
            os << i;
            for (int d = 0; d < 3; ++d)
                os << ',' << (d < dim ? g.center[std::min(d, dim - 1)] : Scalar{0});
            os << ',' << cellEta_[i][0] << ',' << marks[i] << '\n';
        }
    }

    //! Algorithm 6.1 h-refinement request. Marked cells are padded by a
    //! coarse-cell halo and overlapping/touching patches are merged before
    //! emitting CARFIN-style boxes. A halo keeps the coarse/fine transition
    //! away from the high-error cell; one-cell LGR islands next to wells can
    //! perturb well rates more than the local discretization error they remove.
    //! OPM_APOST_REFINE_HALO overrides the default one-cell halo (zero restores
    //! the old one-box-per-mark behavior). At most \p maxBoxes seed cells are
    //! retained before padding; OPM_APOST_MAX_BOXES overrides that limit.
    //!   "I1 I2 J1 J2 K1 K2 NX NY NZ"  per box (NX/NY/NZ are box totals)
    std::string refinementBoxSpec(Scalar zetaRef, int sub = 2, int maxBoxes = 128) const
    {
        if (const char* s = std::getenv("OPM_APOST_MAX_BOXES")) {
            const int v = std::atoi(s);
            if (v > 0) maxBoxes = v;
        }
        std::vector<int> marks;
        const auto counts = refinementMarks(zetaRef, Scalar{0}, &marks);
        if (counts.first == 0) return {};

        // Marked cells, largest eta_sp,K first.
        std::vector<std::size_t> flagged;
        for (std::size_t c = 0; c < marks.size(); ++c)
            if (marks[c] == 1) flagged.push_back(c);
        std::sort(flagged.begin(), flagged.end(),
                  [this](std::size_t a, std::size_t b)
                  { return cellEta_[a][0] > cellEta_[b][0]; });
        if (static_cast<int>(flagged.size()) > maxBoxes)
            flagged.resize(maxBoxes);

        const auto& mapper = simulator_.vanguard().cartesianIndexMapper();
        const auto& dims = mapper.cartesianDimensions();

        int halo = 1;
        if (const char* value = std::getenv("OPM_APOST_REFINE_HALO")) {
            halo = std::max(0, std::atoi(value));
        }

        struct Box {
            std::array<int, 3> lo;
            std::array<int, 3> hi;
        };
        std::vector<std::array<int, 3>> wellIJK;
        wellIJK.reserve(wellCells_.size());
        for (int cell : wellCells_) {
            std::array<int, 3> ijk{0, 0, 0};
            mapper.cartesianCoordinate(cell, ijk);
            wellIJK.push_back(ijk);
        }
        const auto containsWell = [&wellIJK](const Box& box) {
            return std::ranges::any_of(wellIJK, [&box](const auto& ijk) {
                return ijk[0] >= box.lo[0] && ijk[0] <= box.hi[0]
                    && ijk[1] >= box.lo[1] && ijk[1] <= box.hi[1]
                    && ijk[2] >= box.lo[2] && ijk[2] <= box.hi[2];
            });
        };

        std::vector<Box> boxes;
        boxes.reserve(flagged.size());
        for (std::size_t c : flagged) {
            std::array<int, 3> ijk{0, 0, 0};
            mapper.cartesianCoordinate(static_cast<int>(c), ijk);
            Box box;
            for (int d = 0; d < 3; ++d) {
                box.lo[d] = std::max(0, ijk[d] - halo);
                box.hi[d] = std::min(dims[d] - 1, ijk[d] + halo);
            }
            // Programmatic LGR wells cannot mix GLOBAL and LGR completions.
            // Keep automatically generated patch boundaries away from wells;
            // a user may still request a coherent well-containing block
            // explicitly through --adaptive-lgr.
            if (!containsWell(box)) {
                boxes.push_back(box);
            }
        }

        // Merge boxes whose closures touch, unless the rectangular union would
        // engulf a well. Overlapping boxes cannot be emitted separately; in
        // that rare case retain the higher-priority box and discard the other.
        bool changed = true;
        while (changed) {
            changed = false;
            for (std::size_t i = 0; i < boxes.size(); ++i) {
                for (std::size_t j = i + 1; j < boxes.size(); ++j) {
                    bool touch = true;
                    bool overlap = true;
                    for (int d = 0; d < 3; ++d) {
                        touch = touch
                            && boxes[i].lo[d] <= boxes[j].hi[d] + 1
                            && boxes[j].lo[d] <= boxes[i].hi[d] + 1;
                        overlap = overlap
                            && boxes[i].lo[d] <= boxes[j].hi[d]
                            && boxes[j].lo[d] <= boxes[i].hi[d];
                    }
                    if (!touch) {
                        continue;
                    }
                    Box joined = boxes[i];
                    for (int d = 0; d < 3; ++d) {
                        joined.lo[d] = std::min(joined.lo[d], boxes[j].lo[d]);
                        joined.hi[d] = std::max(joined.hi[d], boxes[j].hi[d]);
                    }
                    if (containsWell(joined)) {
                        if (overlap) {
                            boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(j));
                            changed = true;
                            break;
                        }
                        continue;
                    }
                    boxes[i] = joined;
                    boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(j));
                    changed = true;
                    break;
                }
                if (changed) {
                    break;
                }
            }
        }

        std::string spec;
        for (const auto& box : boxes) {
            if (!spec.empty()) spec += " ; ";
            for (int d = 0; d < 3; ++d) {
                if (d > 0) spec += ' ';
                spec += std::to_string(box.lo[d] + 1) + ' '
                      + std::to_string(box.hi[d] + 1);
            }
            for (int d = 0; d < 3; ++d) {
                spec += ' ' + std::to_string((box.hi[d] - box.lo[d] + 1) * sub);
            }
        }
        return spec;
    }

    // ---------------------------------------------------------------------
    //  One-rebuild h-adaptivity path: time-accumulated Dorfler marking +
    //  schedule-wide protected-well mask + exact mask->box packing.
    //  Kept SEPARATE from refinementMarks()/refinementBoxSpec() above so the
    //  existing per-step indicator reports are unchanged.
    // ---------------------------------------------------------------------

    //! Schedule-wide protected well-completion cells, as GLOBAL Cartesian
    //! indices. A generated refinement box will never contain one of these.
    //! Call once (driver, before the estimator-driven grid generator runs);
    //! the set must be stable, not rebuilt from the active well set.
    void setProtectedRefinementCells(std::vector<int> cartesianIdx)
    {
        std::sort(cartesianIdx.begin(), cartesianIdx.end());
        cartesianIdx.erase(std::unique(cartesianIdx.begin(), cartesianIdx.end()),
                           cartesianIdx.end());
        protectedCartesian_ = std::move(cartesianIdx);
    }

    const std::vector<int>& protectedRefinementCells() const { return protectedCartesian_; }

    //! Accumulate E_K^2 += eta_sp,K^2 for every interior cell. Call once per
    //! ACCEPTED (converged) time step, after compute(dt, commitHistory=true).
    void accumulateSpatialEnergy()
    {
        const std::size_t nc = cellEta_.size();
        if (accumulatedSpatialEnergy_.size() != nc)
            accumulatedSpatialEnergy_.assign(nc, 0.0);
        for (std::size_t i = 0; i < nc; ++i)
            accumulatedSpatialEnergy_[i] += cellEta_[i][0] * cellEta_[i][0];
    }

    void resetAccumulatedSpatialEnergy()
    { std::fill(accumulatedSpatialEnergy_.begin(), accumulatedSpatialEnergy_.end(), 0.0); }

    //! Commit the latest converged iterate as temporal history. This is kept
    //! separate from compute() because Newton convergence may still be rejected
    //! by the outer adaptive timestep acceptance test.
    void commitTemporalHistory()
    {
        if (!havePrevIter_)
            return;
        prevU_.swap(prevIterU_);
        havePrev_ = true;
    }

    bool accumulatedSpatialEnergyAvailable() const
    {
        return !accumulatedSpatialEnergy_.empty()
            && std::any_of(accumulatedSpatialEnergy_.begin(),
                           accumulatedSpatialEnergy_.end(),
                           [](double v) { return v > 0.0; });
    }

    //! {nCoarseCellsInFinalMask, nSeedCellsAfterProtectionBeforeHalo}. Purely for
    //! logging -- refinementBoxSpecAccumulated() does the real work.
    std::pair<int, int> refinementMarksAccumulated(double theta) const
    {
        std::vector<char> mask;
        std::array<int, 3> dims{1, 1, 1};
        int nSeed = 0;
        buildRefineMask_(theta, mask, dims, &nSeed);
        int nMask = 0;
        for (char c : mask) if (c) ++nMask;
        return {nMask, nSeed};
    }

    //! Fraction of the TOTAL accumulated spatial energy captured by the most
    //! recent buildRefineMask_() selection (protected cells excluded, before
    //! halo). Diagnostic only.
    double lastRetainedEnergyFraction() const { return lastRetainedEnergyFraction_; }

    //! CARFIN-style box spec covering the time-accumulated Dorfler-marked
    //! cells, with the schedule-wide protected wells carved out. Boxes are
    //! disjoint, their union equals the (post-halo, post-protection) mask
    //! exactly, and no box contains a protected completion cell. Returns "" if
    //! nothing is marked. \p sub 0 => read OPM_APOST_REFINE_FACTOR (default 2).
    std::string refinementBoxSpecAccumulated(double theta, int sub = 0) const
    {
        if (sub <= 0) {
            sub = 2;
            if (const char* s = std::getenv("OPM_APOST_REFINE_FACTOR")) {
                const int v = std::atoi(s);
                if (v > 1) sub = v;
            }
        }
        std::vector<char> mask;
        std::array<int, 3> dims{1, 1, 1};
        buildRefineMask_(theta, mask, dims, nullptr);
        if (std::none_of(mask.begin(), mask.end(), [](char c) { return c != 0; }))
            return {};

        const auto boxes = packMaskIntoBoxes_(mask, dims);

        // ---- invariants (abort loudly rather than emit a bad request) ----
        std::vector<char> cover(mask.size(), 0);
        for (const auto& b : boxes) {
            for (int k = b.lo[2]; k <= b.hi[2]; ++k)
            for (int j = b.lo[1]; j <= b.hi[1]; ++j)
            for (int i = b.lo[0]; i <= b.hi[0]; ++i) {
                const std::size_t c = cartIndex_(i, j, k, dims);
                if (cover[c])
                    throw std::logic_error("APosterioriEstimator: refinement boxes overlap");
                cover[c] = 1;
                if (isProtectedCart_(static_cast<int>(c)))
                    throw std::logic_error("APosterioriEstimator: refinement box contains a "
                                           "protected well-completion cell");
            }
        }
        for (std::size_t c = 0; c < mask.size(); ++c)
            if (static_cast<bool>(cover[c]) != static_cast<bool>(mask[c]))
                throw std::logic_error("APosterioriEstimator: box union != refinement mask");

        std::string spec;
        for (const auto& b : boxes) {
            if (!spec.empty()) spec += " ; ";
            for (int d = 0; d < 3; ++d) {
                if (d > 0) spec += ' ';
                spec += std::to_string(b.lo[d] + 1) + ' ' + std::to_string(b.hi[d] + 1);
            }
            for (int d = 0; d < 3; ++d)
                spec += ' ' + std::to_string((b.hi[d] - b.lo[d] + 1) * sub);
        }
        return spec;
    }

    void recordLinearizationDefect(Scalar dt, const BVector& dx)
    {
        if (!geomValid_)
            updateGeometry();

        const auto& gridView = simulator_.gridView();
        auto& model          = simulator_.model();
        const auto& nbInfo   = model.linearizer().getNeighborInfo();
        const auto& moduleParams = simulator_.problem().moduleParams();
        ElementMapper mapper(gridView, Dune::mcmgElementLayout());

        const std::size_t nc = geom_.size();
        ownFlux_.assign(nc, {});
        ownAccumVal_.assign(nc, CompFlux0{});
        ownAccumDeriv_.assign(nc, {});

        // Pass 1: for every cell K, the flux (and accumulation) Evaluation
        // with derivatives w.r.t. K's own local primary variables only (the
        // same "one-sided" AD OPM's own TPFA residual uses -- the exterior
        // side is intentionally value-only, matching calculateFluxes_).
        // Per-cell independent: writes ownFlux_[i]/ownAccumVal_[i]/ownAccumDeriv_[i] only.
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (const auto& chunk : ElementChunks(gridView, Dune::Partitions::interior,
                                               ThreadManager::maxThreads())) {
          for (const auto& elem : chunk) {
            const unsigned i = mapper.index(elem);
            const auto row = nbInfo[static_cast<int>(i)];
            const std::size_t nf = static_cast<std::size_t>(row.end() - row.begin());
            ownFlux_[i].assign(nf, {});

            const auto& intQuantsIn = model.intensiveQuantities(i, /*timeIdx=*/0);

            // Same reference-mass scaling as compute()'s main flux loop --
            // see componentRefDensities_'s doc comment. Evaluated per source
            // cell i's own PVT region, consistent with the flux itself being
            // a one-sided (donor-cell) AD quantity already.
            const int watPh = FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)
                ? FluidSystem::waterPhaseIdx : -1;
            const int oilPh = FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)
                ? FluidSystem::oilPhaseIdx : -1;
            const int gasPh = FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)
                ? FluidSystem::gasPhaseIdx : -1;
            const std::array<Scalar, numComponents> rhoRef =
                componentRefDensities_(watPh, oilPh, gasPh, intQuantsIn.pvtRegionIndex());

            std::size_t f = 0;
            for (const auto& nb : row) {
                const auto& intQuantsEx = model.intensiveQuantities(nb.neighbor, /*timeIdx=*/0);
                RateVector flux(0.0), darcy(0.0);
                LocalResidual::computeFlux(flux, darcy, i, nb.neighbor,
                                           intQuantsIn, intQuantsEx,
                                           nb.res_nbinfo, moduleParams);
                for (unsigned c = 0; c < numComponents; ++c)
                    ownFlux_[i][f][c] = flux[conti0EqIdx + c] * Scalar(nb.res_nbinfo.faceArea) * rhoRef[c];
                ++f;
            }

            RateVector storage(0.0);
            LocalResidual::template computeStorage<Evaluation>(storage, intQuantsIn);
            for (unsigned c = 0; c < numComponents; ++c) {
                ownAccumVal_[i][c] = rhoRef[c] * getValue(storage[conti0EqIdx + c]);
                ownAccumDeriv_[i][c] = storage[conti0EqIdx + c] * rhoRef[c];
            }
          }
        }

        // Pass 2: assemble F = u(chi^{k-1,n}) + dudchi_K . delta_chi_K
        //                       + dudchi_j . delta_chi_j
        // via the mirror connection u_ji = -u_ij (skew-symmetry), and
        // L_K = dA/dchi_K . delta_chi_K, with delta_chi = -dx (sign flip,
        // see the docstring above).
        Flin_.assign(nc, {});
        Lval_.assign(nc, CompFlux0{});
        haveLin_ = true;

        // Per-cell independent: reads ownFlux_[i] and the mirror ownFlux_[j]
        // (all fully written by Pass 1 above), writes Flin_[i]/Lval_[i] only.
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (const auto& chunk : ElementChunks(gridView, Dune::Partitions::interior,
                                               ThreadManager::maxThreads())) {
          for (const auto& elem : chunk) {
            const unsigned i = mapper.index(elem);
            const auto row = nbInfo[static_cast<int>(i)];
            const std::size_t nf = ownFlux_[i].size();
            Flin_[i].assign(nf, {});

            std::size_t f = 0;
            for (const auto& nb : row) {
                const unsigned j = nb.neighbor;
                // Locate the mirror entry (j,i) in j's own row. Two cells can
                // share more than one connection (a fault face plus an NNC
                // covering the pinched-out remainder, say), so matching on
                // nbJ.neighbor==i alone is ambiguous -- pick the k-th (j,i)
                // entry where k is this connection's own occurrence index
                // among (i,j) pairs in i's row, relying on OPM building both
                // directions of a shared face/NNC list in the same relative
                // order (true for the geometric-face and NNC construction
                // paths; not formally guaranteed by the interface). This is
                // still a heuristic, not a face-id match -- flagged as a
                // known robustness gap for grids with many parallel (i,j)
                // connections.
                std::size_t occurrence = 0;
                for (std::size_t k = 0; k < f; ++k) {
                    if (row[k].neighbor == j)
                        ++occurrence;
                }
                const auto rowJ = nbInfo[static_cast<int>(j)];
                std::size_t g = 0;
                bool found = false;
                std::size_t seen = 0;
                for (const auto& nbJ : rowJ) {
                    if (nbJ.neighbor == i) {
                        if (seen == occurrence) { found = true; break; }
                        ++seen;
                    }
                    ++g;
                }
                // NOTE (parallel runs): j may be a non-owned (ghost/overlap)
                // cell.  ownFlux_[j] is only populated by Pass 1's element
                // loop over Dune::Partitions::interior, so a ghost j leaves
                // ownFlux_[j] empty/stale and this mirror lookup must not be
                // trusted across a partition boundary -- see the class-level
                // parallel caveat.
                if (found && (j >= ownFlux_.size() || g >= ownFlux_[j].size()))
                    found = false;
                for (unsigned c = 0; c < numComponents; ++c) {
                    Scalar F = getValue(ownFlux_[i][f][c]);
                    for (unsigned pv = 0; pv < numEq; ++pv)
                        F -= ownFlux_[i][f][c].derivative(pv) * dx[i][pv];
                    if (found) {
                        for (unsigned pv = 0; pv < numEq; ++pv)
                            F += ownFlux_[j][g][c].derivative(pv) * dx[j][pv];
                    }
                    Flin_[i][f][c] = F;
                }
                ++f;
            }

            for (unsigned c = 0; c < numComponents; ++c) {
                Scalar L = 0;
                for (unsigned pv = 0; pv < numEq; ++pv)
                    L -= ownAccumDeriv_[i][c].derivative(pv) * dx[i][pv];
                Lval_[i][c] = L;
            }
          }
        }
        (void) dt; // reserved: the tau^n prefactor is applied when eta_lin is assembled in compute()
    }

    /*!
     * \brief Compute eta_sp and eta_time for the current iterate.
     * \param dt             the time-step size tau^n.
     * \param commitHistory  store u_alpha as the "previous step" for the next
     *                       temporal estimate.  Pass false on intermediate
     *                       Newton iterations so eta_time keeps comparing
     *                       against the last *converged* step (this is what lets
     *                       eta_sp / eta_time be evaluated every iteration to
     *                       show the error-component plateau); pass true once
     *                       per converged step.
     */
    void compute(Scalar dt, bool commitHistory = true)
    {
        if (!geomValid_)
            updateGeometry();

        const auto& gridView = simulator_.gridView();
        auto& model          = simulator_.model();
        const auto& nbInfo   = model.linearizer().getNeighborInfo();
        ElementMapper mapper(gridView, Dune::mcmgElementLayout());

        // H1-conforming potential lift (eq. eq:averaging): vertex patch averages
        // of the phase pressures.  grad p_hat is then the gradient of the
        // mean-value-coordinate interpolant of these nodal values; the polytopal
        // bubble (eq. eq:averaging_bubble) has zero cell-mean gradient, so for
        // the piecewise-constant u_alpha it drops out and grad p_hat|_K is the
        // least-squares fit of the nodal values (mean-value coords reproduce
        // affine fields).
        computeVertexAverages_();

        const std::size_t nc = geom_.size();
        std::vector<CompFlux> curU(nc, CompFlux{});
        if (cellEta_.size() != nc)
            cellEta_.assign(nc, std::array<double, 4>{0, 0, 0, 0});

        const int watPh = FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)
            ? FluidSystem::waterPhaseIdx : -1;
        const int oilPh = FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)
            ? FluidSystem::oilPhaseIdx : -1;
        const int gasPh = FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)
            ? FluidSystem::gasPhaseIdx : -1;
        const auto comp = [](int ph) {
            return FluidSystem::canonicalToActiveCompIdx(FluidSystem::solventComponentIndex(ph));
        };

        constexpr Scalar invPi = Scalar{0.31830988618379067154}; // 1/pi = C_PW,K (convex)

        Scalar sumSp2 = 0.0;
        Scalar sumSpT1_2 = 0.0;  // T1-only companion sum -- see etaSpatialT1()'s doc comment
        Scalar sumSpMim2 = 0.0;  // mimetic (Corollary 3.13/3.14) companion sum -- see etaSpatialMimetic()
        Scalar sumTime2 = 0.0;
        Scalar sumLin2 = 0.0;

        // Per-cell independent: reads geom_/prevU_/prevIterU_/Flin_/ownAccumVal_
        // /Lval_ at [i] only, writes curU[i] only, and accumulates the five
        // RSS sums. Parallelised over element chunks; the FP reduction order is
        // thread-count dependent (perturbation ~1e-14, immaterial to the
        // adaptive decisions).
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumSp2,sumSpT1_2,sumSpMim2,sumTime2,sumLin2)
#endif
        for (const auto& chunk : ElementChunks(gridView, Dune::Partitions::interior,
                                               ThreadManager::maxThreads())) {
          for (const auto& elem : chunk) {
            const unsigned i = mapper.index(elem);
            const auto& g = geom_[i];
            if (g.volume <= Scalar{0})
                continue;

            const auto row = nbInfo[static_cast<int>(i)];
            const std::size_t nf = static_cast<std::size_t>(row.end() - row.begin());
            if (nf == 0)
                continue;

            const auto& iqIn = model.intensiveQuantities(i, /*timeIdx=*/0);
            const auto& fsIn = iqIn.fluidState();

            // Reference (surface) mass density per component -- see
            // componentRefDensities_'s doc comment. Applied at the source to
            // every flux/accumulation quantity below, so T1/T3/div W/eta_time
            // /eta_lin all end up on a common mass basis.
            const std::array<Scalar, numComponents> rhoRef =
                componentRefDensities_(watPh, oilPh, gasPh, iqIn.pvtRegionIndex());

            // A_alpha(chi^{k,n}), the post-update accumulation, for the
            // accumulation-defect (NA) term of eta_lin below.
            CompFlux0 Anew{};
            if (haveLin_) {
                RateVector storageNow(0.0);
                LocalResidual::template computeStorage<Evaluation>(storageNow, iqIn);
                for (unsigned c = 0; c < numComponents; ++c)
                    Anew[c] = rhoRef[c] * getValue(storageNow[conti0EqIdx + c]);
            }

            // per-connection data
            std::array<std::vector<Scalar>, numComponents> Fc; // integrated component fluxes W_a.n_c out of K
            for (auto& v : Fc)
                v.assign(nf, Scalar{0});
            std::vector<DimVector> dcFace(nf);                 // x_c - x_K (face centre ~ midpoint)
            std::vector<DimVector> dCC(nf);                    // x_j - x_K (cell centre)
            std::vector<Scalar>    wLS(nf);                    // connection transmissibilities
            std::vector<Scalar>    faceAreaVec(nf);            // A_f, for the mimetic defect energy below
            std::vector<char>      isGeomFace(nf, 0);          // geometric sub-face (not an NNC)
            std::array<std::vector<Scalar>, 3> pNb;            // neighbour phase pressures
            for (auto& v : pNb)
                v.assign(nf, Scalar{0});

            std::size_t f = 0;
            for (const auto& nb : row) {
                const unsigned j     = nb.neighbor;
                const Scalar   trans = nb.res_nbinfo.trans;
                const auto& iqEx = model.intensiveQuantities(j, /*timeIdx=*/0);
                const auto& fsEx = iqEx.fluidState();

                // W_a : the scheme's own TPFA component flux (eq.
                // Main_problem_model_flux) at the current potentials --
                // reused via OPM's real LocalResidual::computeFlux (the exact
                // function TpfaLinearizer's own assembly calls), so threshold
                // pressures, transmissibility multipliers, directional
                // mobility and rock-compaction corrections are all included,
                // not re-derived by hand.
                {
                    RateVector flux(0.0), darcy(0.0);
                    LocalResidual::computeFlux(flux, darcy, i, j, iqIn, iqEx,
                                               nb.res_nbinfo, simulator_.problem().moduleParams());
                    for (unsigned c = 0; c < numComponents; ++c)
                        Fc[c][f] = rhoRef[c] * getValue(flux[conti0EqIdx + c]) * nb.res_nbinfo.faceArea;
                }

                DimVector off = geom_[j].center;
                off -= g.center;
                dCC[f] = off;
                dcFace[f] = off;
                dcFace[f] *= Scalar{0.5};
                wLS[f] = std::max(trans, Scalar{0});
                faceAreaVec[f] = nb.res_nbinfo.faceArea;
                // A geometric sub-face carries a real axis direction; an NNC
                // (fault / pinch-out / explicit non-neighbour connection) does
                // not and is excluded from the Pi0_K centroid sum below -- it
                // still contributes to div W_a via Fc.  See the paper's remark
                // that an NNC "is not assigned an artificial face normal or
                // centroid".
                isGeomFace[f] = (nb.res_nbinfo.faceDir != FaceDir::DirEnum::Unknown) ? 1 : 0;
                if (watPh >= 0) pNb[0][f] = getValue(fsEx.pressure(watPh));
                if (oilPh >= 0) pNb[1][f] = getValue(fsEx.pressure(oilPh));
                if (gasPh >= 0) pNb[2][f] = getValue(fsEx.pressure(gasPh));
                ++f;
            }

            // grad p_hat_beta : two reconstructions, selectable for comparison.
            std::array<DimVector, 3> gradP{};
            bool hasLambdaHat = false;
            std::array<Scalar, 3> lambdaHat{};   // lambda_beta(s_hat), indices (water,oil,gas)
            auto meanOf = [](const std::vector<Scalar>& v) {
                Scalar s = 0;
                for (Scalar x : v) s += x;
                return v.empty() ? Scalar{0} : s / static_cast<Scalar>(v.size());
            };
            if (pressureRecon_ == PressureRecon::PatchAverageLift) {
                // eq. eq:averaging : gradient of the mean-value-coordinate
                // interpolant of the vertex patch averages (bubble has zero
                // cell-mean gradient).  Capillary gradients retained per phase.
                const std::size_t nv = g.vtx.size();
                std::array<std::vector<Scalar>, 3> pv;
                for (auto& v : pv)
                    v.assign(nv, Scalar{0});
                for (std::size_t k = 0; k < nv; ++k) {
                    const int a = g.vtx[k];
                    if (watPh >= 0) pv[0][k] = vtxP_[0][a];
                    if (oilPh >= 0) pv[1][k] = vtxP_[1][a];
                    if (gasPh >= 0) pv[2][k] = vtxP_[2][a];
                }
                if (watPh >= 0) gradP[0] = APosteriori::leastSquaresGradient<Scalar, dim>(
                    meanOf(pv[0]), pv[0], g.vtxOff);
                if (oilPh >= 0) gradP[1] = APosteriori::leastSquaresGradient<Scalar, dim>(
                    meanOf(pv[1]), pv[1], g.vtxOff);
                if (gasPh >= 0) gradP[2] = APosteriori::leastSquaresGradient<Scalar, dim>(
                    meanOf(pv[2]), pv[2], g.vtxOff);

                // lambda_beta(s_hat): point saturation lift at x_K from the same
                // vertex patch averages, fed through the real MaterialLaw so
                // u_a mimics -lambda(s) K grad(p) with a single consistent
                // reconstructed state rather than mixing grad p_hat with the
                // discrete cell mobility.
                if (useLiftedRelperm_) {
                    const std::size_t nvS = g.vtx.size();
                    Scalar sHatW = 0, sHatG = 0;
                    for (std::size_t k = 0; k < nvS; ++k) {
                        const int a = g.vtx[k];
                        sHatW += vtxS_[0][a];
                        sHatG += vtxS_[1][a];
                    }
                    if (nvS > 0) {
                        sHatW /= static_cast<Scalar>(nvS);
                        sHatG /= static_cast<Scalar>(nvS);
                    }
                    if (useBubbleCorrection_) {
                        // point-value bubble correction (eq. eq:averaging_bubble
                        // restricted to its value at x_K): shifts the point lift
                        // to reproduce the FV cell mean exactly; b_K(x_K)=1 and
                        // (b_K,1)_K=|K|/(d+1) give an amplification (d+1), with
                        // the cell mean of the un-bubbled lift approximated by
                        // its vertex-patch mean (documented simplification).
                        const Scalar dPlus1 = Scalar(dim + 1);
                        if (watPh >= 0)
                            sHatW += dPlus1 * (getValue(fsIn.saturation(watPh)) - sHatW);
                        if (gasPh >= 0)
                            sHatG += dPlus1 * (getValue(fsIn.saturation(gasPh)) - sHatG);
                    }
                    sHatW = std::clamp(sHatW, Scalar{0}, Scalar{1});
                    sHatG = std::clamp(sHatG, Scalar{0}, Scalar{1 - sHatW});

                    SaturationOverlayFluidState<FluidState> satFs(fsIn);
                    if (watPh >= 0) satFs.setSaturation(watPh, Evaluation(sHatW));
                    if (gasPh >= 0) satFs.setSaturation(gasPh, Evaluation(sHatG));
                    if (oilPh >= 0) satFs.setSaturation(oilPh, Evaluation(Scalar{1} - sHatW - sHatG));

                    std::array<Evaluation, FluidSystem::numPhases> krHat{};
                    MaterialLaw::template relativePermeabilities<
                        std::array<Evaluation, FluidSystem::numPhases>,
                        SaturationOverlayFluidState<FluidState>>(
                        krHat, simulator_.problem().materialLawParams(i), satFs);

                    hasLambdaHat = true;
                    if (watPh >= 0) lambdaHat[0] =
                        getValue(krHat[watPh]) / std::max(getValue(fsIn.viscosity(watPh)), Scalar{1e-30});
                    if (oilPh >= 0) lambdaHat[1] =
                        getValue(krHat[oilPh]) / std::max(getValue(fsIn.viscosity(oilPh)), Scalar{1e-30});
                    if (gasPh >= 0) lambdaHat[2] =
                        getValue(krHat[gasPh]) / std::max(getValue(fsIn.viscosity(gasPh)), Scalar{1e-30});
                }
            }
            else {
                // transmissibility-weighted least-squares fit of the connection
                // phase-pressure drops (raw FV cell values).
                if (watPh >= 0) gradP[0] = APosteriori::leastSquaresGradient<Scalar, dim>(
                    getValue(fsIn.pressure(watPh)), pNb[0], dCC, wLS);
                if (oilPh >= 0) gradP[1] = APosteriori::leastSquaresGradient<Scalar, dim>(
                    getValue(fsIn.pressure(oilPh)), pNb[1], dCC, wLS);
                if (gasPh >= 0) gradP[2] = APosteriori::leastSquaresGradient<Scalar, dim>(
                    getValue(fsIn.pressure(gasPh)), pNb[2], dCC, wLS);
            }

            // constitutive component flux u_a (eq. constitutive_flux), constant on K
            CompFlux uCell = constitutiveComponentFlux_(
                g, iqIn, fsIn, watPh, oilPh, gasPh, gradP,
                hasLambdaHat ? &lambdaHat : nullptr);
            for (unsigned c = 0; c < numComponents; ++c)
                uCell[c] *= rhoRef[c];
            curU[i] = uCell;

            const Scalar cinv = (g.cKK > Scalar{0})
                ? Scalar{1} / std::sqrt(g.cKK) : Scalar{0};

            // Pi0_K W_a sums only over genuine geometric sub-faces: an NNC has
            // no face normal/centroid and must not enter the centroid moment
            // (it still enters div W_a below, via the full Fc sum, as the FV
            // scheme's nonlocal source).
            std::vector<DimVector> dcFaceGeom;
            std::vector<std::size_t> geomIdx;
            dcFaceGeom.reserve(nf);
            geomIdx.reserve(nf);
            for (std::size_t f = 0; f < nf; ++f) {
                if (isGeomFace[f]) {
                    dcFaceGeom.push_back(dcFace[f]);
                    geomIdx.push_back(f);
                }
            }

            // Unit cell-to-face direction per connection (same dcFace convention
            // used throughout, e.g. by piZeroFromFaceFluxes -- not a separately
            // computed true geometric normal; the mvemFluxMassMatrix consistency
            // check N^T C/|K| ~= I, validated on the K-orthogonal case in
            // MvemConsistencyNTransposeCOverVolumeIsIdentity, is the built-in
            // diagnostic for how good this approximation is on a given cell).
            std::vector<DimVector> nHat(nf);
            for (std::size_t f = 0; f < nf; ++f) {
                const Scalar len = dcFace[f].two_norm();
                nHat[f] = (len > Scalar{0}) ? dcFace[f] : DimVector(Scalar{0});
                if (len > Scalar{0})
                    nHat[f] /= len;
            }
            // The MVEM matrix M_K = M^c+M^s depends only on cell geometry, not
            // on the component -- built once per cell, reused for eta_sp AND
            // eta_lin's flux term below. Skipped entirely in cheap mode
            // (nothing there uses it).
            std::vector<Scalar> MtotLocal;                     // fallback storage
            if (!cheapNorms_ && (!mvemCacheBuilt_ || i >= mvemMtotCache_.size()
                                 || mvemMtotCache_[i].empty())) {
                std::vector<DimVector> Ngeom(geomIdx.size());
                for (std::size_t k = 0; k < geomIdx.size(); ++k) {
                    const std::size_t f = geomIdx[k];
                    Ngeom[k] = nHat[f];
                    Ngeom[k] *= faceAreaVec[f];
                }
                const auto mvem = APosteriori::mvemFluxMassMatrix<Scalar, dim>(
                    Ngeom, dcFaceGeom, g.permTensor, g.volume);
                MtotLocal.assign(mvem.nf * mvem.nf, Scalar{0});
                for (std::size_t k = 0; k < MtotLocal.size(); ++k)
                    MtotLocal[k] = mvem.Mc[k] + mvem.Ms[k];
                if (i < mvemMtotCache_.size())
                    mvemMtotCache_[i] = MtotLocal;
            }
            const std::vector<Scalar>& Mtot =
                (!cheapNorms_ && i < mvemMtotCache_.size() && !mvemMtotCache_[i].empty())
                    ? mvemMtotCache_[i] : MtotLocal;

            Scalar cSpMim2 = 0, cTime2 = 0, cLin2 = 0;   // per-cell partial sums for the spatial map
            for (unsigned c = 0; c < numComponents; ++c) {
                std::vector<Scalar> FcGeom(geomIdx.size());
                for (std::size_t k = 0; k < geomIdx.size(); ++k)
                    FcGeom[k] = Fc[c][geomIdx[k]];
                const DimVector piW = APosteriori::piZeroFromFaceFluxes<Scalar, dim>(
                    FcGeom, dcFaceGeom, g.volume);

                // eq. flux_est_cp with W_a in place of Theta_a:
                //   || W_a - u_a ||_{*,K}
                //     <=  || Pi0(W_a - u_a) ||_{*,K}                       (T1)
                //       + || (I-Pi0) u_a ||_{*,K}                          (T2, = 0: u_a in P0(K)^d)
                //       + C_PW h_K c_KK^{-1/2} D_K^{l/2} || div W_a ||_K   (T3)
                DimVector d = piW;
                d -= uCell[c];
                const DimVector s = APosteriori::applyInvSqrtPerm<Scalar, dim>(d, g.permDiag);
                const Scalar T1 = APosteriori::weightedStarNorm<Scalar, dim>(
                    s, g.weightPow, g.volume);

                Scalar divW = 0.0;
                for (std::size_t f = 0; f < nf; ++f)
                    divW += Fc[c][f];
                divW /= g.volume;
                const Scalar T3 = invPi * g.hK * cinv * g.weightPow
                    * std::sqrt(g.volume) * std::abs(divW);

                // T1 (the Pi0 centroid moment) is the genuine computable
                // quantity -- exact moments of a specified constant-per-face
                // field. T3 is NOT a certified bound: it uses a plain scalar
                // Poincare constant (1/pi) and the divergence of the raw,
                // unequilibrated TPFA flux, without an actual RT/VEM lifting
                // of the face-flux moments to justify that constant or a
                // directional K-dependence for c_KK^{-1/2} (divergence is
                // scalar and does not identify which flux direction it came
                // from, so replacing c_KK by a direction-dependent constant
                // is not simply valid either). Confirmed empirically
                // (2026-09-04, SPE9 cell trace): T3 exceeds T1 by 5-800x in
                // cells with near-degenerate vertical permeability, and is
                // the entire source of the ~1e9-1e10 aggregate magnitude --
                // an artifact of this surrogate, not a certified error bound.
                // T1-only is tracked separately (sumSpT1_2/etaSpatialT1())
                // and is what actually drives timestep control; T1+T3
                // (etaSpatial()) remains available for research/reporting
                // only until a real RT/VEM-based T3 replaces this surrogate.
                const Scalar etaSpT1 = std::sqrt(std::max(dt, Scalar{0})) * T1;
                sumSpT1_2 += etaSpT1 * etaSpT1;
                const Scalar etaSp = std::sqrt(std::max(dt, Scalar{0})) * (T1 + T3);
                sumSp2 += etaSp * etaSp;

                // Mimetic (MVEM) replacement for T1+T3 together, following
                // Vohralik & Yousef, CMAME 331 (2018), Sections 3 and 6
                // (Definition 3.3, Lemmas 3.5-3.7, Remark 3.15) -- see
                // mvemFluxMassMatrix's doc comment for the full construction.
                // z = F - N.u_a (F = W_a's own geometric-face fluxes, N.u_a =
                // u_a's exact face interpolation); z^T M_K z ~= ||W_a-u_a||^2.
                // NNCs are excluded from N/C/z (no face normal/centroid to
                // build them from -- same reason T1's Pi0 moment excludes
                // them) and so do not contribute to eta_sp(mim); they still
                // enter div W_a elsewhere via the full Fc sum. z^T M^c z
                // reproduces T1^2 above exactly (see the class doc comment's
                // decomposition remark); z^T M^s z replaces T3.
                std::vector<Scalar> zgeom(geomIdx.size());
                for (std::size_t k = 0; k < geomIdx.size(); ++k) {
                    const std::size_t f = geomIdx[k];
                    zgeom[k] = Fc[c][f] - faceAreaVec[f] * (uCell[c] * nHat[f]);
                }
                // Rigorous (default): full mimetic z^T M_K z. Cheap
                // (setCheapNorms): the T1-only Pi0 moment -- no stability
                // term, diagonal K -- i.e. etaSpT1 (Mtot is not built).
                Scalar etaSpMim;
                if (cheapNorms_) {
                    etaSpMim = etaSpT1;
                }
                else {
                    const Scalar mimEnergy2 = APosteriori::mvemQuadForm<Scalar>(Mtot, zgeom);
                    etaSpMim = std::sqrt(std::max(dt, Scalar{0})) * g.weightPow
                        * std::sqrt(std::max(mimEnergy2, Scalar{0}));
                }
                sumSpMim2 += etaSpMim * etaSpMim; cSpMim2 += etaSpMim * etaSpMim;

                if (etaSp > Scalar{1e8} && std::getenv("OPM_APOST_TRACE")) {
                    std::cerr << "[apost-trace] i=" << i << " c=" << c
                              << " rhoRef=" << rhoRef[c]
                              << " T1=" << T1 << " T3=" << T3
                              << " cKK=" << g.cKK << " divW=" << divW
                              << " etaSp_K=" << etaSp << "\n";
                }

                if (i == 0 && c == 0 && std::getenv("OPM_APOST_DEBUG")) {
                    std::cerr << "[apost-debug] i=0 c=0 dt=" << dt
                              << " V_K=" << g.volume << " h_K=" << g.hK
                              << " c_KK=" << g.cKK
                              << " permDiag=(" << g.permDiag[0] << "," << g.permDiag[1] << "," << g.permDiag[2] << ")"
                              << " piW=(" << piW[0] << "," << piW[1] << "," << piW[2] << ")"
                              << " uCell=(" << uCell[c][0] << "," << uCell[c][1] << "," << uCell[c][2] << ")"
                              << " diff_d=(" << d[0] << "," << d[1] << "," << d[2] << ")"
                              << " Kinv_s=(" << s[0] << "," << s[1] << "," << s[2] << ")"
                              << " T1=" << T1 << " divW=" << divW << " T3=" << T3
                              << " etaSp=" << etaSp << " nf=" << nf << "\n";
                    for (std::size_t f = 0; f < nf; ++f)
                        std::cerr << "  [apost-debug] conn f=" << f << " Fc0=" << Fc[0][f]
                                  << " isGeom=" << (int)isGeomFace[f] << " dcFace=(" << dcFace[f][0]
                                  << "," << dcFace[f][1] << "," << dcFace[f][2] << ")\n";
                }

                if (havePrev_) {
                    DimVector du = uCell[c];
                    du -= prevU_[i][c];
                    // Same *,K energy norm as eta_D: rigorous (default) is the
                    // full-tensor P0 form D_K^l |K| (du.K^{-1}du), which for a
                    // P0 field equals the mimetic z^T M_K z on z = N du
                    // exactly; cheap (setCheapNorms) is the diagonal-K form.
                    Scalar starNorm2;
                    if (cheapNorms_) {
                        const DimVector sd = APosteriori::applyInvSqrtPerm<Scalar, dim>(du, g.permDiag);
                        const Scalar n = APosteriori::weightedStarNorm<Scalar, dim>(sd, g.weightPow, g.volume);
                        starNorm2 = n * n;
                    }
                    else {
                        starNorm2 = APosteriori::fullTensorStarNorm2<Scalar, dim>(
                            du, g.permTensor, g.weightPow, g.volume);
                    }
                    const Scalar etaT = std::sqrt(std::max(dt, Scalar{0}) / Scalar{3})
                        * std::sqrt(std::isfinite(starNorm2) ? std::max(starNorm2, Scalar{0})
                                                             : starNorm2);
                    // A transiently non-finite intensive-quantity on a hard
                    // Newton iterate makes du (hence starNorm2, etaT) NaN; let
                    // it POISON the sum -- the caller then treats eta_time as
                    // unavailable for this call and keeps OPM's native
                    // convergence/stepping (it self-corrects next iterate). Do
                    // NOT silently drop the cell and report a smaller value.
                    sumTime2 += etaT * etaT;
                    cTime2   += etaT * etaT;
                }

                if (haveLin_) {
                    // eta_lin, eq. Disc_estimator_eq_lin_alpha, both terms:
                    //
                    // (1) flux defect: Theta_lin's connection moments are
                    //     F^{k,n}_{alpha,ic} - u^{k,n}_{alpha,ic}, with
                    //     F captured pre-update (recordLinearizationDefect)
                    //     and u^{k,n} = Fc[c][f] (post-update, this call).
                    //     Unlike W_alpha-u_alpha above, Theta_lin's connection
                    //     moments are ALREADY a face-flux defect (no separate
                    //     P0 field to difference against), so the SAME MVEM
                    //     matrix Mtot built above applies directly to these
                    //     moments (z = Theta_lin's own moments, not F-N.u) --
                    //     geometric faces only, same reason NNCs are excluded
                    //     from eta_sp's construction above.
                    std::vector<Scalar> thetaLinGeom(geomIdx.size());
                    for (std::size_t k = 0; k < geomIdx.size(); ++k) {
                        const std::size_t f = geomIdx[k];
                        thetaLinGeom[k] = Flin_[i][f][c] - Fc[c][f];
                    }
                    // linEnergy2 is the UNWEIGHTED energy squared in both
                    // branches; g.weightPow (= D_K^{l/2}) is applied once
                    // outside. Rigorous: z^T M_K z. Cheap: |Pi0 Theta_lin|^2
                    // with diagonal K, no stability term.
                    Scalar linEnergy2;
                    if (cheapNorms_) {
                        const DimVector piLin = APosteriori::piZeroFromFaceFluxes<Scalar, dim>(
                            thetaLinGeom, dcFaceGeom, g.volume);
                        const DimVector sLin = APosteriori::applyInvSqrtPerm<Scalar, dim>(piLin, g.permDiag);
                        linEnergy2 = g.volume * sLin.two_norm2();
                    }
                    else {
                        linEnergy2 = APosteriori::mvemQuadForm<Scalar>(Mtot, thetaLinGeom);
                    }
                    const Scalar etaLinFlux = std::sqrt(std::max(dt, Scalar{0})) * g.weightPow
                        * std::sqrt(std::max(linEnergy2, Scalar{0}));

                    // (2) nonlinear accumulation defect (eq. est_NA form):
                    //     (tau^n)^{-1/2} eps^{-1/2} c_KK^{-1/2} h_K D_K^{l/2}
                    //     || A(chi^{k,n}) - A(chi^{k-1,n}) - L^{k,n} ||_K
                    //     NOTE: no extra Phiref factor here -- OPM's own
                    //     LocalResidual::computeStorage() already folds the
                    //     cell's current porosity into its surfaceVolume
                    //     output (intQuants.porosity(), which itself carries
                    //     the reference-porosity * rock-compaction-multiplier
                    //     product), and Anew/ownAccumVal_/Lval_ are ALL built
                    //     from that same computeStorage() call. Multiplying
                    //     by g.phiRef again here would double-count porosity.
                    const Scalar Aold = ownAccumVal_[i][c];
                    const Scalar naDefect = std::abs(Anew[c] - Aold - Lval_[i][c]);
                    const Scalar tauInv = (dt > Scalar{0}) ? Scalar{1} / std::sqrt(dt) : Scalar{0};
                    const Scalar epsInv = (epsilon_ > Scalar{0}) ? Scalar{1} / std::sqrt(epsilon_) : Scalar{0};
                    const Scalar etaNA = tauInv * epsInv * cinv * g.hK * g.weightPow
                        * std::sqrt(g.volume) * naDefect;

                    const Scalar etaL = etaLinFlux + etaNA;
                    sumLin2 += etaL * etaL; cLin2 += etaL * etaL;
                }
                else if (havePrevIter_) {
                    // Fallback (recordLinearizationDefect() never called this
                    // step): iterate-to-iterate flux-change proxy. NOT the
                    // paper's Theta_lin -- documented stand-in only -- but it
                    // uses the SAME *,K energy norm as eta_D/eta_time: full
                    // tensor P0 form by default, diagonal under cheapNorms_.
                    DimVector dl = uCell[c];
                    dl -= prevIterU_[i][c];
                    Scalar norm2;
                    if (cheapNorms_) {
                        const DimVector sl = APosteriori::applyInvSqrtPerm<Scalar, dim>(dl, g.permDiag);
                        const Scalar n = APosteriori::weightedStarNorm<Scalar, dim>(sl, g.weightPow, g.volume);
                        norm2 = n * n;
                    }
                    else {
                        norm2 = APosteriori::fullTensorStarNorm2<Scalar, dim>(
                            dl, g.permTensor, g.weightPow, g.volume);
                    }
                    const Scalar etaL = std::sqrt(std::max(dt, Scalar{0}))
                        * std::sqrt(std::max(norm2, Scalar{0}));
                    sumLin2 += etaL * etaL; cLin2 += etaL * etaL;
                }
            }
            if (i < cellEta_.size()) {
                cellEta_[i][0] = std::sqrt(std::max(cSpMim2, Scalar{0}));
                cellEta_[i][1] = std::sqrt(std::max(cTime2,  Scalar{0}));
                cellEta_[i][2] = std::sqrt(std::max(cLin2,   Scalar{0}));
                // cellEta_[i][3] (eta_alg) is filled by computeAlgebraicEstimator()
            }
          }
        }
        if (!cheapNorms_)
            mvemCacheBuilt_ = true;   // M_K now cached until the next updateGeometry()

        const auto& comm = gridView.comm();
        sumSp2    = comm.sum(sumSp2);
        sumSpT1_2 = comm.sum(sumSpT1_2);
        sumSpMim2 = comm.sum(sumSpMim2);
        sumTime2  = comm.sum(sumTime2);
        sumLin2   = comm.sum(sumLin2);

        etaSp_    = std::sqrt(std::max(sumSp2, Scalar{0}));
        etaSpT1_  = std::sqrt(std::max(sumSpT1_2, Scalar{0}));
        etaSpMim_ = std::sqrt(std::max(sumSpMim2, Scalar{0}));
        etaTime_ = (havePrev_ && std::isfinite(sumTime2))
                       ? std::sqrt(std::max(sumTime2, Scalar{0}))
                       : std::numeric_limits<Scalar>::quiet_NaN();
        const bool haveLinNow = haveLin_ || havePrevIter_;
        etaLin_  = (haveLinNow && std::isfinite(sumLin2))
                       ? std::sqrt(std::max(sumLin2, Scalar{0}))
                       : std::numeric_limits<Scalar>::quiet_NaN();
        etaLinRigorousLastCall_ = haveLin_;
        haveLin_ = false; // consumed: next compute() falls back to the proxy
                          // unless recordLinearizationDefect() is called again first

        if (commitHistory) {
            prevU_.swap(curU);
            havePrev_ = true;
        }
        else {
            prevIterU_ = curU;
        }
        havePrevIter_ = true;
    }

    //! T1+T3 (full eq. flux_est_cp split). T3 is currently an uncertified
    //! surrogate (see the doc comment at its assembly site in compute()) --
    //! this getter is for research/reporting only. Timestep control uses
    //! etaSpatialT1() instead.
    Scalar etaSpatial()      const { return etaSp_; }
    //! T1 only (the Pi0 centroid moment, a genuine computable quantity) --
    //! kept for comparison; superseded by etaSpatialMimetic() below as the
    //! default control quantity.
    Scalar etaSpatialT1()    const { return etaSpT1_; }
    //! MVEM (mimetic virtual element) replacement for T1+T3 together, z^T M_K z
    //! with M_K = M^c + M^s (Vohralik & Yousef CMAME 331 (2018) Sections 3
    //! and 6; see mvemFluxMassMatrix's and the class's doc comments). M^c
    //! reproduces T1^2 exactly; M^s replaces T3's c_KK^{-1/2}*|div| surrogate
    //! with a face-DOF-projector kernel measure. This is what drives
    //! Criteria_space_time_balance / --enable-aposteriori-timestep-control
    //! (superseding etaSpatialT1()).
    //!
    //! Two caveats remain: (a) covers only the flux-energy term of the
    //! paper's full three-term Theorem 3.12 estimate -- the cross term and
    //! the potential-reconstruction stiffness term (a P1-FEM assembly on a
    //! simplicial sub-tessellation, eq. 3.16-3.21) are not implemented;
    //! (b) the M^s stability scaling (epsilon, default 1e-2) is the one
    //! design degree of freedom the MFD family leaves open (Lemma 3.7 gives
    //! mesh-independent spectral equivalence, not equality with constant 1).
    //! The certified route (exact for the RT reconstruction) is Lemma 3.5's
    //! RT Schur complement, which needs the sub-tessellation and a local
    //! mixed-FE solve -- not done.
    Scalar etaSpatialMimetic() const { return etaSpMim_; }
    Scalar etaTemporal()     const { return etaTime_; }
    Scalar etaLinearization() const { return etaLin_; }
    bool   temporalAvailable()     const { return havePrev_ && std::isfinite(etaTime_); }
    bool   linearizationAvailable() const {
        return (havePrevIter_ || etaLinRigorousLastCall_) && std::isfinite(etaLin_);
    }
    //! True if the most recent etaLinearization() came from the rigorous
    //! Theta_lin/L construction (recordLinearizationDefect() was called this
    //! iteration), false if it fell back to the iterate-diff proxy.
    bool   linearizationRigorous() const { return etaLinRigorousLastCall_; }

    //! Discard the stored previous-step reconstruction (e.g. after a chop) so
    //! the next temporal estimate is skipped rather than computed from a stale
    //! state.
    void invalidateHistory() { havePrev_ = false; }

    //! Start a new Newton loop: the next compute() call has no previous
    //! iterate to diff against for eta_lin.  Call once per timestep, before
    //! the first Newton iteration.
    void resetNewtonIterateHistory() { havePrevIter_ = false; }

private:
    // ---- one-rebuild h-adaptivity: mask + box helpers --------------------
    struct MaskBox { std::array<int, 3> lo; std::array<int, 3> hi; };

    static std::size_t cartIndex_(int i, int j, int k, const std::array<int, 3>& dims)
    {
        return static_cast<std::size_t>((static_cast<std::int64_t>(k) * dims[1] + j)
                                        * dims[0] + i);
    }

    bool isProtectedCart_(int cart) const
    {
        return std::binary_search(protectedCartesian_.begin(),
                                  protectedCartesian_.end(), cart);
    }

    //! Steps 1-5 of the plan: Dorfler-mark from accumulatedSpatialEnergy_,
    //! drop protected cells, dilate by the halo, drop protected cells again
    //! (dilation may have re-added them). Output: a coarse Cartesian Boolean
    //! \p mask (row-major i-fastest) and the coarse \p dims. When \p nSeed is
    //! non-null it receives the post-protection, pre-halo Dorfler count.
    void buildRefineMask_(double theta, std::vector<char>& mask,
                          std::array<int, 3>& dims, int* nSeed) const
    {
        const auto& mapper = simulator_.vanguard().cartesianIndexMapper();
        const auto& cdims  = mapper.cartesianDimensions();
        for (int d = 0; d < 3; ++d)
            dims[d] = (d < static_cast<int>(cdims.size())) ? static_cast<int>(cdims[d]) : 1;
        const std::size_t ncart =
            static_cast<std::size_t>(dims[0]) * dims[1] * dims[2];
        mask.assign(ncart, 0);
        if (nSeed) *nSeed = 0;

        const std::size_t nc = std::min(accumulatedSpatialEnergy_.size(), cellEta_.size());
        if (nc == 0 || !(theta > 0.0 && theta < 1.0)) return;

        // ---- FULL protected mask, built BEFORE Dorfler ranking -------------
        // Point completions PLUS the whole vertical (i,j) interval a vertical
        // well spans. Previously the column interval was carved out only AFTER
        // ranking/denominator, so an intervening high-energy cell could satisfy
        // the target and then be deleted, silently losing captured energy
        // (review 2026-09-10). The SAME set is excluded from ranking, from the
        // denominator, and from the post-halo subtraction.
        std::vector<char> fullProt(ncart, 0);
        {
            std::vector<std::array<int, 3>> prot;
            prot.reserve(protectedCartesian_.size());
            for (int cart : protectedCartesian_) {
                if (cart < 0 || static_cast<std::size_t>(cart) >= ncart) continue;
                fullProt[static_cast<std::size_t>(cart)] = 1;
                prot.push_back({cart % dims[0], (cart / dims[0]) % dims[1],
                                cart / (dims[0] * dims[1])});
            }
            std::sort(prot.begin(), prot.end());
            for (std::size_t a = 0; a < prot.size();) {
                std::size_t b = a;
                int kmin = prot[a][2], kmax = prot[a][2];
                while (b < prot.size() && prot[b][0] == prot[a][0]
                       && prot[b][1] == prot[a][1]) {
                    kmin = std::min(kmin, prot[b][2]);
                    kmax = std::max(kmax, prot[b][2]);
                    ++b;
                }
                for (int k = kmin; k <= kmax; ++k)
                    fullProt[cartIndex_(prot[a][0], prot[a][1], k, dims)] = 1;
                a = b;
            }
        }
        const auto isProt = [&](int cart) {
            return cart >= 0 && static_cast<std::size_t>(cart) < ncart
                && fullProt[static_cast<std::size_t>(cart)] != 0;
        };

        // Dorfler: smallest cell set with sum E_K^2 >= theta * total, over
        // NON-protected interior cells only. totalAll is the full estimator
        // energy (incl. protected) so lastRetainedEnergyFraction_ is honest.
        std::vector<std::pair<double, int>> ord;      // (E_K^2, cartIdx)
        ord.reserve(nc);
        double total = 0.0, totalAll = 0.0;
        for (std::size_t i = 0; i < nc; ++i) {
            const double e2 = accumulatedSpatialEnergy_[i];
            if (!(e2 > 0.0)) continue;
            totalAll += e2;
            std::array<int, 3> ijk{0, 0, 0};
            mapper.cartesianCoordinate(static_cast<int>(i), ijk);
            const int cart = static_cast<int>(cartIndex_(ijk[0], ijk[1], ijk[2], dims));
            if (isProt(cart)) continue;
            total += e2;
            ord.emplace_back(e2, cart);
        }
        if (!(total > 0.0)) { lastRetainedEnergyFraction_ = 0.0; return; }
        std::sort(ord.begin(), ord.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Optional hard budget on the number of refined coarse cells
        // (OPM_APOST_MAX_REFINED_CELLS); the highest-energy seeds are kept.
        std::size_t cellBudget = ord.size();
        if (const char* s = std::getenv("OPM_APOST_MAX_REFINED_CELLS")) {
            const long v = std::atol(s);
            if (v > 0) cellBudget = static_cast<std::size_t>(v);
        }

        const double target = theta * total;
        double acc = 0.0;
        std::size_t nSel = 0;
        for (const auto& [e2, cart] : ord) {
            if (acc >= target || nSel >= cellBudget) break;
            acc += e2;
            ++nSel;
            mask[static_cast<std::size_t>(cart)] = 1;
        }
        lastRetainedEnergyFraction_ = (totalAll > 0.0) ? acc / totalAll : 0.0;
        if (nSeed)
            for (char c : mask) if (c) ++(*nSeed);

        // Halo dilation (only along axes that actually have >1 coarse cell).
        int halo = 1;
        if (const char* v = std::getenv("OPM_APOST_REFINE_HALO"))
            halo = std::max(0, std::atoi(v));
        for (int pass = 0; pass < halo; ++pass) {
            std::vector<char> grown = mask;
            for (int k = 0; k < dims[2]; ++k)
            for (int j = 0; j < dims[1]; ++j)
            for (int i = 0; i < dims[0]; ++i) {
                if (!mask[cartIndex_(i, j, k, dims)]) continue;
                for (int d = 0; d < 3; ++d) {
                    if (dims[d] <= 1) continue;
                    for (int s : {-1, 1}) {
                        std::array<int, 3> a{i, j, k};
                        a[d] += s;
                        if (a[d] < 0 || a[d] >= dims[d]) continue;
                        grown[cartIndex_(a[0], a[1], a[2], dims)] = 1;
                    }
                }
            }
            mask.swap(grown);
        }

        // Subtract the SAME full protected mask again -- halo dilation can have
        // re-introduced protected cells.
        for (std::size_t c = 0; c < ncart; ++c)
            if (fullProt[c]) mask[c] = 0;
    }

    //! Pack a Boolean coarse-cell mask into disjoint axis-aligned boxes whose
    //! union is exactly the mask. Greedy: grow in i, then whole rows in j,
    //! then whole planes in k. All boxes share the same subdivision factor, so
    //! touching faces stay conforming.
    std::vector<MaskBox> packMaskIntoBoxes_(const std::vector<char>& maskIn,
                                            const std::array<int, 3>& dims) const
    {
        std::vector<char> m = maskIn;   // consumed as boxes are claimed
        std::vector<MaskBox> boxes;
        const auto get = [&](int i, int j, int k) -> char& {
            return m[cartIndex_(i, j, k, dims)];
        };
        for (int k0 = 0; k0 < dims[2]; ++k0)
        for (int j0 = 0; j0 < dims[1]; ++j0)
        for (int i0 = 0; i0 < dims[0]; ++i0) {
            if (!get(i0, j0, k0)) continue;
            int i1 = i0;
            while (i1 + 1 < dims[0] && get(i1 + 1, j0, k0)) ++i1;
            int j1 = j0;
            for (bool grow = true; grow && j1 + 1 < dims[1]; ) {
                for (int i = i0; i <= i1; ++i)
                    if (!get(i, j1 + 1, k0)) { grow = false; break; }
                if (grow) ++j1;
            }
            int k1 = k0;
            for (bool grow = true; grow && k1 + 1 < dims[2]; ) {
                for (int j = j0; j <= j1 && grow; ++j)
                    for (int i = i0; i <= i1; ++i)
                        if (!get(i, j, k1 + 1)) { grow = false; break; }
                if (grow) ++k1;
            }
            for (int k = k0; k <= k1; ++k)
                for (int j = j0; j <= j1; ++j)
                    for (int i = i0; i <= i1; ++i)
                        get(i, j, k) = 0;
            boxes.push_back(MaskBox{{i0, j0, k0}, {i1, j1, k1}});
        }
        return boxes;
    }

    //! Continuous constitutive component flux, eq. (constitutive_flux):
    //!   v_hat_beta = -lambda_beta(s_hat) K (grad p_hat_beta - rho_beta g grad z),
    //!   u_w = b_w v_w,   u_o = b_o v_o + r_v b_g v_g,   u_g = b_g v_g + r_s b_o v_o,
    //! in surface-volume units (matching the scheme's conserved A_alpha and the
    //! flux W_alpha).  \p lambdaHat, if non-null, supplies lambda_beta(s_hat)
    //! evaluated at the lifted saturation (index 0/1/2 = water/oil/gas), so the
    //! flux mimics the continuous law with a single consistent reconstructed
    //! state; otherwise the FV cell mobility is used.  b_beta, rho_beta, R_s,
    //! R_v remain at the FV cell state in both cases (documented Tier-A scope).
    //! The per-phase pressure gradients \p gradP (index 0/1/2 = water/oil/gas)
    //! retain the capillary gradients grad p_hat_beta = grad p_hat_o -
    //! p_c,ob'(s) grad s.  Constant on K (lowest order): (I - Pi0_K) u_alpha = 0.
    template<class IQ, class FS>
    CompFlux
    constitutiveComponentFlux_(const CellGeom& g,
                               const IQ& iqIn,
                               const FS& fsIn,
                               int watPh, int oilPh, int gasPh,
                               const std::array<DimVector, 3>& gradP,
                               const std::array<Scalar, 3>* lambdaHat) const
    {
        const DimVector& grav = simulator_.problem().gravity();

        auto phaseVelocity = [&](int ph, int gpIdx) -> DimVector {
            DimVector v(0.0);
            if (ph < 0)
                return v;
            const Scalar mob = lambdaHat ? (*lambdaHat)[gpIdx] : getValue(iqIn.mobility(ph));
            const Scalar rho = getValue(fsIn.density(ph));
            // v = -lambda K (grad p - rho g): full permeability tensor, so the
            // reconstructed velocity is consistent with the full-tensor energy
            // norm applied to it later (a diagonal-only product would describe
            // a different constitutive law for non-diagonal K).
            DimVector dphi(0.0);
            for (int d = 0; d < dim; ++d)
                dphi[d] = gradP[gpIdx][d] - rho * grav[d];
            DimVector Kdphi(0.0);
            g.permTensor.mv(dphi, Kdphi);
            for (int d = 0; d < dim; ++d)
                v[d] = -mob * Kdphi[d];
            return v;
        };

        const DimVector vW = phaseVelocity(watPh, 0);
        const DimVector vO = phaseVelocity(oilPh, 1);
        const DimVector vG = phaseVelocity(gasPh, 2);

        const Scalar bW = (watPh >= 0) ? getValue(fsIn.invB(watPh)) : Scalar{0};
        const Scalar bO = (oilPh >= 0) ? getValue(fsIn.invB(oilPh)) : Scalar{0};
        const Scalar bG = (gasPh >= 0) ? getValue(fsIn.invB(gasPh)) : Scalar{0};
        const Scalar Rs = (oilPh >= 0 && gasPh >= 0) ? getValue(fsIn.Rs()) : Scalar{0};
        const Scalar Rv = (oilPh >= 0 && gasPh >= 0) ? getValue(fsIn.Rv()) : Scalar{0};

        CompFlux u;
        for (auto& e : u)
            e = DimVector(0.0);

        auto add = [](DimVector& dst, const DimVector& src, Scalar s) {
            for (int d = 0; d < dim; ++d)
                dst[d] += s * src[d];
        };

        // Active (not canonical) component indices: for a reduced (e.g.
        // two-phase) system canonicalToActiveCompIdx is not the identity, and
        // u[] is sized numComponents (active count) -- indexing it by the raw
        // canonical index would write into the wrong slot (or out of range).
        // Must match Fc[]'s indexing in compute() (the comp() lambda there).
        if (watPh >= 0)
            add(u[FluidSystem::canonicalToActiveCompIdx(FluidSystem::solventComponentIndex(watPh))], vW, bW);
        if (oilPh >= 0) {
            const unsigned oc = FluidSystem::canonicalToActiveCompIdx(FluidSystem::solventComponentIndex(oilPh));
            add(u[oc], vO, bO);
            if (gasPh >= 0)
                add(u[oc], vG, Rv * bG);
        }
        if (gasPh >= 0) {
            const unsigned gc = FluidSystem::canonicalToActiveCompIdx(FluidSystem::solventComponentIndex(gasPh));
            add(u[gc], vG, bG);
            if (oilPh >= 0)
                add(u[gc], vO, Rs * bO);
        }
        return u;
    }

    //! Vertex patch averages of the phase pressures and (s_w, s_g)
    //! (eq. eq:averaging).  Serial-exact; on a parallel partition boundary a
    //! vertex sees only its local cells (documented limitation, like the
    //! well-centre gather).
    void computeVertexAverages_()
    {
        const auto& gridView = simulator_.gridView();
        auto& model          = simulator_.model();
        const auto& vertexMapper = model.vertexMapper();
        ElementMapper mapper(gridView, Dune::mcmgElementLayout());

        const int watPh = FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)
            ? FluidSystem::waterPhaseIdx : -1;
        const int oilPh = FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)
            ? FluidSystem::oilPhaseIdx : -1;
        const int gasPh = FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)
            ? FluidSystem::gasPhaseIdx : -1;

        const std::size_t nv = vertexMapper.size();
        for (auto& v : vtxP_)
            v.assign(nv, Scalar{0});
        for (auto& v : vtxS_)
            v.assign(nv, Scalar{0});
        std::vector<int> cnt(nv, 0);

        for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
            const unsigned i = mapper.index(elem);
            const auto& fs = model.intensiveQuantities(i, /*timeIdx=*/0).fluidState();
            const Scalar pw = (watPh >= 0) ? getValue(fs.pressure(watPh)) : Scalar{0};
            const Scalar po = (oilPh >= 0) ? getValue(fs.pressure(oilPh)) : Scalar{0};
            const Scalar pg = (gasPh >= 0) ? getValue(fs.pressure(gasPh)) : Scalar{0};
            const Scalar sw = (watPh >= 0) ? getValue(fs.saturation(watPh)) : Scalar{0};
            const Scalar sg = (gasPh >= 0) ? getValue(fs.saturation(gasPh)) : Scalar{0};
            const int ncrn = elem.geometry().corners();
            for (int k = 0; k < ncrn; ++k) {
                const auto a = vertexMapper.subIndex(elem, k, dim);
                vtxP_[0][a] += pw;
                vtxP_[1][a] += po;
                vtxP_[2][a] += pg;
                vtxS_[0][a] += sw;
                vtxS_[1][a] += sg;
                ++cnt[a];
            }
        }
        for (std::size_t a = 0; a < nv; ++a) {
            const Scalar inv = (cnt[a] > 0) ? Scalar{1} / static_cast<Scalar>(cnt[a]) : Scalar{0};
            vtxP_[0][a] *= inv;
            vtxP_[1][a] *= inv;
            vtxP_[2][a] *= inv;
            vtxS_[0][a] *= inv;
            vtxS_[1][a] *= inv;
        }
    }

    //! Smallest eigenvalue of the (symmetric) cell permeability tensor.
    static Scalar smallestEigenvalue_(const Dune::FieldMatrix<Scalar, dim, dim>& K)
    {
        Scalar mn = std::numeric_limits<Scalar>::max();
        for (int d = 0; d < dim; ++d)
            mn = std::min(mn, K[d][d]);
        try {
            Dune::FieldVector<Scalar, dim> ev(0.0);
            Dune::FMatrixHelp::eigenValues(K, ev);
            for (int d = 0; d < dim; ++d)
                if (std::isfinite(ev[d]))
                    mn = std::min(mn, ev[d]);
        }
        catch (...) {
            // keep the diagonal minimum
        }
        return std::max(mn, Scalar{0});
    }

    //! Per-active-component reference (surface) mass density, for converting
    //! every flux/accumulation quantity from raw surface-VOLUME units to a
    //! common surface-MASS basis before they are combined into a single norm.
    //!
    //! Without this, water/oil/gas contributions are not commensurable: gas's
    //! surface volume is typically orders of magnitude larger than its
    //! reservoir volume (small B_g), so eta_sp,K/eta_lin,K/eta_time,K summed
    //! directly over components is dominated by whichever component happens
    //! to have the largest surface-volume-per-reservoir-volume ratio, not by
    //! whichever actually carries the largest physical discretization error.
    //! Scaling by each component's own reference density (mass = surface
    //! volume * rho_ref) is a reasonable, standard black-oil norm choice for
    //! making different components' conservation equations commensurable --
    //! it is a design choice appropriate to this black-oil implementation,
    //! NOT a requirement inherited from the paper's (immiscible two-phase)
    //! theory, which has no analogous per-component unit mismatch to correct.
    //! Applied consistently at the source (Fc, u_alpha, the recorded
    //! flux/accumulation Jacobians) so every downstream quantity -- T1, T3,
    //! div W, eta_time's iterate difference, both eta_lin terms -- inherits
    //! it automatically.
    //!
    //! Verified empirically (2026-09-04, SPE9): this does NOT reduce the
    //! estimator's aggregate magnitude -- pre-scaling, every eta_sp,K
    //! contribution above ~1e9 was the gas component (small reference
    //! density, large surface-volume flux); post-scaling, the aggregate is
    //! ~3x LARGER, because oil's much larger reference density now amplifies
    //! its own T3 term past where gas's was. A full-run cell trace confirmed
    //! T3 -- not component identity -- is the actual driver in both cases
    //! (exceeding T1 by 5-800x wherever c_KK is near-degenerate); T3 is
    //! currently an uncertified surrogate (see its assembly site in
    //! compute()) and does not drive timestep control regardless.
    //!
    //! Components other than the three canonical phases (solvent, polymer,
    //! ...) are left unscaled (rho_ref = 1); those modules are not exercised
    //! by the decks tested so far and their own reference-density convention
    //! would need separate handling.
    std::array<Scalar, numComponents>
    componentRefDensities_(int watPh, int oilPh, int gasPh, unsigned pvtRegionIdx) const
    {
        std::array<Scalar, numComponents> rhoRef{};
        rhoRef.fill(Scalar{1});
        const auto comp = [](int ph) {
            return FluidSystem::canonicalToActiveCompIdx(FluidSystem::solventComponentIndex(ph));
        };
        if (watPh >= 0) rhoRef[comp(watPh)] = FluidSystem::referenceDensity(watPh, pvtRegionIdx);
        if (oilPh >= 0) rhoRef[comp(oilPh)] = FluidSystem::referenceDensity(oilPh, pvtRegionIdx);
        if (gasPh >= 0) rhoRef[comp(gasPh)] = FluidSystem::referenceDensity(gasPh, pvtRegionIdx);
        return rhoRef;
    }

    Simulator& simulator_;

    PressureRecon pressureRecon_ {PressureRecon::PatchAverageLift};
    bool useLiftedRelperm_ {true};
    bool useBubbleCorrection_ {true};
    Scalar ell_ {0};
    std::vector<int> wellCells_;

    //! sum_n eta_sp,K,n^2 per compressed interior cell -- the time-accumulated
    //! spatial energy E_K^2 used for Dorfler marking in the one-rebuild
    //! h-adaptivity path (accumulateSpatialEnergy() / *Accumulated()). The
    //! implemented eta_sp,K already carries sqrt(tau_n), so its square already
    //! contributes the time-step weight -- do NOT multiply by tau_n again.
    std::vector<double> accumulatedSpatialEnergy_;

    //! Sorted GLOBAL Cartesian indices of every well-completion cell that
    //! appears anywhere in the remaining schedule (schedule-wide protected
    //! mask). Refinement boxes must never contain one of these -- a coarse
    //! completion cell must stay GLOBAL, never become LGR. Set once by the
    //! driver via setProtectedRefinementCells(); NOT rebuilt per Newton step.
    std::vector<int> protectedCartesian_;

    //! Fraction of total accumulated spatial energy the last mask selection
    //! captured (diagnostic; set by buildRefineMask_, which is const).
    mutable double lastRetainedEnergyFraction_ {0.0};

    std::array<std::vector<Scalar>, 3> vtxP_;   //!< vertex patch-average phase pressures
    std::array<std::vector<Scalar>, 2> vtxS_;   //!< vertex patch-average (s_w, s_g)

    std::vector<CellGeom> geom_;
    bool geomValid_ {false};

    //! Cached MVEM flux mass matrix M_K per interior cell (flat nf*nf, row
    //! major). Rebuilt lazily on the first compute() after updateGeometry().
    std::vector<std::vector<Scalar>> mvemMtotCache_;
    bool mvemCacheBuilt_ {false};

    std::vector<CompFlux> prevU_;
    bool havePrev_ {false};

    std::vector<CompFlux> prevIterU_;   //!< previous Newton *iterate*'s u_a (cheap eta_lin fallback proxy)
    bool havePrevIter_ {false};

    // --- eta_lin (recordLinearizationDefect / eq. Newton_it_flux, lin_fv_balance) ---
    // PARTIAL linearization indicator: geometric-face flux Taylor defect +
    // storage Taylor defect. Does NOT yet include the nonlinear well/source
    // Taylor defect Q(chi^k) - Q_lin^k or NNC linearization terms, so a small
    // eta_lin is not proof that the full nonlinear reservoir residual remainder
    // is small (external review, 2026-09-06).
    std::vector<std::vector<CompEval>> ownFlux_;      //!< [cell][conn] u_ic(chi^{k-1,n}), deriv wrt own cell only
    std::vector<CompFlux0>             ownAccumVal_;  //!< [cell] A_alpha(chi^{k-1,n})
    std::vector<CompEval>              ownAccumDeriv_;//!< [cell] A_alpha Evaluation (for its derivative wrt own cell)
    std::vector<std::vector<CompFlux0>> Flin_;        //!< [cell][conn] F^{k,n}_{alpha,ic} (eq. Newton_it_flux)
    std::vector<CompFlux0>             Lval_;         //!< [cell] L^{k,n}_{alpha,K} (linearized accumulation increment)
    bool haveLin_ {false};
    Scalar epsilon_ {1};  //!< Neumann-scaling parameter (eq. eps_norm); paper recommends 1
    bool cheapNorms_ {false};  //!< drop all *,K energy norms to the cheap diagonal form -- see setCheapNorms()

    Scalar etaSp_   {0};
    Scalar etaSpT1_ {0};
    Scalar etaSpMim_ {0};
    Scalar etaTime_ {0};
    Scalar etaLin_  {0};
    Scalar etaAlg_  {0};

    //! per-cell {eta_sp(mim), eta_time, eta_lin, eta_alg} for the spatial map;
    //! filled by compute() and computeAlgebraicEstimator(), dumped by
    //! dumpCellEstimators().
    std::vector<std::array<double, 4>> cellEta_;
    bool   etaLinRigorousLastCall_ {false};
};

} // namespace Opm

#endif // OPM_APOSTERIORI_SPATIAL_TEMPORAL_ESTIMATOR_HPP
