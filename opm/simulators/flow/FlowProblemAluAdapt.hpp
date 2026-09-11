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
*/
/*!
 * \file
 * \copydoc Opm::FlowProblemAluAdapt
 */
#ifndef OPM_FLOW_PROBLEM_ALU_ADAPT_HPP
#define OPM_FLOW_PROBLEM_ALU_ADAPT_HPP

#include <opm/simulators/flow/FlowProblemBlackoil.hpp>
#include <opm/simulators/flow/AdaptiveStateTransfer.hpp>

#include <opm/common/OpmLog/OpmLog.hpp>

#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/common/partitionset.hh>

#include <fmt/format.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_set>

namespace Opm {

/*!
 * \brief Black-oil Flow problem for native (non-dune-fem) in-place h-adaptivity
 *        on a locally-adaptive grid (dune-ALUGrid).
 *
 * Adds the adaptation hooks FvBaseDiscretizationAdaptiveNative calls:
 *   markForGridAdaptation()  -- mark leaf fathers (Phase 1: fixed test box;
 *                               Phase 5: the a posteriori SpatialMarker)
 *   prepareForAdapt()        -- snapshot component inventory + vanguard Cartesian ids
 *   gridChanged()            -- rebuild transmissibility / porosity / rock / Pff data
 *   verifyAfterAdapt()       -- component-inventory conservation gate
 *
 * markForGridAdaptation() DELIBERATELY shadows MultiPhaseBaseProblem's
 * saturation-variation heuristic (it marks inside a per-phase loop and
 * miscounts); that returns later behind --adapt-indicator=saturation.
 *
 * Well cells are ALWAYS vetoed from refinement (never remeshed), regardless
 * of what drives the marking (fixed test box now, the a posteriori estimator
 * from Phase 5 on). Two reasons:
 *   - engineering: OPM's well connections resolve their compressed cell
 *     index live (WellConnectionAuxiliaryModule -> vanguard().
 *     compressedIndexForInterior(cartesian_idx)), not cached once. A well
 *     cell that is never split needs no further bookkeeping across an
 *     adapt -- the existing Cartesian-to-compressed rebuild (Phase 2) remaps
 *     it for free. Splitting it into children instead would need a Peaceman
 *     WI redesign and a one-to-many PerforationData schema OPM doesn't have
 *     (even CpGrid's well-in-LGR path, compressedIndexForInteriorLGR, routes
 *     to a single child cell -- it does not split flow across children).
 *   - theoretical: the paper's weighted norm already discounts the near-well
 *     singularity analytically via the D_K/weight term; refining the well's
 *     own cell doesn't reduce that term, so there is no estimator upside to
 *     match the engineering cost.
 */
template <class TypeTag>
class FlowProblemAluAdapt : public FlowProblemBlackoil<TypeTag>
{
    using Base = FlowProblemBlackoil<TypeTag>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;

public:
    explicit FlowProblemAluAdapt(Simulator& simulator)
        : Base(simulator)
    {
        // Field-property lookup for a refined leaf must go through the
        // vanguard's CartesianIndexMapper (correctly rebuilt across every
        // adapt()), not LookUpData's own level-0-index-set fallback -- see
        // LookUpData::setCartesianIndexMapper()'s doc. Without this, PORV/
        // PVTNUM/SATNUM/PERM*/... are silently wrong for EVERY level-0 cell
        // (refined or not) as soon as maxLevel() > 0 anywhere, invisible on
        // a uniform deck but ~18% off on a heterogeneous one (found via SPE9).
        this->setLookUpCartesianIndexMapper(
            &this->simulator().vanguard().cartesianIndexMapper());
    }

    unsigned markForGridAdaptation()
    {
        const char* spec = std::getenv("OPM_ALU_ADAPT_TEST_BOX");
        if (spec == nullptr || aluAdaptTestBoxDone_) {
            return 0;
        }
        std::array<int, 6> b{};
        {
            std::istringstream is(spec);
            for (int& v : b) {
                if (!(is >> v)) {
                    OpmLog::warning("OPM_ALU_ADAPT_TEST_BOX: expected 6 integers "
                                    "'i1 i2 j1 j2 k1 k2' (1-based inclusive).");
                    return 0;
                }
            }
        }
        auto& grid = this->simulator().vanguard().grid();
        const auto& gridView = this->simulator().vanguard().gridView();
        const auto& mapper = this->simulator().vanguard().cartesianIndexMapper();
        const auto& dims = mapper.cartesianDimensions();

        Dune::MultipleCodimMultipleGeomTypeMapper<GridView>
            elemMapper(gridView, Dune::mcmgElementLayout());

        const auto wellProtected = wellProtectedCells_();
        unsigned n = 0;
        unsigned vetoed = 0;
        for (const auto& e : elements(gridView, Dune::Partitions::interior)) {
            const int elemIdx = elemMapper.index(e);
            const int cart = mapper.cartesianIndex(elemIdx);
            const int i = cart % dims[0];
            const int j = (cart / dims[0]) % dims[1];
            const int k = cart / (dims[0] * dims[1]);
            if (i + 1 >= b[0] && i + 1 <= b[1] &&
                j + 1 >= b[2] && j + 1 <= b[3] &&
                k + 1 >= b[4] && k + 1 <= b[5]) {
                if (wellProtected.count(elemIdx) != 0) {
                    ++vetoed;
                    continue; // never refine a well cell -- see class doc.
                }
                grid.mark(1, e);
                ++n;
            }
        }
        aluAdaptTestBoxDone_ = true;
        OpmLog::info(fmt::format(
            "[alu-hadapt] OPM_ALU_ADAPT_TEST_BOX i{}-{} j{}-{} k{}-{} "
            "-> marked {} leaf cell(s) for refinement ({} vetoed: well cell)",
            b[0], b[1], b[2], b[3], b[4], b[5], n, vetoed));
        return grid.comm().sum(n);
    }

    //! Called BEFORE grid.adapt().
    void prepareForAdapt()
    {
        invBefore_ = blackOilComponentInventory<TypeTag>(this->simulator());
        this->simulator().vanguard().snapshotForAdapt();
    }

    //! Called AFTER the grid + solution have been remapped: rebuild every
    //! geometry-dependent problem-side quantity (same recipe as the
    //! GEO_MODIFIER schedule-event path).
    void gridChanged()
    {
        Base::gridChanged();

        // LookUpData's OWN element mapper is built once and never follows a
        // later adapt() on its own (same reason the discretization's mappers
        // need placement-new, not just being left alone) -- refresh it
        // FIRST, or every field-property lookup below reads through a stale
        // mapper (silently out-of-range, not just wrong-cell).
        this->refreshLookUpElementMapper();

        // Rebuild every per-cell quantity against the adapted leaf grid. The
        // leaf assigners (LookUpData) map a refined child to its level-0
        // ancestor, so region numbers / porosity / rock / relperm params are
        // inherited from the parent.
        this->readMaterialParameters_();     // pvtnum, satnum, poro, rock, materialLaw

        // The centroid provider captured a value-copy of the pre-adapt
        // CartesianIndexMapper -- rebuild it from the (in-place updated) mapper.
        this->transmissibilities_.setCentroids(
            this->simulator().vanguard().cellCentroids());
        this->transmissibilities_.update(/*global=*/true);

        this->updatePffDofData_();
        this->model().linearizer().updateDiscretizationParameters();

        // The ECL output module's per-cell field buffers (pressure,
        // saturations, ...) are sized once and, as an optimization, left
        // untouched on ordinary sub-steps -- see the alloc_fields gate in
        // GenericOutputModule::doAllocBuffers, which assumes the cell count
        // never changes mid-simulation. That assumption just broke: the
        // grid was adapted on this very sub-step, so those buffers are
        // still sized for the pre-adapt cell count. A subsequent
        // EclWriter::prepareLocalCellData() call would then write element
        // data past the end of an undersized buffer (silent heap
        // corruption, surfacing later as a garbage globalSpaceIndex()).
        // Force one full reallocation now by passing substep=false.
        this->eclWriter().mutableOutputModule().allocBuffers(
            static_cast<unsigned>(this->model().numGridDof()),
            static_cast<unsigned>(this->simulator().episodeIndex()),
            /*substep=*/false,
            /*log=*/false,
            /*isRestart=*/false);

        // FIPNUM and friends are mapped onto the leaf grid once, at output
        // module construction time (a refined child inherits its parent's
        // region). That mapping is now stale -- there are more leaf cells
        // than region-array slots -- so rebuild it from source.
        this->eclWriter().mutableOutputModule().refreshRegionsAfterAdapt();

        // The well model's local_num_cells_ / legacy PVT-region & depth
        // caches / perforated-cell flags are likewise sized ONCE (its
        // constructor + init()). Refresh before the next beginReportStep()
        // -> initializeWellState() indexes them by the new leaf count --
        // see refreshAfterGridAdapt()'s own comment for the exact overrun.
        // Well cells are never refined (wellProtectedCells_()), so no
        // connection's parent-cell mapping needs updating -- only sizes.
        this->wellModel().refreshAfterGridAdapt();
    }

    //! Called AFTER intensive quantities have been recomputed on the adapted
    //! grid: rebuild the explicit per-cell state that beginTimeStep normally
    //! maintains (max oil/water saturation, rock-compaction trans multiplier).
    void finishAdaptExplicitQuantities()
    {
        this->updateExplicitQuantities_(/*first_step_after_restart=*/true);
    }

    //! Called AFTER gridChanged(): component-inventory conservation gate.
    void verifyAfterAdapt()
    {
        const auto invAfter = blackOilComponentInventory<TypeTag>(this->simulator());
        double worst = 0.0;
        const char* names[3] = {"water", "oil", "gas"};
        for (int c = 0; c < 3; ++c) {
            const double denom = std::max(std::abs(invBefore_[c]), 1e-300);
            const double rel = std::abs(invAfter[c] - invBefore_[c]) / denom;
            worst = std::max(worst, rel);
            OpmLog::info(fmt::format(
                "[alu-hadapt] inventory {:>5}: {: .8e} -> {: .8e}  rel {:.2e}",
                names[c], invBefore_[c], invAfter[c], rel));
        }
        double tol = 1e-9;
        if (const char* s = std::getenv("OPM_ALU_ADAPT_INVENTORY_TOL")) {
            const double v = std::atof(s);
            if (std::isfinite(v) && v > 0.0) tol = v;
        }
        if (worst > tol) {
            throw std::runtime_error(fmt::format(
                "[alu-hadapt] component-inventory conservation FAILED after adapt "
                "(worst rel {:.3e} > tol {:.1e}).", worst, tol));
        }
    }

private:
    //! Compressed leaf-cell indices that currently host at least one well
    //! connection (any status: shut wells keep their connection list). Used
    //! by EVERY marking driver (fixed test box now, the estimator from
    //! Phase 5 on) to veto refinement of well cells -- see class doc.
    std::unordered_set<int> wellProtectedCells_() const
    {
        std::unordered_set<int> cells;
        const auto& wells = this->wellModel();
        for (int w = 0; w < wells.numLocalWells(); ++w) {
            for (const auto& perf : wells.perfData(w)) {
                cells.insert(perf.cell_index);
            }
        }
        return cells;
    }

    bool aluAdaptTestBoxDone_ {false};
    std::array<double, 3> invBefore_ {0.0, 0.0, 0.0};
};

} // namespace Opm

#endif // OPM_FLOW_PROBLEM_ALU_ADAPT_HPP
