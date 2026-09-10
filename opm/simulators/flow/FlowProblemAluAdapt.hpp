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
    {}

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

        unsigned n = 0;
        for (const auto& e : elements(gridView, Dune::Partitions::interior)) {
            const int cart = mapper.cartesianIndex(elemMapper.index(e));
            const int i = cart % dims[0];
            const int j = (cart / dims[0]) % dims[1];
            const int k = cart / (dims[0] * dims[1]);
            if (i + 1 >= b[0] && i + 1 <= b[1] &&
                j + 1 >= b[2] && j + 1 <= b[3] &&
                k + 1 >= b[4] && k + 1 <= b[5]) {
                grid.mark(1, e);
                ++n;
            }
        }
        aluAdaptTestBoxDone_ = true;
        OpmLog::info(fmt::format(
            "[alu-hadapt] OPM_ALU_ADAPT_TEST_BOX i{}-{} j{}-{} k{}-{} "
            "-> marked {} leaf cell(s) for refinement",
            b[0], b[1], b[2], b[3], b[4], b[5], n));
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

        this->transmissibilities_.update(/*global=*/true);

        this->referencePorosity_[1] = this->referencePorosity_[0];
        this->updateReferencePorosity_();
        this->rockFraction_[1] = this->rockFraction_[0];
        this->updateRockFraction_();
        this->updatePffDofData_();
        this->model().linearizer().updateDiscretizationParameters();
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
    bool aluAdaptTestBoxDone_ {false};
    std::array<double, 3> invBefore_ {0.0, 0.0, 0.0};
};

} // namespace Opm

#endif // OPM_FLOW_PROBLEM_ALU_ADAPT_HPP
