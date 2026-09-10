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

#include <opm/common/OpmLog/OpmLog.hpp>

#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/common/partitionset.hh>

#include <fmt/format.h>

#include <array>
#include <cstdlib>
#include <sstream>
#include <string>

namespace Opm {

/*!
 * \brief Black-oil Flow problem for native (non-dune-fem) in-place h-adaptivity
 *        on a locally-adaptive grid (dune-ALUGrid).
 *
 * Adds the adaptation hooks that FvBaseDiscretizationAdaptiveNative calls, on top
 * of FlowProblemBlackoil:
 *   - markForGridAdaptation(): the plug point for the a posteriori
 *     SpatialMarker. Until that lands (Phase 5) it returns 0, so
 *     --enable-grid-adaptation is a no-op plumbing path. It DELIBERATELY shadows
 *     MultiPhaseBaseProblem::markForGridAdaptation() (a saturation-variation
 *     heuristic that calls grid.mark() inside a per-phase loop and miscounts) --
 *     that heuristic returns later behind --adapt-indicator=saturation.
 *   - gridChanged(): rebuild problem-side geometry-dependent data after an
 *     in-place adapt. Phase 0: base behaviour only.
 */
template <class TypeTag>
class FlowProblemAluAdapt : public FlowProblemBlackoil<TypeTag>
{
    using Base = FlowProblemBlackoil<TypeTag>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;

public:
    explicit FlowProblemAluAdapt(Simulator& simulator)
        : Base(simulator)
    {}

    /*!
     * \brief Number of leaf cells marked for refinement/coarsening this step.
     *
     * PHASE 1: if OPM_ALU_ADAPT_TEST_BOX="i1 i2 j1 j2 k1 k2" (1-based inclusive)
     * is set, mark the leaf cells of that logical-Cartesian box for refinement,
     * ONCE, and return the count -- a fixed-box driver to exercise the
     * mesh-adaptation lifecycle before the estimator SpatialMarker (Phase 5)
     * and the conservative transfer (Phase 2) exist.
     *
     * Deliberately shadows MultiPhaseBaseProblem::markForGridAdaptation() (a
     * saturation-variation heuristic that marks inside a per-phase loop and
     * miscounts) -- that returns later behind --adapt-indicator=saturation.
     */
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

        Dune::MultipleCodimMultipleGeomTypeMapper<std::decay_t<decltype(gridView)>>
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
            "[alu-hadapt PHASE 1] OPM_ALU_ADAPT_TEST_BOX i{}-{} j{}-{} k{}-{} "
            "-> marked {} leaf cell(s) for refinement",
            b[0], b[1], b[2], b[3], b[4], b[5], n));
        return grid.comm().sum(n);
    }

    /*!
     * \brief Rebuild problem-side geometry-dependent data after an in-place
     *        grid adaptation.
     *
     * PHASE 1+: transmissibility, well connection->cell map + WI, cell
     * depths/thickness, threshold pressure, output projection map.
     */
    void gridChanged()
    {
        Base::gridChanged();
    }

private:
    bool aluAdaptTestBoxDone_ {false};
};

} // namespace Opm

#endif // OPM_FLOW_PROBLEM_ALU_ADAPT_HPP
