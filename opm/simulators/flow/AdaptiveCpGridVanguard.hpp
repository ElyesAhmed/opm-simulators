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
*/
/*!
 * \file
 * \copydoc Opm::AdaptiveCpGridVanguard
 */
#ifndef OPM_ADAPTIVE_CPGRID_VANGUARD_HPP
#define OPM_ADAPTIVE_CPGRID_VANGUARD_HPP

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/models/utils/parametersystem.hpp>

#include <opm/simulators/flow/AdaptiveLgr.hpp>
#include <opm/simulators/flow/CpGridVanguard.hpp>

#include <opm/grid/cpgrid/refinement/ConformingBlockBuilder.hpp>
#include <opm/grid/cpgrid/refinement/RefinementBuilder.hpp>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Opm {

/*!
 * \brief A CpGridVanguard that additionally performs adaptive local refinement
 *        on a deck that has no CARFIN, driven by the --adaptive-lgr parameter.
 *
 * This is the AdaptiveCpGrid demonstration as a *separate Vanguard* (used only by
 * the flow_blackoil_adaptive executable), so the standard CpGridVanguard and the
 * standard flow binaries are completely unchanged. After the coarse grid is
 * built, addLgrs() refines the requested region by reusing the exact same static
 * refinement machinery (CpGrid::addLgrsUpdateLeafView with the conforming
 * builder), so the resulting grid -- and the simulation -- match the equivalent
 * static-CARFIN run. Serial demonstration; the builder is constructed from the
 * input grid's (sanitized-at-construction) COORD/ZCORN.
 */
template <class TypeTag>
class AdaptiveCpGridVanguard : public CpGridVanguard<TypeTag>
{
    using Base = CpGridVanguard<TypeTag>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;

public:
    explicit AdaptiveCpGridVanguard(Simulator& simulator)
        : Base(simulator)
    {}

    static void registerParameters()
    {
        Base::registerParameters();
        Parameters::Register<Parameters::AdaptiveLgr>(
            "Adaptive local grid refinement on a deck with no CARFIN: one or more "
            "CARFIN-style boxes 'I1 I2 J1 J2 K1 K2 NX NY NZ' (1-based, "
            "';'-separated). Empty (default) => behaves like flow_blackoil.");
    }

    //! \brief The refinement spec to apply; overridable so a dynamic driver
    //! can substitute a new mark set on a mid-run rebuild.
    virtual std::string adaptiveLgrSpec() const
    { return Parameters::Get<Parameters::AdaptiveLgr>(); }

    //! \brief Add deck LGRs (base) then any --adaptive-lgr refinement.
    void addLgrs()
    {
        // Honour a deck CARFIN if present (an adaptive deck normally has none).
        Base::addLgrs();

        const std::string spec = this->adaptiveLgrSpec();
        if (spec.empty()) {
            return;
        }
        const auto boxes = parseAdaptiveLgrSpec(spec);
        if (boxes.empty()) {
            return;
        }

        // Build the conforming refinement builder from the coarse input grid's
        // corner-point description and register it for the refinement call.
        const auto& inputGrid = this->eclState().getInputGrid();
        const std::array<int,3> dims = inputGrid.getNXYZ();
        std::vector<int> actnum = inputGrid.getACTNUM();
        auto previous = Opm::Refinement::setBuilder(
            std::make_unique<Opm::Refinement::ConformingBlockBuilder>(
                dims, inputGrid.getCOORD(), inputGrid.getZCORN(), std::move(actnum)));

        std::vector<std::array<int,3>> cellsPerDim, startIJK, endIJK;
        std::vector<std::string> names;
        for (const auto& b : boxes) {
            cellsPerDim.push_back(b.cellsPerDim);
            startIJK.push_back(b.startIJK);
            endIJK.push_back(b.endIJK);
            names.push_back(b.name);
        }
        OpmLog::info("\nAdaptive refinement (--adaptive-lgr): refining "
                     + std::to_string(boxes.size())
                     + " box(es) on the coarse grid post-construction");
        try {
            this->grid_->addLgrsUpdateLeafView(cellsPerDim, startIJK, endIJK, names);
        }
        catch (...) {
            Opm::Refinement::setBuilder(std::move(previous));
            throw;
        }
        Opm::Refinement::setBuilder(std::move(previous));

        // Same post-refinement bookkeeping the base addLgrs() does for a deck
        // CARFIN: refresh the leaf view, rebuild the stale Cartesian->compressed
        // map (so wells map to the right leaf cell), refresh depths/thickness.
        this->updateGridView_();
        this->updateCartesianToCompressedMapping_();
        this->updateCellDepths_();
        this->updateCellThickness_();

        // Wells in the refined region (plan S6b): COMPDAT wells carry no
        // trajectory to replay, so synthesize an equivalent one from their
        // connection cells (coarse input-grid geometry -- the polyline is
        // purely geometric) and then re-derive every trajectory well's
        // connections against the refined leaf grid.
        if (this->grid_->maxLevel() > 0) {
            this->schedule().synthesizeWellTrajectories(
                [&inputGrid](std::size_t globalIdx)
                { return inputGrid.getCellCenter(globalIdx); },
                [&inputGrid](std::size_t globalIdx)
                { return inputGrid.getCellDims(globalIdx); });

            this->recomputeWellTrajectoriesInLgr_();
        }
    }
};

} // namespace Opm

#endif // OPM_ADAPTIVE_CPGRID_VANGUARD_HPP
