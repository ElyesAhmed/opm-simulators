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
#include <opm/simulators/flow/AdaptiveRefinementProtection.hpp>
#include <opm/simulators/flow/AdaptiveStateTransfer.hpp>
#include <opm/simulators/flow/AluAdaptTransfer.hpp>

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/input/eclipse/EclipseState/Phase.hpp>
#include <opm/input/eclipse/EclipseState/SimulationConfig/RockConfig.hpp>
#include <opm/input/eclipse/Units/Units.hpp>

#include <dune/grid/common/mcmgmapper.hh>
#include <dune/grid/common/partitionset.hh>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
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
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
    using ParentInventory =
        std::unordered_map<std::int64_t, std::array<double, 4>>;
    using InitialFluidState = typename std::decay_t<
        decltype(std::declval<Base&>().initialFluidStates())>::value_type;

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

        const auto& dims = this->simulator().vanguard()
            .cartesianIndexMapper().cartesianDimensions();
        const auto protectedCells = buildProtectedRefinementCells(
            this->simulator().vanguard().schedule(),
            {dims[0], dims[1], dims[2]},
            refinementProtectionHaloFromEnvironment());
        protectedCartesianCells_.insert(protectedCells.begin(), protectedCells.end());
    }

    void writeOutput(bool verbose) override
    {
        if (isReportStepEnd_()) {
            writeParentSaturation_();
        }
        Base::writeOutput(verbose);
    }

    unsigned markForGridAdaptation()
    {
        const char* spec = std::getenv("OPM_ALU_ADAPT_TEST_BOX");
        if (spec != nullptr && testBoxAdaptCount_ < testBoxAdaptLimit_()) {
            return markTestBox_(spec);
        }
        if (!estimatorEnergyReady_
            || estimatorAdaptCount_ >= estimatorAdaptLimit_()) {
            return 0;
        }
        return markEstimatorCells_();
    }

    bool setAposterioriSpatialEnergy(
        const std::vector<double>& energy,
        const std::vector<double>& waterComponentEnergy,
        const std::vector<double>& oilComponentEnergy,
        const std::vector<double>& gasComponentEnergy,
        const std::vector<double>& latestEnergy,
        const std::vector<double>& latestWaterComponentEnergy,
        const std::vector<double>& latestOilComponentEnergy,
        const std::vector<double>& latestGasComponentEnergy)
    {
        if (std::getenv("OPM_ALU_ESTIMATOR_THETA") == nullptr) {
            return false;
        }
        if (estimatorAdaptCount_ >= estimatorAdaptLimit_()) {
            return false;
        }
        ++acceptedEstimatorSteps_;
        const double completedDay = unit::convert::to(
            this->simulator().time() + this->simulator().timeStepSize(),
            unit::day);
        if (completedDay < estimatorStartDay_()) {
            return false;
        }
        if (completedDay < lastEstimatorSnapshotDay_ + estimatorMinEventDays_()) {
            return false;
        }
        if (acceptedEstimatorSteps_ % estimatorInterval_() != 0) {
            return false;
        }
        estimatorSpatialEnergy_ = energy;
        estimatorWaterComponentSpatialEnergy_ = waterComponentEnergy;
        estimatorOilComponentSpatialEnergy_ = oilComponentEnergy;
        estimatorGasComponentSpatialEnergy_ = gasComponentEnergy;
        latestEstimatorSpatialEnergy_ = latestEnergy;
        latestEstimatorWaterComponentSpatialEnergy_ = latestWaterComponentEnergy;
        latestEstimatorOilComponentSpatialEnergy_ = latestOilComponentEnergy;
        latestEstimatorGasComponentSpatialEnergy_ = latestGasComponentEnergy;
        estimatorEnergyReady_ = true;
        lastEstimatorSnapshotDay_ = completedDay;
        OpmLog::info(fmt::format(
            "[alu-hadapt] estimator snapshot accepted at day {:.6g} "
            "after {} accepted step(s)",
            completedDay, acceptedEstimatorSteps_));
        return true;
    }

private:
    unsigned markTestBox_(const char* spec)
    {
        if (this->simulator().vanguard().grid().comm().size() != 1) {
            throw std::logic_error(
                "Native ALUGrid adaptation is currently supported in serial only");
        }
        rejectUnsupportedAdaptation_();

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
                if (isProtectedForLevel_(cart, e.level(), dims)) {
                    ++vetoed;
                    continue;
                }
                grid.mark(1, e);
                ++n;
            }
        }
        ++testBoxAdaptCount_;
        OpmLog::info(fmt::format(
            "[alu-hadapt] OPM_ALU_ADAPT_TEST_BOX event {} of {}: "
            "i{}-{} j{}-{} k{}-{} "
            "-> marked {} leaf cell(s) for refinement ({} vetoed: protected cell)",
            testBoxAdaptCount_, testBoxAdaptLimit_(),
            b[0], b[1], b[2], b[3], b[4], b[5], n, vetoed));
        return grid.comm().sum(n);
    }

    //! Called BEFORE grid.adapt().
public:
    void prepareForAdapt()
    {
        invBefore_ = blackOilComponentInventory<TypeTag>(this->simulator());
        parentInvBefore_ = inventoryByCartesianParent_();
        initialFluidStatesBefore_.capture(this->simulator().gridView(),
            [this](std::size_t idx) { return this->initialFluidStates().at(idx); });
        if (!this->rockTableIdx_.empty()) {
            rockTableBefore_.capture(this->simulator().gridView(),
                [this](std::size_t idx) { return this->rockTableIdx_.at(idx); });
        }
        this->wellModel().prepareForGridAdapt();
        this->simulator().vanguard().snapshotForAdapt();
    }

    //! Called AFTER the grid + solution have been remapped: rebuild every
    //! geometry-dependent problem-side quantity (same recipe as the
    //! GEO_MODIFIER schedule-event path).
    void gridChanged()
    {
        // Nonthermal intensive quantities still obtain temperature from the
        // initial fluid state. Remap it before any new-grid PVT evaluation;
        // resizing alone would also lose the identity of surviving cells.
        const auto& gv = this->simulator().gridView();
        auto& initial = this->initialFluidStates();
        initial.resize(this->model().numGridDof());
        Dune::MultipleCodimMultipleGeomTypeMapper<GridView>
            mapper(gv, Dune::mcmgElementLayout());
        verifyProtectedCellsAfterAdapt_(gv, mapper);
        for (const auto& elem : elements(gv, Dune::Partitions::interior)) {
            initial.at(mapper.index(elem)) = initialFluidStatesBefore_.lookup(gv, elem);
        }
        // ROCK region lookup is also leaf-indexed and is used by the first
        // post-adapt porosity evaluation. readMaterialParameters_ does not
        // rebuild this array. Inherit the parent's region, preserving empty
        // storage as the existing single-default-region representation.
        if (!this->rockTableIdx_.empty()) {
            this->rockTableIdx_.resize(this->model().numGridDof());
            for (const auto& elem : elements(gv, Dune::Partitions::interior)) {
                this->rockTableIdx_.at(mapper.index(elem)) = rockTableBefore_.lookup(gv, elem);
            }
        }
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
        validateRefinedInterfaces_();

        // PffGridVector owns a value-copy of the GridView and its own element
        // mapper. Updating only its values after adapt leaves that mapper tied
        // to the pre-adapt leaf index set, which assigns valid
        // transmissibilities to the wrong residual rows. This is invisible to
        // the direct transmissibility/interface checks but destabilizes a
        // transport front as soon as it reaches a coarse/fine interface.
        this->refreshPffDofDataAfterGridAdapt_();
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
        // Force one full reallocation now. A non-substep call alone is not
        // sufficient when this report step does not write a restart file:
        // doAllocBuffers() then leaves alloc_fields false and retains the
        // pre-adapt vectors. The restart allocation path is the existing
        // mechanism for unconditionally sizing every active field buffer.
        this->eclWriter().mutableOutputModule().allocBuffers(
            static_cast<unsigned>(this->model().numGridDof()),
            static_cast<unsigned>(this->simulator().episodeIndex()),
            /*substep=*/false,
            /*log=*/false,
            /*isRestart=*/true);

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
            if (!std::isfinite(invBefore_[c]) || !std::isfinite(invAfter[c]) ||
                !std::isfinite(rel)) {
                throw std::runtime_error(fmt::format(
                    "[alu-hadapt] non-finite {} inventory across adapt: {} -> {}",
                    names[c], invBefore_[c], invAfter[c]));
            }
            worst = std::max(worst, rel);
            OpmLog::info(fmt::format(
                "[alu-hadapt] inventory {:>5}: {: .8e} -> {: .8e}  rel {:.2e}",
                names[c], invBefore_[c], invAfter[c], rel));
        }
        double tol = 1e-10;
        if (const char* s = std::getenv("OPM_ALU_ADAPT_INVENTORY_TOL")) {
            const double v = std::atof(s);
            if (std::isfinite(v) && v > 0.0) tol = v;
        }
        if (worst > tol) {
            throw std::runtime_error(fmt::format(
                "[alu-hadapt] component-inventory conservation FAILED after adapt "
                "(worst rel {:.3e} > tol {:.1e}).", worst, tol));
        }
        verifyPerParentConservation(parentInvBefore_,
                                    inventoryByCartesianParent_(),
                                    tol,
                                    /*throwOnFail=*/true);
        if (lastEstimatorLeafBudget_ > 0
            && this->model().numGridDof() > lastEstimatorLeafBudget_) {
            throw std::runtime_error(fmt::format(
                "[alu-hadapt] estimator leaf budget exceeded after closure: "
                "{} leaves > budget {}",
                this->model().numGridDof(), lastEstimatorLeafBudget_));
        }
        lastEstimatorLeafBudget_ = 0;
    }

private:
    unsigned markEstimatorCells_()
    {
        estimatorEnergyReady_ = false;
        auto& grid = this->simulator().vanguard().grid();
        if (grid.comm().size() != 1) {
            throw std::logic_error(
                "Native ALUGrid adaptation is currently supported in serial only");
        }
        rejectUnsupportedAdaptation_();

        const auto& gv = this->simulator().gridView();
        if (estimatorSpatialEnergy_.size() != gv.size(0)) {
            throw std::logic_error(fmt::format(
                "Estimator energy has {} cells but the current ALUGrid has {} leaves",
                estimatorSpatialEnergy_.size(), gv.size(0)));
        }

        const double theta = estimatorTheta_();
        const unsigned maxLevel = estimatorMaxLevel_();
        const std::size_t maxLeaves = estimatorMaxLeaves_();
        const std::size_t currentLeaves = gv.size(0);
        if (maxLeaves <= currentLeaves) {
            OpmLog::info(fmt::format(
                "[alu-hadapt] estimator marking skipped: leaf budget {} reached",
                maxLeaves));
            return 0;
        }
        const std::size_t parentBudget = (maxLeaves - currentLeaves) / 7;
        if (parentBudget == 0) {
            return 0;
        }

        const auto& cartMapper =
            this->simulator().vanguard().cartesianIndexMapper();
        const auto& dims = cartMapper.cartesianDimensions();
        Dune::MultipleCodimMultipleGeomTypeMapper<GridView>
            mapper(gv, Dune::mcmgElementLayout());

        std::vector<std::pair<double, unsigned>> candidates;
        candidates.reserve(currentLeaves);
        double totalEnergy = 0.0;
        for (const auto& elem : elements(gv, Dune::Partitions::interior)) {
            const unsigned idx = mapper.index(elem);
            const double energy = estimatorSpatialEnergy_[idx];
            if (!std::isfinite(energy) || energy < 0.0) {
                throw std::logic_error(fmt::format(
                    "Invalid accumulated estimator energy {} in leaf {}", energy, idx));
            }
            const int cart = cartMapper.cartesianIndex(idx);
            if (energy == 0.0 || static_cast<unsigned>(elem.level()) >= maxLevel
                || isProtectedForLevel_(cart, elem.level(), dims)) {
                continue;
            }
            candidates.emplace_back(energy, idx);
            totalEnergy += energy;
        }
        if (candidates.empty() || totalEnergy <= 0.0) {
            return 0;
        }
        std::ranges::sort(candidates,
            [](const auto& lhs, const auto& rhs) {
                return lhs.first > rhs.first;
            });

        std::vector<char> selected(currentLeaves, 0);
        double retainedEnergy = 0.0;
        std::size_t selectedCount = 0;
        for (const auto& [energy, idx] : candidates) {
            if (selectedCount >= parentBudget
                || retainedEnergy >= theta * totalEnergy) {
                break;
            }
            selected[idx] = 1;
            retainedEnergy += energy;
            ++selectedCount;
        }
        if (selectedCount == 0) {
            return 0;
        }

        dumpEstimatorMarking_(selected, mapper, cartMapper, dims);

        unsigned marked = 0;
        for (const auto& elem : elements(gv, Dune::Partitions::interior)) {
            if (selected[mapper.index(elem)]) {
                grid.mark(1, elem);
                ++marked;
            }
        }
        ++estimatorAdaptCount_;
        lastEstimatorLeafBudget_ = maxLeaves;
        const bool budgetLimited =
            selectedCount >= parentBudget
            && retainedEnergy < theta * totalEnergy;
        const unsigned adaptLimit = estimatorAdaptLimit_();
        const std::string adaptLimitLabel =
            adaptLimit == std::numeric_limits<unsigned>::max()
            ? "unlimited"
            : std::to_string(adaptLimit);
        OpmLog::info(fmt::format(
            "[alu-hadapt] estimator Dorfler event {} of {}: theta={:.3f}, "
            "{} of {} eligible leaves marked, retained {:.1f}% energy, "
            "leaf budget {} (current {})",
            estimatorAdaptCount_, adaptLimitLabel, theta,
            marked, candidates.size(), 100.0 * retainedEnergy / totalEnergy,
            maxLeaves, currentLeaves));
        if (budgetLimited) {
            OpmLog::info(fmt::format(
                "[alu-hadapt] estimator event {} budget limited: "
                "Dorfler target not reached because only {} parent slots remain",
                estimatorAdaptCount_, parentBudget));
        }
        return grid.comm().sum(marked);
    }

    template<class Mapper, class CartesianMapper, class Dimensions>
    void dumpEstimatorMarking_(const std::vector<char>& selected,
                               const Mapper& mapper,
                               const CartesianMapper& cartMapper,
                               const Dimensions& dims)
    {
        const char* path = std::getenv("OPM_ALU_MARK_FIELD_CSV");
        if (path == nullptr || *path == '\0') {
            return;
        }
        std::string outputPath(path);
        if (std::getenv("OPM_ALU_MARK_FIELD_SERIES") != nullptr) {
            const std::string suffix = fmt::format(
                "-event{:03d}.csv", estimatorAdaptCount_ + 1);
            const auto extension = outputPath.rfind(".csv");
            if (extension == std::string::npos)
                outputPath += suffix;
            else
                outputPath.replace(extension, 4, suffix);
        }

        constexpr int water = FluidSystem::waterPhaseIdx;
        constexpr int gas = FluidSystem::gasPhaseIdx;
        const bool waterActive = FluidSystem::phaseIsActive(water);
        const bool gasActive = FluidSystem::phaseIsActive(gas);
        std::ofstream output(outputPath, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "Unable to open estimator marking output file '" + outputPath + "'");
        }
        output << std::setprecision(17)
               << "event,time_days,leaf,cartesian,level,x,y,z,sw,sg,"
                  "accumulated_eta_sp,"
                  "accumulated_eta_sp_water_component,"
                  "accumulated_eta_sp_oil_component,"
                  "accumulated_eta_sp_gas_component,"
                  "step_eta_sp,step_eta_sp_water_component,"
                  "step_eta_sp_oil_component,step_eta_sp_gas_component,"
                  "selected,protected\n";

        ElementContext elemCtx(this->simulator());
        for (const auto& elem :
             elements(this->simulator().gridView(), Dune::Partitions::interior)) {
            const unsigned idx = mapper.index(elem);
            elemCtx.updatePrimaryStencil(elem);
            elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);
            const auto& iq = elemCtx.intensiveQuantities(0, /*timeIdx=*/0);
            const auto center = elem.geometry().center();
            const int cart = cartMapper.cartesianIndex(idx);
            const double sw = waterActive
                ? getValue(iq.fluidState().saturation(water)) : 0.0;
            const double sg = gasActive
                ? getValue(iq.fluidState().saturation(gas)) : 0.0;
            output << estimatorAdaptCount_ + 1 << ','
                   << lastEstimatorSnapshotDay_ << ','
                   << idx << ',' << cart << ',' << elem.level() << ','
                   << center[0] << ',' << center[1] << ',' << center[2] << ','
                   << sw << ',' << sg << ','
                   << std::sqrt(estimatorSpatialEnergy_[idx]) << ','
                   << (idx < estimatorWaterComponentSpatialEnergy_.size()
                           ? std::sqrt(estimatorWaterComponentSpatialEnergy_[idx])
                           : 0.0)
                   << ','
                   << (idx < estimatorOilComponentSpatialEnergy_.size()
                           ? std::sqrt(estimatorOilComponentSpatialEnergy_[idx])
                           : 0.0)
                   << ','
                   << (idx < estimatorGasComponentSpatialEnergy_.size()
                           ? std::sqrt(estimatorGasComponentSpatialEnergy_[idx])
                           : 0.0)
                   << ','
                   << (idx < latestEstimatorSpatialEnergy_.size()
                           ? std::sqrt(latestEstimatorSpatialEnergy_[idx])
                           : 0.0)
                   << ','
                   << (idx < latestEstimatorWaterComponentSpatialEnergy_.size()
                           ? std::sqrt(latestEstimatorWaterComponentSpatialEnergy_[idx])
                           : 0.0)
                   << ','
                   << (idx < latestEstimatorOilComponentSpatialEnergy_.size()
                           ? std::sqrt(latestEstimatorOilComponentSpatialEnergy_[idx])
                           : 0.0)
                   << ','
                   << (idx < latestEstimatorGasComponentSpatialEnergy_.size()
                           ? std::sqrt(latestEstimatorGasComponentSpatialEnergy_[idx])
                           : 0.0)
                   << ','
                   << static_cast<int>(selected[idx]) << ','
                   << static_cast<int>(isProtectedForLevel_(
                          cart, elem.level(), dims))
                   << '\n';
        }
    }

    double estimatorTheta_() const
    {
        const char* value = std::getenv("OPM_ALU_ESTIMATOR_THETA");
        const double theta = value ? std::atof(value) : -1.0;
        if (!(theta > 0.0 && theta < 1.0)) {
            throw std::invalid_argument(
                "OPM_ALU_ESTIMATOR_THETA must be strictly between 0 and 1");
        }
        return theta;
    }

    unsigned estimatorMaxLevel_() const
    {
        unsigned value = 1;
        if (const char* text = std::getenv("OPM_ALU_ADAPT_MAX_LEVEL")) {
            std::istringstream input(text);
            char trailing = '\0';
            if (!(input >> value) || (input >> trailing) || value == 0) {
                throw std::invalid_argument(
                    "OPM_ALU_ADAPT_MAX_LEVEL must be a positive integer");
            }
        }
        return value;
    }

    std::size_t estimatorMaxLeaves_() const
    {
        const char* text = std::getenv("OPM_ALU_ADAPT_MAX_LEAVES");
        std::size_t value = 0;
        std::istringstream input(text ? text : "");
        char trailing = '\0';
        if (!text || !(input >> value) || (input >> trailing) || value == 0) {
            throw std::invalid_argument(
                "OPM_ALU_ADAPT_MAX_LEAVES must be a positive integer "
                "when estimator marking is enabled");
        }
        return value;
    }

    unsigned estimatorInterval_() const
    {
        unsigned value = 1;
        if (const char* text = std::getenv("OPM_ALU_ESTIMATOR_EVERY")) {
            std::istringstream input(text);
            char trailing = '\0';
            if (!(input >> value) || (input >> trailing) || value == 0) {
                throw std::invalid_argument(
                    "OPM_ALU_ESTIMATOR_EVERY must be a positive integer");
            }
        }
        return value;
    }

    double estimatorStartDay_() const
    {
        double value = 0.0;
        if (const char* text = std::getenv("OPM_ALU_ESTIMATOR_START_DAY")) {
            std::istringstream input(text);
            char trailing = '\0';
            if (!(input >> value) || (input >> trailing)
                || !std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "OPM_ALU_ESTIMATOR_START_DAY must be a finite "
                    "non-negative number");
            }
        }
        return value;
    }

    double estimatorMinEventDays_() const
    {
        double value = 0.0;
        if (const char* text = std::getenv("OPM_ALU_ESTIMATOR_MIN_EVENT_DAYS")) {
            std::istringstream input(text);
            char trailing = '\0';
            if (!(input >> value) || (input >> trailing)
                || !std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "OPM_ALU_ESTIMATOR_MIN_EVENT_DAYS must be a finite "
                    "non-negative number");
            }
        }
        return value;
    }

    unsigned estimatorAdaptLimit_() const
    {
        unsigned value = 1;
        if (const char* text = std::getenv("OPM_ALU_ESTIMATOR_MAX_EVENTS")) {
            std::istringstream input(text);
            char trailing = '\0';
            if (!(input >> value) || (input >> trailing)) {
                throw std::invalid_argument(
                    "OPM_ALU_ESTIMATOR_MAX_EVENTS must be a non-negative "
                    "integer (zero means unlimited)");
            }
        }
        return value == 0 ? std::numeric_limits<unsigned>::max() : value;
    }

    void validateRefinedInterfaces_() const
    {
        if (std::getenv("OPM_ALU_VALIDATE_INTERFACES") == nullptr) {
            return;
        }

        struct PairData {
            unsigned visits {0};
            double areaSum {0.0};
            std::array<double, GridView::dimensionworld> normalSum {};
        };

        const auto& gv = this->simulator().gridView();
        const auto& cartMapper =
            this->simulator().vanguard().cartesianIndexMapper();
        Dune::MultipleCodimMultipleGeomTypeMapper<GridView>
            mapper(gv, Dune::mcmgElementLayout());
        std::unordered_map<std::uint64_t, PairData> pairs;
        std::unordered_map<int, std::decay_t<
            decltype(this->transmissibilities_.permeability(0))>> parentPerm;

        for (const auto& elem : elements(gv, Dune::Partitions::interior)) {
            const unsigned inside = mapper.index(elem);
            const int cart = cartMapper.cartesianIndex(inside);
            const auto& perm = this->transmissibilities_.permeability(inside);
            const auto [it, inserted] = parentPerm.emplace(cart, perm);
            if (!inserted) {
                for (int row = 0; row < GridView::dimensionworld; ++row) {
                    for (int col = 0; col < GridView::dimensionworld; ++col) {
                        if (it->second[row][col] != perm[row][col]) {
                            throw std::logic_error(fmt::format(
                                "Refined children of Cartesian parent {} inherited "
                                "different permeability tensors", cart));
                        }
                    }
                }
            }

            for (const auto& intersection : intersections(gv, elem)) {
                if (!intersection.neighbor()
                    || intersection.inside().level() == intersection.outside().level()) {
                    continue;
                }
                const unsigned outside = mapper.index(intersection.outside());
                const unsigned lo = std::min(inside, outside);
                const unsigned hi = std::max(inside, outside);
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(lo) << 32) | hi;
                auto& data = pairs[key];
                ++data.visits;

                const auto area = intersection.geometry().volume();
                const auto normal = intersection.centerUnitOuterNormal();
                if (!std::isfinite(area) || area <= 0.0) {
                    throw std::logic_error(
                        "Non-finite or non-positive 2:1 interface area");
                }
                data.areaSum += area;
                for (int d = 0; d < GridView::dimensionworld; ++d) {
                    data.normalSum[d] += area * normal[d];
                }

                const double forward =
                    this->transmissibilities_.transmissibility(inside, outside);
                const double reverse =
                    this->transmissibilities_.transmissibility(outside, inside);
                if (!std::isfinite(forward) || forward <= 0.0 || forward != reverse) {
                    throw std::logic_error(fmt::format(
                        "Invalid asymmetric 2:1 transmissibility between leaves {} and {}",
                        inside, outside));
                }
            }
        }

        if (pairs.empty()) {
            throw std::logic_error(
                "Interface validation requested but no 2:1 interfaces were found");
        }
        for (const auto& [key, data] : pairs) {
            double normalResidual2 = 0.0;
            for (const double value : data.normalSum) {
                normalResidual2 += value * value;
            }
            const double relativeNormalResidual =
                std::sqrt(normalResidual2) / std::max(data.areaSum, 1e-300);
            if (data.visits != 2 || relativeNormalResidual > 1e-10) {
                throw std::logic_error(fmt::format(
                    "Unpaired 2:1 interface {}: visits={}, relative normal residual={:.3e}",
                    key, data.visits, relativeNormalResidual));
            }
        }
        OpmLog::info(fmt::format(
            "[alu-hadapt] 2:1 interfaces: {} pairs verified "
            "(inherited permeability, positive symmetric transmissibility, "
            "opposite area normals)",
            pairs.size()));
    }

    bool isProtectedForLevel_(const int cart,
                              const int currentLevel,
                              const std::array<int, 3>& dims) const
    {
        const int ci = cart % dims[0];
        const int cj = (cart / dims[0]) % dims[1];
        const int ck = cart / (dims[0] * dims[1]);
        const int radius = std::max(currentLevel, 0);
        for (int dk = -radius; dk <= radius; ++dk) {
            const int k = ck + dk;
            if (k < 0 || k >= dims[2]) {
                continue;
            }
            for (int dj = -radius; dj <= radius; ++dj) {
                const int j = cj + dj;
                if (j < 0 || j >= dims[1]) {
                    continue;
                }
                for (int di = -radius; di <= radius; ++di) {
                    const int i = ci + di;
                    if (i < 0 || i >= dims[0]) {
                        continue;
                    }
                    const int neighbor = i + dims[0] * (j + dims[1] * k);
                    if (protectedCartesianCells_.count(neighbor) != 0) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    unsigned testBoxAdaptLimit_() const
    {
        unsigned limit = 1;
        if (const char* value = std::getenv("OPM_ALU_ADAPT_TEST_REPEATS")) {
            std::istringstream input(value);
            unsigned parsed = 0;
            char trailing = '\0';
            if ((input >> parsed) && !(input >> trailing) && parsed > 0) {
                limit = std::min(parsed, 10U);
            }
            else {
                throw std::invalid_argument(
                    "OPM_ALU_ADAPT_TEST_REPEATS must be an integer in [1,10]");
            }
        }
        return limit;
    }

    ParentInventory inventoryByCartesianParent_()
    {
        constexpr int water = FluidSystem::waterPhaseIdx;
        constexpr int oil = FluidSystem::oilPhaseIdx;
        constexpr int gas = FluidSystem::gasPhaseIdx;
        const bool waterActive = FluidSystem::phaseIsActive(water);
        const bool oilActive = FluidSystem::phaseIsActive(oil);
        const bool gasActive = FluidSystem::phaseIsActive(gas);

        ParentInventory result;
        ElementContext elemCtx(this->simulator());
        const auto& cartMapper = this->simulator().vanguard().cartesianIndexMapper();
        for (const auto& elem :
             elements(this->simulator().gridView(), Dune::Partitions::interior)) {
            elemCtx.updatePrimaryStencil(elem);
            elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);
            const auto& iq = elemCtx.intensiveQuantities(0, /*timeIdx=*/0);
            const unsigned idx = elemCtx.globalSpaceIndex(0, /*timeIdx=*/0);
            const auto& fs = iq.fluidState();
            const std::int64_t parent = cartMapper.cartesianIndex(idx);

            const double pv = this->simulator().model().dofTotalVolume(idx)
                            * getValue(iq.porosity());
            double surfaceWater = 0.0;
            double surfaceOil = 0.0;
            double surfaceGas = 0.0;
            if (waterActive) {
                surfaceWater = getValue(fs.saturation(water))
                             * getValue(fs.invB(water)) * pv;
            }
            if (oilActive) {
                surfaceOil = getValue(fs.saturation(oil))
                           * getValue(fs.invB(oil)) * pv;
            }
            if (gasActive) {
                surfaceGas = getValue(fs.saturation(gas))
                           * getValue(fs.invB(gas)) * pv;
            }

            auto& inventory = result[parent];
            inventory[0] += pv;
            inventory[1] += surfaceWater;
            inventory[2] += surfaceOil
                          + (oilActive && gasActive
                             ? getValue(fs.Rv()) * surfaceGas : 0.0);
            inventory[3] += surfaceGas
                          + (oilActive && gasActive
                             ? getValue(fs.Rs()) * surfaceOil : 0.0);
        }
        return result;
    }

    void writeParentSaturation_()
    {
        const char* path = std::getenv("OPM_ALU_PARENT_FIELD_CSV");
        if (path == nullptr || *path == '\0') {
            return;
        }

        constexpr int water = FluidSystem::waterPhaseIdx;
        constexpr int gas = FluidSystem::gasPhaseIdx;
        if (!FluidSystem::phaseIsActive(water)) {
            return;
        }
        const bool gasActive = FluidSystem::phaseIsActive(gas);

        // volume, pore volume, saturation moments, leaf count
        std::unordered_map<std::int64_t, std::array<double, 7>> parentData;
        ElementContext elemCtx(this->simulator());
        const auto& cartMapper = this->simulator().vanguard().cartesianIndexMapper();
        for (const auto& elem :
             elements(this->simulator().gridView(), Dune::Partitions::interior)) {
            elemCtx.updatePrimaryStencil(elem);
            elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);
            const auto& iq = elemCtx.intensiveQuantities(0, /*timeIdx=*/0);
            const unsigned idx = elemCtx.globalSpaceIndex(0, /*timeIdx=*/0);
            const double volume = this->simulator().model().dofTotalVolume(idx);
            const double poreVolume = volume * getValue(iq.porosity());
            const double sw = getValue(iq.fluidState().saturation(water));
            const double sg = gasActive
                ? getValue(iq.fluidState().saturation(gas)) : 0.0;

            auto& data = parentData[cartMapper.cartesianIndex(idx)];
            data[0] += volume;
            data[1] += poreVolume;
            data[2] += sw * volume;
            data[3] += sw * poreVolume;
            data[4] += sg * volume;
            data[5] += sg * poreVolume;
            data[6] += 1.0;
        }

        std::vector<std::int64_t> parents;
        parents.reserve(parentData.size());
        for (const auto& [parent, data] : parentData) {
            static_cast<void>(data);
            parents.push_back(parent);
        }
        std::ranges::sort(parents);

        std::ofstream output(
            path, parentFieldHeaderWritten_ ? std::ios::app : std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "Unable to open adaptive parent-field output file '" + std::string(path) + "'");
        }
        output << std::setprecision(17);
        if (!parentFieldHeaderWritten_) {
            output << "time_days,parent_cartesian,geometric_volume,pore_volume,"
                      "sw_volume_average,sw_pore_volume_average,"
                      "sg_volume_average,sg_pore_volume_average,leaf_count\n";
            parentFieldHeaderWritten_ = true;
        }

        const double time = unit::convert::to(
            this->simulator().time() + this->simulator().timeStepSize(),
            unit::day);
        for (const auto parent : parents) {
            const auto& data = parentData.at(parent);
            output << time << ',' << parent << ',' << data[0] << ',' << data[1] << ','
                   << data[2] / data[0] << ',' << data[3] / data[1] << ','
                   << data[4] / data[0] << ',' << data[5] / data[1] << ','
                   << static_cast<std::size_t>(data[6]) << '\n';
        }
    }

    bool isReportStepEnd_() const
    {
        const double completedTime =
            this->simulator().time() + this->simulator().timeStepSize();
        const double nextReportTime = this->simulator().vanguard().schedule()
            .seconds(this->simulator().episodeIndex() + 1);
        return (nextReportTime - completedTime)
            <= (2 * std::numeric_limits<float>::epsilon()) * nextReportTime;
    }

    void rejectUnsupportedAdaptation_() const
    {
        const auto& eclState = this->simulator().vanguard().eclState();
        const auto& runspec = eclState.runspec();
        const auto& phases = runspec.phases();
        const auto& rock = eclState.getSimulationConfig().rock_config();
        const int episode = std::max(0, this->simulator().episodeIndex());
        const auto& oilvap = this->simulator().vanguard().schedule()[episode].oilvap();
        using P = Phase;

        const char* unsupported = nullptr;
        if (this->materialLawManager()->hysteresisConfig().enableHysteresis()) {
            unsupported = "saturation-function hysteresis";
        }
        else if (eclState.aquifer().active()) {
            unsupported = "analytic or numerical aquifers";
        }
        else if (phases.active(P::SOLVENT)) {
            unsupported = "the solvent model";
        }
        else if (phases.active(P::POLYMER)) {
            unsupported = "the polymer model";
        }
        else if (phases.active(P::POLYMW)) {
            unsupported = "polymer molecular weight";
        }
        else if (phases.active(P::FOAM)) {
            unsupported = "the foam model";
        }
        else if (phases.active(P::BRINE)) {
            unsupported = "the brine model";
        }
        else if (phases.active(P::ZFRACTION)) {
            unsupported = "the z-fraction model";
        }
        else if (phases.active(P::ENERGY)) {
            unsupported = "the thermal or energy model";
        }
        else if (runspec.micp()) {
            unsupported = "the MICP model";
        }
        else if (runspec.co2Storage()) {
            unsupported = "CO2STORE";
        }
        else if (runspec.h2Storage()) {
            unsupported = "H2STORE";
        }
        else if (runspec.co2Sol()) {
            unsupported = "dissolved CO2";
        }
        else if (runspec.h2Sol()) {
            unsupported = "dissolved H2";
        }
        else if (!eclState.tracer().empty()) {
            unsupported = "passive tracers";
        }
        else if (rock.active() &&
                 (rock.hysteresis_mode() != RockConfig::Hysteresis::REVERS ||
                  rock.water_compaction())) {
            unsupported = "path-dependent rock compaction";
        }
        else if (oilvap.defined()) {
            unsupported = "VAPPARS, DRSDT, or DRVDT history controls";
        }

        if (unsupported != nullptr) {
            throw std::invalid_argument(fmt::format(
                "Native ALUGrid adaptation does not yet transfer {}; "
                "refusing to mutate the grid.", unsupported));
        }
    }

    void verifyProtectedCellsAfterAdapt_(
        const GridView& gridView,
        const Dune::MultipleCodimMultipleGeomTypeMapper<GridView>& mapper) const
    {
        std::unordered_map<int, unsigned> leafCount;
        const auto& cartMapper = this->simulator().vanguard().cartesianIndexMapper();
        for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
            const int cart = cartMapper.cartesianIndex(mapper.index(elem));
            if (protectedCartesianCells_.count(cart) != 0) {
                ++leafCount[cart];
            }
        }
        for (const auto& [cart, count] : leafCount) {
            if (count != 1) {
                throw std::runtime_error(fmt::format(
                    "Protected Cartesian cell {} produced {} leaf cells after adaptation",
                    cart, count));
            }
        }
    }

    AluAdaptSnapshot<GridView, InitialFluidState> initialFluidStatesBefore_;
    AluAdaptSnapshot<GridView, unsigned short> rockTableBefore_;
    std::unordered_set<int> protectedCartesianCells_;
    bool parentFieldHeaderWritten_ = false;
    ParentInventory parentInvBefore_;
    unsigned testBoxAdaptCount_ {0};
    std::vector<double> estimatorSpatialEnergy_;
    std::vector<double> estimatorWaterComponentSpatialEnergy_;
    std::vector<double> estimatorOilComponentSpatialEnergy_;
    std::vector<double> estimatorGasComponentSpatialEnergy_;
    std::vector<double> latestEstimatorSpatialEnergy_;
    std::vector<double> latestEstimatorWaterComponentSpatialEnergy_;
    std::vector<double> latestEstimatorOilComponentSpatialEnergy_;
    std::vector<double> latestEstimatorGasComponentSpatialEnergy_;
    std::size_t acceptedEstimatorSteps_ {0};
    double lastEstimatorSnapshotDay_ {
        -std::numeric_limits<double>::infinity()
    };
    std::size_t lastEstimatorLeafBudget_ {0};
    unsigned estimatorAdaptCount_ {0};
    bool estimatorEnergyReady_ {false};
    std::array<double, 3> invBefore_ {0.0, 0.0, 0.0};
};

} // namespace Opm

#endif // OPM_FLOW_PROBLEM_ALU_ADAPT_HPP
