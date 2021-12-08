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
/*!
 * \file
 *
 * \copydoc Opm::EclWriter
 */
#ifndef EWOMS_ECL_WRITER_HH
#define EWOMS_ECL_WRITER_HH

#include "collecttoiorank.hh"
#include "ecloutputblackoilmodule.hh"

#include <opm/parser/eclipse/Units/UnitSystem.hpp>

#include <opm/simulators/utils/ParallelRestart.hpp>
#include <opm/simulators/utils/DeferredLoggingErrorHelpers.hpp>

#include <ebos/eclgenericwriter.hh>

#if OPM_HAVE_DAMARIS
#include <Damaris.h>
#endif

#include <string>

namespace Opm::Properties {

template<class TypeTag, class MyTypeTag>
struct EnableEclOutput {
    using type = UndefinedProperty;
};
template<class TypeTag, class MyTypeTag>
struct EnableAsyncEclOutput {
    using type = UndefinedProperty;
};
template<class TypeTag, class MyTypeTag>
struct EclOutputDoublePrecision {
    using type = UndefinedProperty;
};
template<class TypeTag, class MyTypeTag>
struct EnableEsmry {
    using type = UndefinedProperty;
};

} // namespace Opm::Properties

namespace Opm {

namespace Action { class State; }
class EclipseIO;
class UDQState;

/*!
 * \ingroup EclBlackOilSimulator
 *
 * \brief Collects necessary output values and pass it to opm-output.
 *
 * Caveats:
 * - For this class to do do anything meaningful, you will have to
 *   have the OPM module opm-output.
 * - The only DUNE grid which is currently supported is Dune::CpGrid
 *   from the OPM module "opm-grid". Using another grid won't
 *   fail at compile time but you will provoke a fatal exception as
 *   soon as you try to write an ECL output file.
 * - This class requires to use the black oil model with the element
 *   centered finite volume discretization.
 */
template <class TypeTag>
class EclWriter : public EclGenericWriter<GetPropType<TypeTag, Properties::Grid>,
                                          GetPropType<TypeTag, Properties::EquilGrid>,
                                          GetPropType<TypeTag, Properties::GridView>,
                                          GetPropType<TypeTag, Properties::ElementMapper>,
                                          GetPropType<TypeTag, Properties::Scalar>>
{
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Vanguard = GetPropType<TypeTag, Properties::Vanguard>;
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using Grid = GetPropType<TypeTag, Properties::Grid>;
    using EquilGrid = GetPropType<TypeTag, Properties::EquilGrid>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using Element = typename GridView::template Codim<0>::Entity;
    using ElementMapper = GetPropType<TypeTag, Properties::ElementMapper>;
    using ElementIterator = typename GridView::template Codim<0>::Iterator;
    using BaseType = EclGenericWriter<Grid,EquilGrid,GridView,ElementMapper,Scalar>;

    enum { enableEnergy = getPropValue<TypeTag, Properties::EnableEnergy>() };
    enum { enableTemperature = getPropValue<TypeTag, Properties::EnableTemperature>() };
    enum { enableSolvent = getPropValue<TypeTag, Properties::EnableSolvent>() };

public:
    static void registerParameters()
    {
        EclOutputBlackOilModule<TypeTag>::registerParameters();

        EWOMS_REGISTER_PARAM(TypeTag, bool, EnableAsyncEclOutput,
                             "Write the ECL-formated results in a non-blocking way (i.e., using a separate thread).");
        EWOMS_REGISTER_PARAM(TypeTag, bool, EnableEsmry,
                             "Write ESMRY file for fast loading of summary data.");
    }

    // The Simulator object should preferably have been const - the
    // only reason that is not the case is due to the SummaryState
    // object owned deep down by the vanguard.
    EclWriter(Simulator& simulator)
        : BaseType(simulator.vanguard().schedule(),
                   simulator.vanguard().eclState(),
                   simulator.vanguard().summaryConfig(),
                   simulator.vanguard().grid(),
                   simulator.vanguard().grid().comm().rank() == 0 ? &simulator.vanguard().equilGrid() : nullptr,
                   simulator.vanguard().gridView(),
                   simulator.vanguard().cartesianIndexMapper(),
                   simulator.vanguard().grid().comm().rank() == 0 ? &simulator.vanguard().equilCartesianIndexMapper() : nullptr,
                   EWOMS_GET_PARAM(TypeTag, bool, EnableAsyncEclOutput), EWOMS_GET_PARAM(TypeTag, bool, EnableEsmry))
        , simulator_(simulator)
    {
#if OPM_HAVE_DAMARIS      
        this->damarisUpdate = true ; 
#endif         
        this->eclOutputModule_ = std::make_unique<EclOutputBlackOilModule<TypeTag>>(simulator, this->wbp_index_list_, this->collectToIORank_);
        this->wbp_index_list_.clear();
    }

    ~EclWriter()
    { }

    const EquilGrid& globalGrid() const
    {
        return simulator_.vanguard().equilGrid();
    }

    /*!
     * \brief collect and pass data and pass it to eclIO writer
     */
    void evalSummaryState(bool isSubStep)
    {
        const int reportStepNum = simulator_.episodeIndex() + 1;
        /*
          The summary data is not evaluated for timestep 0, that is
          implemented with a:

             if (time_step == 0)
                 return;

          check somewhere in the summary code. When the summary code was
          split in separate methods Summary::eval() and
          Summary::add_timestep() it was necessary to pull this test out
          here to ensure that the well and group related keywords in the
          restart file, like XWEL and XGRP were "correct" also in the
          initial report step.

          "Correct" in this context means unchanged behavior, might very
          well be more correct to actually remove this if test.
        */
        if (reportStepNum == 0)
            return;

        const Scalar curTime = simulator_.time() + simulator_.timeStepSize();
        const Scalar totalCpuTime =
            simulator_.executionTimer().realTimeElapsed() +
            simulator_.setupTimer().realTimeElapsed() +
            simulator_.vanguard().externalSetupTime();

        const auto localWellData            = simulator_.problem().wellModel().wellData();
        const auto localGroupAndNetworkData = simulator_.problem().wellModel()
            .groupAndNetworkData(reportStepNum);


        const auto localAquiferData = simulator_.problem().aquiferModel().aquiferData();
        const auto localWellTestState = simulator_.problem().wellModel().wellTestState();
        this->prepareLocalCellData(isSubStep, reportStepNum);

        if (this->collectToIORank_.isParallel())
            this->collectToIORank_.collect({},
                                           eclOutputModule_->getBlockData(),
                                           eclOutputModule_->getWBPData(),
                                           localWellData,
                                           localGroupAndNetworkData,
                                           localAquiferData,
                                           localWellTestState);


        std::map<std::string, double> miscSummaryData;
        std::map<std::string, std::vector<double>> regionData;
        auto inplace = eclOutputModule_->outputFipLog(miscSummaryData, regionData, isSubStep, simulator_.gridView().comm());
        eclOutputModule_->outputFipresvLog(miscSummaryData, regionData, isSubStep, simulator_.gridView().comm());

        bool forceDisableProdOutput = false;
        bool forceDisableInjOutput = false;
        bool forceDisableCumOutput = false;

        // Add TCPU
        if (totalCpuTime != 0.0) {
            miscSummaryData["TCPU"] = totalCpuTime;
        }

        this->evalSummary(reportStepNum, curTime,
                          this->collectToIORank_.isParallel() ?
                            this->collectToIORank_.globalWBPData() :
                            this->eclOutputModule_->getWBPData(),
                          localWellData,
                          localGroupAndNetworkData,
                          localAquiferData,
                          this->collectToIORank_.isParallel() ?
                            this->collectToIORank_.globalBlockData() :
                            this->eclOutputModule_->getBlockData(),
                          miscSummaryData, regionData,
                          summaryState(), udqState(),
                          inplace,
                          eclOutputModule_->initialInplace());

        eclOutputModule_->outputProdLog(reportStepNum, isSubStep, forceDisableProdOutput);
        eclOutputModule_->outputInjLog(reportStepNum, isSubStep, forceDisableInjOutput);
        eclOutputModule_->outputCumLog(reportStepNum, isSubStep, forceDisableCumOutput);
    }

    void writeOutput(bool isSubStep)
    {
#if OPM_HAVE_DAMARIS      
        using DataEntry = std::tuple<std::string,
                                 UnitSystem::measure,
                                 data::TargetType,
                                 const std::vector<Scalar>&>;
                                 
        if (this->damarisUpdate == true)
        {
            int damaris_err = DAMARIS_OK;
            
            const int nranks = simulator_.vanguard().grid().comm().size() ;
            const int rank   = simulator_.vanguard().grid().comm().rank() ;
          
            std::vector<unsigned long long> elements_rank_sizes(nranks);    // one for each rank // to be gathered from each client rank
            std::vector<unsigned long long> elements_rank_offsets(nranks);  // one for each rank, first one 0 // to be computed - Probably could use MPI_Scan()?
            
            const auto& gridView = simulator_.vanguard().gridView();
            const int n_elements_local_grid = gridView.size(/*codim=*/0);  // I think this might be the full model size?
            const int n_elements_local_vector = this->collectToIORank_.getPRESSURE_size() ;
            const int n_elements_local = n_elements_local_vector ;
            
            std::cout << "INFO: n_elements_local_grid   = " << n_elements_local_grid << std::endl ;
            std::cout << "INFO: n_elements_local_vector = " << n_elements_local_vector << std::endl ;




            // This gets the n_elements_local from all ranks and copies them to a vector of all the values on all ranks (elements_rank_sizes[]).
            // MPI_Allgather(&n_elements_local, 1, MPI_UNSIGNED_LONG, elements_rank_sizes, 1, MPI_UNSIGNED_LONG, w->damaris_mpi_comm);
            simulator_.vanguard().grid().comm().allgather(n_elements_local, 1, elements_rank_sizes);
            elements_rank_offsets[0] = 0ULL ;  // 
            for (int t1 = 1 ; t1 < nranks; t1++)
            {
                elements_rank_offsets[t1] = elements_rank_offsets[t1-1] + elements_rank_sizes[t1-1];
            }
            
            if (rank == 0 ) {
               int n_elements_global_max  = elements_rank_offsets[nranks-1] + n_elements_local ;
                std::cout << "INFO: n_elements_global_max = " << n_elements_global_max << std::endl ;
            }
            
            
            // Set the paramater so that the Damaris servers can allocate the correct amount of memory for the variabe
            // Damaris parameters only support int data types. This will limit models to be under size of 2^32-1 elements
            int temp_int = static_cast<int>(elements_rank_sizes[rank]) ;
            damaris_err = damaris_parameter_set("n_elements_local",&temp_int, sizeof(int));
            if (damaris_err != DAMARIS_OK ) {
                 std::cerr << "ERROR: Damaris library produced an error result for damaris_parameter_set(n_elements_local,&temp_int, sizeof(int));" << std::endl ;
            }
            
            temp_int = static_cast<int>(n_elements_local_grid) ;
            damaris_err = damaris_parameter_set("n_elements_total",&temp_int, sizeof(int));
            if (damaris_err != DAMARIS_OK ) {
                 std::cerr << "ERROR: Damaris library produced an error result for damaris_parameter_set(n_elements_local,&temp_int, sizeof(int));" << std::endl ;
            }
            
            // Use damaris_set_position to set the offset in the global size of the array.
            // This is used so that output functionality (e.g. HDF5Store) knows global offsets of the data of the ranks
            int64_t temp_int64_t[1] ;
            temp_int64_t[0] = static_cast<int64_t>(elements_rank_offsets[rank]) ;
            damaris_err = damaris_set_position("PRESSURE",temp_int64_t) ;
            if (damaris_err != DAMARIS_OK ) {
                 std::cerr << "ERROR: Damaris library produced an error result for damaris_set_position(\"PRESSURE\",temp_int64_t);" << std::endl ;
            }  
            this->damarisUpdate = false ;
        }
        // now to find the field data
        
        // this->collectToIORank_.getPRESSURE_ptr() ;
        
        damaris_write("PRESSURE", (void *) this->collectToIORank_.getPRESSURE_ptr() ) ;
 
#else          
        const int reportStepNum = simulator_.episodeIndex() + 1;

        this->prepareLocalCellData(isSubStep, reportStepNum);
        this->eclOutputModule_->outputErrorLog(simulator_.gridView().comm());

        // output using eclWriter if enabled
        auto localWellData = simulator_.problem().wellModel().wellData();
        auto localGroupAndNetworkData = simulator_.problem().wellModel()
            .groupAndNetworkData(reportStepNum);

        auto localAquiferData = simulator_.problem().aquiferModel().aquiferData();
        auto localWellTestState = simulator_.problem().wellModel().wellTestState();

        data::Solution localCellData = {};
        if (! isSubStep) {
            this->eclOutputModule_->assignToSolution(localCellData);

            // add cell data to perforations for Rft output
            this->eclOutputModule_->addRftDataToWells(localWellData, reportStepNum);
        }

        if (this->collectToIORank_.isParallel()) {
            this->collectToIORank_.collect(localCellData,
                                           eclOutputModule_->getBlockData(),
                                           eclOutputModule_->getWBPData(),
                                           localWellData,
                                           localGroupAndNetworkData,
                                           localAquiferData,
                                           localWellTestState);
        }

        if (this->collectToIORank_.isIORank()) {
            const Scalar curTime = simulator_.time() + simulator_.timeStepSize();
            const Scalar nextStepSize = simulator_.problem().nextTimeStepSize();
            this->doWriteOutput(reportStepNum, isSubStep,
                                std::move(localCellData),
                                std::move(localWellData),
                                std::move(localGroupAndNetworkData),
                                std::move(localAquiferData),
                                std::move(localWellTestState),
                                this->actionState(),
                                this->udqState(),
                                this->summaryState(),
                                simulator_.problem().thresholdPressure().data(),
                                curTime, nextStepSize,
                                EWOMS_GET_PARAM(TypeTag, bool, EclOutputDoublePrecision));
        }
#endif         
    }

    void beginRestart()
    {
        bool enableHysteresis = simulator_.problem().materialLawManager()->enableHysteresis();
        bool enableSwatinit = simulator_.vanguard().eclState().fieldProps().has_double("SWATINIT");
        bool opm_rst_file = EWOMS_GET_PARAM(TypeTag, bool, EnableOpmRstFile);
        bool read_temp = enableEnergy || (opm_rst_file && enableTemperature);
        std::vector<RestartKey> solutionKeys{
            {"PRESSURE", UnitSystem::measure::pressure},
            {"SWAT", UnitSystem::measure::identity, static_cast<bool>(FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx))},
            {"SGAS", UnitSystem::measure::identity, static_cast<bool>(FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx))},
            {"TEMP" , UnitSystem::measure::temperature, read_temp},
            {"SSOLVENT" , UnitSystem::measure::identity, enableSolvent},
            {"RS", UnitSystem::measure::gas_oil_ratio, FluidSystem::enableDissolvedGas()},
            {"RV", UnitSystem::measure::oil_gas_ratio, FluidSystem::enableVaporizedOil()},
            {"SOMAX", UnitSystem::measure::identity, simulator_.problem().vapparsActive(simulator_.episodeIndex())},
            {"PCSWM_OW", UnitSystem::measure::identity, enableHysteresis},
            {"KRNSW_OW", UnitSystem::measure::identity, enableHysteresis},
            {"PCSWM_GO", UnitSystem::measure::identity, enableHysteresis},
            {"KRNSW_GO", UnitSystem::measure::identity, enableHysteresis},
            {"PPCW", UnitSystem::measure::pressure, enableSwatinit}
        };

        const auto& inputThpres = eclState().getSimulationConfig().getThresholdPressure();
        std::vector<RestartKey> extraKeys = {{"OPMEXTRA", UnitSystem::measure::identity, false},
                                             {"THRESHPR", UnitSystem::measure::pressure, inputThpres.active()}};

        {
            const auto& tracers = simulator_.vanguard().eclState().tracer();
            for (const auto& tracer : tracers)
                solutionKeys.emplace_back(tracer.fname(), UnitSystem::measure::identity, true);
        }

        // The episodeIndex is rewined one back before beginRestart is called
        // and can not be used here.
        // We just ask the initconfig directly to be sure that we use the correct
        // index.
        const auto& initconfig = simulator_.vanguard().eclState().getInitConfig();
        int restartStepIdx = initconfig.getRestartStep();

        const auto& gridView = simulator_.vanguard().gridView();
        unsigned numElements = gridView.size(/*codim=*/0);
        eclOutputModule_->allocBuffers(numElements, restartStepIdx, /*isSubStep=*/false, /*log=*/false, /*isRestart*/ true);

        {
            SummaryState& summaryState = simulator_.vanguard().summaryState();
            Action::State& actionState = simulator_.vanguard().actionState();
            auto restartValues = loadParallelRestart(this->eclIO_.get(), actionState, summaryState, solutionKeys, extraKeys,
                                                     gridView.grid().comm());
            for (unsigned elemIdx = 0; elemIdx < numElements; ++elemIdx) {
                unsigned globalIdx = this->collectToIORank_.localIdxToGlobalIdx(elemIdx);
                eclOutputModule_->setRestart(restartValues.solution, elemIdx, globalIdx);
            }

            auto& tracer_model = simulator_.problem().tracerModel();
            for (int tracer_index = 0; tracer_index < tracer_model.numTracers(); tracer_index++) {
                const auto& tracer_name = tracer_model.fname(tracer_index);
                const auto& tracer_solution = restartValues.solution.data(tracer_name);
                for (unsigned elemIdx = 0; elemIdx < numElements; ++elemIdx) {
                    unsigned globalIdx = this->collectToIORank_.localIdxToGlobalIdx(elemIdx);
                    tracer_model.setTracerConcentration(tracer_index, globalIdx, tracer_solution[globalIdx]);
                }
            }

            if (inputThpres.active()) {
                Simulator& mutableSimulator = const_cast<Simulator&>(simulator_);
                auto& thpres = mutableSimulator.problem().thresholdPressure();
                const auto& thpresValues = restartValues.getExtra("THRESHPR");
                thpres.setFromRestart(thpresValues);
            }
            restartTimeStepSize_ = restartValues.getExtra("OPMEXTRA")[0];

            // initialize the well model from restart values
            simulator_.problem().wellModel().initFromRestartFile(restartValues);

            if (!restartValues.aquifer.empty())
                simulator_.problem().mutableAquiferModel().initFromRestart(restartValues.aquifer);
        }
    }

    void endRestart()
    {}

    const EclOutputBlackOilModule<TypeTag>& eclOutputModule() const
    { return *eclOutputModule_; }

    Scalar restartTimeStepSize() const
    { return restartTimeStepSize_; }

private:
    static bool enableEclOutput_()
    { return EWOMS_GET_PARAM(TypeTag, bool, EnableEclOutput); }

    const EclipseState& eclState() const
    { return simulator_.vanguard().eclState(); }

    SummaryState& summaryState()
    { return simulator_.vanguard().summaryState(); }

    Action::State& actionState()
    { return simulator_.vanguard().actionState(); }

    UDQState& udqState()
    { return simulator_.vanguard().udqState(); }

    const Schedule& schedule() const
    { return simulator_.vanguard().schedule(); }

    void prepareLocalCellData(const bool isSubStep,
                              const int  reportStepNum)
    {
        const auto& gridView = simulator_.vanguard().gridView();
        const int numElements = gridView.size(/*codim=*/0);
        const bool log = this->collectToIORank_.isIORank();

        eclOutputModule_->allocBuffers(numElements, reportStepNum,
                                      isSubStep, log, /*isRestart*/ false);

        ElementContext elemCtx(simulator_);
        ElementIterator elemIt = gridView.template begin</*codim=*/0>();

        const ElementIterator& elemEndIt = gridView.template end</*codim=*/0>();
        OPM_BEGIN_PARALLEL_TRY_CATCH();
        for (; elemIt != elemEndIt; ++elemIt) {
            const Element& elem = *elemIt;

            elemCtx.updatePrimaryStencil(elem);
            elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);

            eclOutputModule_->processElement(elemCtx);
        }
        OPM_END_PARALLEL_TRY_CATCH("EclWriter::prepareLocalCellData() failed: ", simulator_.vanguard().grid().comm())
    }

    Simulator& simulator_;
    std::unique_ptr<EclOutputBlackOilModule<TypeTag>> eclOutputModule_;
    Scalar restartTimeStepSize_;
#if OPM_HAVE_DAMARIS
    bool damarisUpdate ;  ///< Whenever this is true writeOutput() will set up Damris offsets of model fields
#endif
};
} // namespace Opm

#endif
