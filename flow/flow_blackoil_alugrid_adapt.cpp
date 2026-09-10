/*
  Copyright 2026, SINTEF Digital

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
 * \brief Black-oil Flow on a locally-adaptive dune-ALUGrid, with NATIVE
 *        (non-dune-fem) estimator-driven h-adaptivity.
 *
 * This is flow_blackoil_alugrid + a discretization that permits
 * --enable-grid-adaptation without pulling in dune-fem. PHASE 0: the adaptation
 * execution path is not yet implemented, so with --enable-grid-adaptation=true
 * the run is identical to =false as long as nothing is marked (the default),
 * and throws if the problem marks a cell. The CpGrid flow_blackoil* executables
 * are untouched and remain the regression references.
 */
#include <config.h>

#include <dune/alugrid/grid.hh>

// for equilgrid in writer -- include before eclgenericwriter_impl.hh (specializations)
#include <opm/grid/CpGrid.hpp>
#include <opm/grid/cpgrid/GridHelpers.hpp>

#include <opm/models/blackoil/blackoilconvectivemixingmodule.hh>
#include <opm/models/blackoil/blackoildiffusionmodule.hh>
#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>
#include <opm/models/discretization/common/fvbasediscretizationadaptivenative.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>

#include <opm/simulators/flow/Main.hpp>

// not explicitly instanced in the library
#include <opm/simulators/flow/AluGridVanguard.hpp>
#include <opm/simulators/flow/FlowProblemAluAdapt.hpp>
#include <opm/simulators/flow/CollectDataOnIORank_impl.hpp>
#include <opm/simulators/flow/EclGenericWriter_impl.hpp>
#include <opm/simulators/flow/FlowGenericProblem_impl.hpp>
#include <opm/simulators/flow/GenericThresholdPressure_impl.hpp>
#include <opm/simulators/flow/GenericTracerModel_impl.hpp>
#include <opm/simulators/flow/Transmissibility_impl.hpp>
#include <opm/simulators/flow/equil/InitStateEquil_impl.hpp>
#include <opm/simulators/utils/GridDataOutput_impl.hpp>

namespace Opm::Properties {

namespace TTag {

struct FlowProblemAluGridAdapt
{ using InheritsFrom = std::tuple<FlowProblem>; };

}

template<class TypeTag>
struct Grid<TypeTag, TTag::FlowProblemAluGridAdapt>
{
    static const int dim = 3;
#if HAVE_MPI
     using type = Dune::ALUGrid<dim, dim, Dune::cube, Dune::nonconforming, Dune::ALUGridMPIComm>;
#else
     using type = Dune::ALUGrid<dim, dim, Dune::cube, Dune::nonconforming, Dune::ALUGridNoComm>;
#endif
};

// alugrid needs cp grid as equilgrid
template<class TypeTag>
struct EquilGrid<TypeTag, TTag::FlowProblemAluGridAdapt>
{ using type = Dune::CpGrid; };

template<class TypeTag>
struct Vanguard<TypeTag, TTag::FlowProblemAluGridAdapt>
{ using type = Opm::AluGridVanguard<TypeTag>; };

// Adds the adaptation hooks (markForGridAdaptation / gridChanged) the native
// adaptive discretization calls.
template<class TypeTag>
struct Problem<TypeTag, TTag::FlowProblemAluGridAdapt>
{ using type = Opm::FlowProblemAluAdapt<TypeTag>; };

// The one real difference from flow_blackoil_alugrid: a discretization base that
// permits --enable-grid-adaptation with a native (non-fem) execution path.
template<class TypeTag>
struct BaseDiscretizationType<TypeTag, TTag::FlowProblemAluGridAdapt>
{ using type = FvBaseDiscretizationAdaptiveNative<TypeTag>; };

// Same TPFA stack as flow_blackoil (flow_blackoil_alugrid uses the element-context
// path, but the a posteriori estimator needs the TPFA linearizer / local residual).
template<class TypeTag>
struct Linearizer<TypeTag, TTag::FlowProblemAluGridAdapt>
{ using type = TpfaLinearizer<TypeTag>; };

template<class TypeTag>
struct LocalResidual<TypeTag, TTag::FlowProblemAluGridAdapt>
{ using type = BlackOilLocalResidualTPFA<TypeTag>; };

template<class TypeTag>
struct EnableDiffusion<TypeTag, TTag::FlowProblemAluGridAdapt>
{ static constexpr bool value = false; };

template<class TypeTag>
struct AvoidElementContext<TypeTag, TTag::FlowProblemAluGridAdapt>
{ static constexpr bool value = true; };

} // namespace Opm::Properties

namespace Opm {

template<>
class SupportsFaceTag<Dune::ALUGrid<3, 3, Dune::cube, Dune::nonconforming>>
    : public std::bool_constant<true>
{};

} // namespace Opm

int main(int argc, char** argv)
{
    using TypeTag = Opm::Properties::TTag::FlowProblemAluGridAdapt;
    auto mainObject = std::make_unique<Opm::Main>(argc, argv);
    auto ret = mainObject->runStatic<TypeTag>();
    // Destruct mainObject as the destructor calls MPI_Finalize!
    mainObject.reset();
    return ret;
}
