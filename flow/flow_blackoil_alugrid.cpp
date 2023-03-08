/*
  Copyright 2020, NORCE AS

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
#include "config.h"
#include <opm/simulators/flow/Main.hpp>
#include <opm/models/common/transfluxmodule.hh>
#include <opm/models/discretization/ecfv/ecfvdiscretization.hh>

namespace Opm {
namespace Properties {
namespace TTag {
struct EclFlowProblemAlugrid {
    using InheritsFrom = std::tuple<EclFlowProblem>;
};
}
template<class TypeTag>
struct EclEnableAquifers<TypeTag, TTag::EclFlowProblemAlugrid> {
    static constexpr bool value = false;
};

// Set the problem property
template <class TypeTag>
struct FluxModule<TypeTag, TTag::EclFlowProblemAlugrid> {
    using type = Opm::TransFluxModule<TypeTag>;
};
//template<class TypeTag>
//struct GradientCalculator<TypeTag, TTag::FvBaseDiscretization> { 
//    using type = FvBaseGradientCalculator<TypeTag>; };

//}
//template<class TypeTag>
//struct GradientCalculator<TypeTag, TTag::EclFlowProblemAlugrid> { 
//    using type = FvBaseGradientCalculator<TypeTag>; 
//};
// use automatic differentiation for this simulator
template<class TypeTag>
struct LocalLinearizerSplice<TypeTag, TTag::EclFlowProblemAlugrid> { using type = TTag::AutoDiffLocalLinearizer; };
// use the element centered finite volume spatial discretization
template<class TypeTag>
struct SpatialDiscretizationSplice<TypeTag, TTag::EclFlowProblemAlugrid> { using type = TTag::EcfvDiscretization; };
// By default, include the intrinsic permeability tensor to the VTK output files
template<class TypeTag>
struct VtkWriteIntrinsicPermeabilities<TypeTag, TTag::EclFlowProblemAlugrid> { static constexpr bool value = false; };

// enable the storage cache by default for this problem
template<class TypeTag>
struct EnableStorageCache<TypeTag, TTag::EclFlowProblemAlugrid> { static constexpr bool value = false; };

// enable the cache for intensive quantities by default for this problem
template<class TypeTag>
struct EnableIntensiveQuantityCache<TypeTag, TTag::EclFlowProblemAlugrid> { static constexpr bool value = true; };
// this problem works fine if the linear solver uses single precision scalars
template<class TypeTag>
struct LinearSolverScalar<TypeTag, TTag::EclFlowProblemAlugrid> { using type = float; };
};

}
int main(int argc, char** argv)
{
    using TypeTag = Opm::Properties::TTag::EclFlowProblemAlugrid;
    auto mainObject = Opm::Main(argc, argv);
    return mainObject.runStatic<TypeTag>();
}
