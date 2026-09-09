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
 * \brief Compile / instantiation test for APosterioriSpatialTemporalEstimator.
 *
 * Instantiates the estimator against the full black-oil TypeTag used by the
 * other simulator unit tests, forcing every member function to be compiled
 * against the real FlowProblemBlackoil / BlackOilModel API.  There is no
 * runtime assertion: linking the explicit instantiation is the test.
 */
#include <config.h>

#include "TestTypeTag.hpp"

#include <opm/simulators/flow/FlowProblemBlackoil.hpp>

#include <opm/models/discretization/common/tpfalinearizer.hh>
#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>

#include <opm/simulators/flow/APosterioriSpatialTemporalEstimator.hpp>

#include <type_traits>

#define BOOST_TEST_MODULE TestAPosterioriEstimatorCompile
#include <boost/test/unit_test.hpp>

// The estimator reads inter-cell connectivity from the TPFA linearizer
// (getNeighborInfo()), so instantiate it against a TPFA-assembly TypeTag --
// which is what the real `flow` executable uses.
namespace Opm::Properties {
namespace TTag {
struct APostEstimatorTestTypeTag { using InheritsFrom = std::tuple<TestTypeTag>; };
}
template<class TypeTag>
struct Linearizer<TypeTag, TTag::APostEstimatorTestTypeTag>
{ using type = TpfaLinearizer<TypeTag>; };
template<class TypeTag>
struct LocalResidual<TypeTag, TTag::APostEstimatorTestTypeTag>
{ using type = BlackOilLocalResidualTPFA<TypeTag>; };
} // namespace Opm::Properties

template class Opm::APosterioriSpatialTemporalEstimator<
    Opm::Properties::TTag::APostEstimatorTestTypeTag>;

BOOST_AUTO_TEST_CASE(Instantiates)
{
    using Estimator = Opm::APosterioriSpatialTemporalEstimator<
        Opm::Properties::TTag::APostEstimatorTestTypeTag>;
    static_assert(std::is_class_v<Estimator>);
    BOOST_CHECK(true);
}
