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

#include <config.h>
#include <ebos/equil/initstateequil.hh>
#include <ebos/equil/initstateequil_impl.hh>

#include <opm/grid/CpGrid.hpp>

#if HAVE_DUNE_ALUGRID
#include <dune/alugrid/grid.hh>
#endif

#if HAVE_DUNE_FEM
#include <dune/fem/gridpart/adaptiveleafgridpart.hh>
#include <dune/fem/gridpart/common/gridpart2gridview.hh>
#include <ebos/femcpgridcompat.hh>
#endif

namespace Opm {
namespace EQUIL {
namespace DeckDependent {

using MatLaw = EclMaterialLawManager<ThreePhaseMaterialTraits<double,0,1,2>>;
#define INSTANCE_COMP(Grid, GridView, Mapper) \
    template class InitialStateComputer<BlackOilFluidSystem<double>, \
                                        Grid, \
                                        GridView, \
                                        Mapper, \
                                        Dune::CartesianIndexMapper<Grid>>; \
    template InitialStateComputer<BlackOilFluidSystem<double>, \
                                  Grid, \
                                  GridView, \
                                  Mapper, \
                                  Dune::CartesianIndexMapper<Grid>>::\
        InitialStateComputer(MatLaw&, \
                             const EclipseState&, \
                             const Grid&, \
                             const GridView&, \
                             const Dune::CartesianIndexMapper<Grid>&, \
                             const double, \
                             const int, \
                             const bool);
using Grid = Dune::CpGrid;
using GridView = Dune::GridView<Dune::DefaultLeafGridViewTraits<Grid>>;
using Mapper = Dune::MultipleCodimMultipleGeomTypeMapper<GridView>;
INSTANCE_COMP(Grid, GridView, Mapper)

#if HAVE_DUNE_FEM
using GridViewFem = Dune::Fem::GridPart2GridViewImpl<
                                        Dune::Fem::AdaptiveLeafGridPart<
                                            Grid,
                                            Dune::PartitionIteratorType(4),
                                            false>>;
using MapperFem = Dune::MultipleCodimMultipleGeomTypeMapper<GridViewFem>;
INSTANCE_COMP(Grid, GridViewFem, MapperFem)
#endif // HAVE_DUNE_FEM

#if HAVE_DUNE_ALUGRID
#if HAVE_MPI
using ALUGridComm = Dune::ALUGridMPIComm;
#else
using ALUGridComm = Dune::ALUGridNoComm;
#endif //HAVE_MPI
using ALUGrid3CN = Dune::ALUGrid<3, 3, Dune::cube, Dune::nonconforming, ALUGridComm>;
using ALUGridView = Dune::GridView<Dune::ALU3dLeafGridViewTraits<const ALUGrid3CN, Dune::PartitionIteratorType(4)>>;
using ALUGridMapper = Dune::MultipleCodimMultipleGeomTypeMapper<ALUGridView>;
INSTANCE_COMP(ALUGrid3CN, ALUGridView, ALUGridMapper)
#if HAVE_DUNE_FEM
using MapperFemAluGrid = Dune::MultipleCodimMultipleGeomTypeMapper<Dune::Fem::GridPart2GridViewImpl<Dune::Fem::AdaptiveLeafGridPart<ALUGrid3CN> > >;
using GridViewFemAluGrid = Dune::Fem::GridPart2GridViewImpl<
                                     Dune::Fem::AdaptiveLeafGridPart<ALUGrid3CN>>;
INSTANCE_COMP(ALUGrid3CN, GridViewFemAluGrid, MapperFemAluGrid)
#endif // HAVE_DUNE_FEM
#endif // HAVE_DUNE_ALUGRID

} // namespace DeckDependent

namespace Details {
    template class PressureTable<BlackOilFluidSystem<double>,EquilReg>;
    template void verticalExtent<std::vector<int>,
                                 Dune::CollectiveCommunication<Dune::MPIHelper::MPICommunicator>>(
                                 const std::vector<int>&,
                                 const std::vector<std::pair<double,double>>&,
                                 const Dune::CollectiveCommunication<Dune::MPIHelper::MPICommunicator>&,
                                 std::array<double,2>&);

    using MatLaw = EclMaterialLawManager<ThreePhaseMaterialTraits<double,0,1,2>>;
    template class PhaseSaturations<MatLaw,BlackOilFluidSystem<double>,
                                    EquilReg,std::size_t>;

    template std::pair<double,double> cellZMinMax(const Dune::cpgrid::Entity<0>& element);
}

} // namespace EQUIL
} // namespace Opm
