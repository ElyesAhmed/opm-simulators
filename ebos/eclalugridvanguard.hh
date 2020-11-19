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
 * \copydoc Opm::EclAluGridVanguard
 */
#ifndef EWOMS_ECL_ALU_GRID_VANGUARD_HH
#define EWOMS_ECL_ALU_GRID_VANGUARD_HH

#include "eclbasevanguard.hh"
#include "ecltransmissibility.hh"
#include "alucartesianindexmapper.hh"

#include <dune/alugrid/grid.hh>
#include <dune/alugrid/common/fromtogridfactory.hh>
#include <opm/grid/CpGrid.hpp>
#include <opm/simulators/utils/ParallelEclipseState.hpp>
#include <opm/simulators/utils/PropsCentroidsDataHandle.hpp>

#include <dune/grid/common/mcmgmapper.hh>


namespace Opm {
template <class TypeTag>
class EclAluGridVanguard;

} // namespace Opm

namespace Opm::Properties {

namespace TTag {
struct EclAluGridVanguard {
    using InheritsFrom = std::tuple<EclBaseVanguard>;
};
}

// declare the properties
template<class TypeTag>
struct Vanguard<TypeTag, TTag::EclAluGridVanguard> {
    using type = Opm::EclAluGridVanguard<TypeTag>;
};
template<class TypeTag>
struct Grid<TypeTag, TTag::EclAluGridVanguard> {
    using type = Dune::ALUGrid<3, 3, Dune::cube, Dune::nonconforming>;
};
template<class TypeTag>
struct EquilGrid<TypeTag, TTag::EclAluGridVanguard> {
    using type = Dune::CpGrid;
};

} // namespace Opm::Properties

namespace Opm {

/*!
 * \ingroup EclBlackOilSimulator
 *
 * \brief Helper class for grid instantiation of ECL file-format using problems.
 *
 * This class uses Dune::ALUGrid as the simulation grid.
 */
template <class TypeTag>
class EclAluGridVanguard : public EclBaseVanguard<TypeTag>
{
    friend class EclBaseVanguard<TypeTag>;
    typedef EclBaseVanguard<TypeTag> ParentType;

    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;

public:
    using Grid = GetPropType<TypeTag, Properties::Grid>;
    using EquilGrid = GetPropType<TypeTag, Properties::EquilGrid>;
    using GridView = GetPropType<TypeTag, Properties::GridView>;

private:
    typedef Opm::AluCartesianIndexMapper<Grid> CartesianIndexMapper;
    typedef Dune::CartesianIndexMapper<EquilGrid> EquilCartesianIndexMapper;

    static const int dimension = Grid::dimension;

public:
    EclAluGridVanguard(Simulator& simulator)
        : EclBaseVanguard<TypeTag>(simulator), mpiRank()
    {
        this->callImplementationInit();
    }

    ~EclAluGridVanguard()
    {
        //delete cartesianIndexMapper_;
        delete equilCartesianIndexMapper_;
        delete grid_;
        delete equilGrid_;
    }

    /*!
     * \brief Return a reference to the simulation grid.
     */
    Grid& grid()
    { return *grid_; }

    /*!
     * \brief Return a reference to the simulation grid.
     */
    const Grid& grid() const
    { return *grid_; }

    /*!
     * \brief Returns a refefence to the grid which should be used by the EQUIL
     *        initialization code.
     *
     * The EQUIL keyword is used to specify the initial condition of the reservoir in
     * hydrostatic equilibrium. Since the code which does this is not accepting arbitrary
     * DUNE grids (the code is part of the opm-core module), this is not necessarily the
     * same as the grid which is used for the actual simulation.
     */
    const EquilGrid& equilGrid() const
    { return *equilGrid_; }

    /*!
     * \brief Indicates that the initial condition has been computed and the memory used
     *        by the EQUIL grid can be released.
     *
     * Depending on the implementation, subsequent accesses to the EQUIL grid lead to
     * crashes.
     */
    void releaseEquilGrid()
    {
        delete equilCartesianIndexMapper_;
        equilCartesianIndexMapper_ = 0;

        delete equilGrid_;
        equilGrid_ = 0;
    }

    /*!
     * \brief Distribute the simulation grid over multiple processes
     *
     * (For parallel simulation runs.)
     */
    void loadBalance()
    {
        auto gridView = grid().leafGridView();
        auto dataHandle = cartesianIndexMapper_->dataHandle(gridView);
        grid().loadBalance(*dataHandle);

        // communicate non-interior cells values
        grid().communicate(*dataHandle,
                           Dune::InteriorBorder_All_Interface,
                           Dune::ForwardCommunication );

        if (grid().size(0))
        {
            globalTrans_.reset(new EclTransmissibility<TypeTag>(*this));
            globalTrans_->update(false);
        }

        auto& parallelEclState = dynamic_cast<ParallelEclipseState&>(this->eclState());
        // reset cartesian index mapper for auto creation of field properties
        parallelEclState.resetCartesianMapper(cartesianIndexMapper_.get());
        parallelEclState.switchToDistributedProps();
    }

    template<class DataHandle>
    void scatterData(DataHandle& handle) const
    {

    }

    template<class DataHandle>
    void gatherData(DataHandle& handle) const
    {

    }

    template<class DataHandle, class InterfaceType, class CommunicationDirection>
    void communicate (DataHandle& data, InterfaceType iftype, CommunicationDirection dir) const
    {

    }

    /*!
     * \brief Free the memory occupied by the global transmissibility object.
     *
     * After writing the initial solution, this array should not be necessary anymore.
     */
    void releaseGlobalTransmissibilities()
    {
        globalTrans_.reset();
    }

    /*!
     * \brief Returns the object which maps a global element index of the simulation grid
     *        to the corresponding element index of the logically Cartesian index.
     */
    const CartesianIndexMapper& cartesianIndexMapper() const
    { return *cartesianIndexMapper_; }

    /*!
     * \brief Returns mapper from compressed to cartesian indices for the EQUIL grid
     */
    const EquilCartesianIndexMapper& equilCartesianIndexMapper() const
    { return *equilCartesianIndexMapper_; }

    /*!
     * \brief Get function to query cell centroids for a distributed grid.
     *
     * Currently this only non-empty for a loadbalanced CpGrid.
     * It is a function return the centroid for the given element
     * index.
     */
    std::function<std::array<double,dimensionworld>(int)>
    cellCentroids() const
    {
        return this->cellCentroids_(cartesianIndexMapper_.get());
    }

    const EclTransmissibility<TypeTag>& globalTransmissibility() const
    {
        assert( globalTrans_ != nullptr );
        return *globalTrans_;
    }

    void releaseGlobalTransmissibility()
    {
        globalTrans_.reset();
    }

    const std::vector<int>& globalCell()
    {
        return cartesianCellId_;
    }

    unsigned int gridEquilIdxToGridIdx(unsigned int elemIndex) const {
        return equilGridToGrid_[elemIndex];
    }

    unsigned int gridIdxToEquilGridIdx(unsigned int elemIndex) const {
        return ordering_[elemIndex];
    }


protected:
    void createGrids_()
    {
        // we use separate grid objects: one for the calculation of the initial condition
        // via EQUIL and one for the actual simulation. The reason is that the EQUIL code
        // cannot cope with arbitrary Dune grids and is also allergic to distributed
        // grids.

        /////
        // create the EQUIL grid
        /////
        equilGrid_ = new EquilGrid();
        equilGrid_->processEclipseFormat(mpiRank == 0 ? &this->eclState().getInputGrid()
                                                 : nullptr,
                                    /*isPeriodic=*/false,
                                    /*flipNormals=*/false,
                                    /*clipZ=*/false,
                                    mpiRank == 0 ? this->eclState().fieldProps().porv(true)
                                                 : std::vector<double>(),
                                    this->eclState().getInputNNC());

        cartesianCellId_ = equilGrid_->globalCell();

        for (unsigned i = 0; i < dimension; ++i)
            cartesianDimension_[i] = equilGrid_->logicalCartesianSize()[i];

        equilCartesianIndexMapper_ = new EquilCartesianIndexMapper(*equilGrid_);

        /////
        // create the simulation grid
        /////
        Dune::FromToGridFactory<Grid> factory;
        grid_ = factory.convert(*equilGrid_, cartesianCellId_, ordering_);

        equilGridToGrid_.resize(ordering_.size());
        for (size_t index = 0; index<ordering_.size(); ++index) {
            equilGridToGrid_[ordering_[index]] = index;
        }

        cartesianIndexMapper_.reset(std::make_unique<CartesianIndexMapper>(*grid_, cartesianDimension_, cartesianCellId_));
    }

    void filterConnections_()
    {
        // not handling the removal of completions for this type of grid yet.
    }

    Grid* grid_;
    EquilGrid* equilGrid_;
    std::vector<int> cartesianCellId_;
    std::vector<unsigned int> ordering_;
    std::vector<unsigned int> equilGridToGrid_;
    std::array<int,dimension> cartesianDimension_;
    std::unique_ptr<CartesianIndexMapper> cartesianIndexMapper_;
    EquilCartesianIndexMapper* equilCartesianIndexMapper_;
    std::unique_ptr<EclTransmissibility<TypeTag> > globalTrans_;
    int mpiRank;

};

} // namespace Opm

#endif
