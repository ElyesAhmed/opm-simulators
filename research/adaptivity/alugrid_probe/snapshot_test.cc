#include <config.h>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/alugrid/grid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/common/rangegenerators.hh>
#include "../../../opm/simulators/flow/AluAdaptTransfer.hpp"
#include <array>
#include <cmath>
#include <iostream>

int main(int argc, char** argv)
{
    auto& mpi = Dune::MPIHelper::instance(argc, argv);
    if (mpi.size() != 1) throw std::runtime_error("serial test");
    using Grid = Dune::ALUGrid<3, 3, Dune::cube, Dune::nonconforming>;
    auto grid = Dune::StructuredGridFactory<Grid>::createCubeGrid(
        Dune::FieldVector<double, 3>(0), Dune::FieldVector<double, 3>(1),
        std::array<unsigned int, 3>{2, 2, 2});
    using View = Grid::LeafGridView;
    Opm::AluAdaptSnapshot<View, double> snapshot;
    // Distinct temperatures identify each original cell; a constant field
    // would hide a wrong-cell transfer.
    auto temperature = [](const auto& e) {
        auto x = e.geometry().center();
        return 280.0 + 10*x[0] + 20*x[1] + 40*x[2];
    };
    for (int cycle = 0; cycle < 2; ++cycle) {
        const auto before = grid->leafGridView();
        Dune::MultipleCodimMultipleGeomTypeMapper<View> mapper(before, Dune::mcmgElementLayout());
        std::vector<double> values(before.size(0));
        for (const auto& e : elements(before)) {
            auto parent = e;
            while (parent.hasFather()) parent = parent.father();
            values.at(mapper.index(e)) = temperature(parent);
        }
        snapshot.capture(before, [&](std::size_t i) { return values.at(i); });
        for (const auto& e : elements(before))
            if (e.geometry().center()[0] < 0.5) grid->mark(1, e);
        grid->preAdapt(); grid->adapt(); grid->postAdapt();
        const auto after = grid->leafGridView();
        std::size_t checked = 0;
        for (const auto& e : elements(after)) {
            auto parent = e;
            while (parent.hasFather()) parent = parent.father();
            if (snapshot.lookup(after, e) != temperature(parent))
                throw std::runtime_error("Initial temperature mapped to wrong ancestor");
            ++checked;
        }
        std::cout << "cycle=" << cycle << " checked=" << checked << '\n';
    }
}
