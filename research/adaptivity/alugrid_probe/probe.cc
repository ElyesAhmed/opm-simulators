#include <config.h>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/alugrid/grid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>
#include <dune/grid/utility/persistentcontainer.hh>
#include <dune/grid/common/rangegenerators.hh>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Backend capability test only: scalar cell inventories, not black-oil state.
using Grid = Dune::ALUGrid<3, 3, Dune::cube, Dune::nonconforming>;
struct State { double density = 0; double mass = 0; double volume = 0; };

int main(int argc, char** argv)
{
    auto& mpi = Dune::MPIHelper::instance(argc, argv);
    if (mpi.size() != 1) throw std::runtime_error("This probe validates serial transfer only");
    auto grid = Dune::StructuredGridFactory<Grid>::createCubeGrid(
        Dune::FieldVector<double,3>(0.0), Dune::FieldVector<double,3>(1.0),
        std::array<unsigned int,3>{2,2,2});
    Dune::PersistentContainer<Grid, State> data(*grid, 0);
    for (const auto& e : elements(grid->leafGridView())) {
        const auto x = e.geometry().center();
        data[e].density = 1 + x[0] + 2*x[1] + 3*x[2];
    }
    auto totals = [&]() {
        std::array<double,2> sum{};
        for (const auto& e : elements(grid->leafGridView())) {
            const double v = e.geometry().volume();
            sum[0] += v;
            sum[1] += v * data[e].density;
        }
        return sum;
    };
    const auto initial = totals();
    auto check = [&](int cycle, const char* phase) {
        const auto now = totals();
        if (std::abs(now[0] - initial[0]) > 1e-12 ||
            std::abs(now[1] - initial[1]) > 1e-12)
            throw std::runtime_error("Transfer changed volume or scalar inventory");
        std::cout << "cycle=" << cycle << " phase=" << phase
                  << " leaves=" << grid->size(0) << " volume=" << now[0]
                  << " mass=" << now[1] << '\n';
    };
    check(0, "initial");
    for (int cycle=0; cycle<4; ++cycle) {
        // A moving marked region: alternate the left and right half.
        for (const auto& e : elements(grid->leafGridView()))
            if ((e.geometry().center()[0] < 0.5) == (cycle % 2 == 0)) grid->mark(1,e);
        grid->preAdapt();
        grid->adapt();
        data.resize();
        for (const auto& e : elements(grid->leafGridView()))
            if (e.isNew()) data[e].density = data[e.father()].density;
        grid->postAdapt();
        if (grid->size(0) <= 8) throw std::runtime_error("Refinement did not change the leaf grid");
        check(cycle, "refined");
        // Exercise restriction on nonconstant child data. The perturbation
        // has zero global integral but changes individual parent inventories.
        for (const auto& e : elements(grid->leafGridView()))
            if (e.hasFather()) data[e].density += 0.1 * (e.geometry().center()[1] - 0.5);
        check(cycle, "child_data_changed");

        // Coarsen complete sibling families; restrict extensive quantities
        // before deleting children, then reconstruct the parent density.
        for (const auto& e : elements(grid->levelGridView(0))) {
            data[e].mass = 0;
            data[e].volume = 0;
        }
        for (const auto& e : elements(grid->leafGridView())) {
            if (!e.hasFather()) continue;
            auto& parent = data[e.father()];
            const double v = e.geometry().volume();
            parent.mass += v * data[e].density;
            parent.volume += v;
            grid->mark(-1,e);
        }
        for (const auto& e : elements(grid->levelGridView(0)))
            if (data[e].volume > 0) data[e].density = data[e].mass / data[e].volume;
        grid->preAdapt();
        grid->adapt();
        data.resize();
        grid->postAdapt();
        if (grid->size(0) != 8) throw std::runtime_error("Complete-family coarsening failed");
        for (const auto& e : elements(grid->leafGridView()))
            if (data[e].volume > 0 &&
                std::abs(e.geometry().volume() * data[e].density - data[e].mass) > 1e-12)
                throw std::runtime_error("Restriction changed a parent-family inventory");
        check(cycle, "coarsened");
    }
}
