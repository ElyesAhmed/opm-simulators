# ALUGrid capability probe

Four moving-region cycles on an eight-cell cube grid: refine half the macro cells, prolong scalar density, restrict full sibling families by volume-weighted extensive sums, coarsen back to eight cells. Checks volume/inventory at every transition. Serial only; no reservoir PDE or black-oil thermodynamics.

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/.local" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target alugrid_cycle -j2
ctest --test-dir build --output-on-failure -V
```

Requires installed DUNE common/grid/geometry and dune-alugrid (tested with DUNE 2.9 / ALUGrid 2.9.1). The OPM build need not be reconfigured. Generated output belongs in build/, not in the source tree.
