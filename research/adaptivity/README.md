# Adaptivity research baseline

`alugrid_probe/` is an isolated DUNE module testing native hexahedral refinement/coarsening and scalar inventory transfer. It does not modify Flow's production timestep loop or establish black-oil transfer correctness.

The current architectural review and preserved benchmarks are in `/home/elyesa/Projects/Paper_aposteriro/adaptivity/`. The experimental `flow_blackoil_adaptive_dynamic` driver remains a historical single-rebuild baseline, not the intended repeated-adaptation controller.
