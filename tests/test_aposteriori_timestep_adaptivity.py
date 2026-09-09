#!/usr/bin/env python3
# This file is part of the Open Porous Media project (OPM).
#
# OPM is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 2 of the License, or (at your option) any later
# version.
#
# OPM is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along
# with OPM. If not, see <http://www.gnu.org/licenses/>.
"""
Runs a deck three times with `flow_blackoil`: (a) the a posteriori estimators
off (standard AdaptiveTimeStepping); (b) --enable-aposteriori-estimators=true
--enable-aposteriori-timestep-control=true, the production default, where the
override only ever LIMITS AdaptiveTimeStepping's own suggestion
(min(native, estimator) -- see EnableAposterioriTimestepGrowthOverride's doc
comment) so it can shrink a step but never force growth on its own; (c) the
same plus --enable-aposteriori-timestep-growth-override=true, the
experimental bidirectional mode where the estimator's rescale is applied
directly, in both directions. Checks that:

  (1) all three runs complete successfully;
  (2) both the (b) and (c) runs actually evaluated and applied a rescale
      (the "DRIVING next dt" log line, with a finite positive suggested dt);
  (3) all three runs reach the same simulated end time (no run gave up early);
  (4) the (c) bidirectional run's timestep / Newton-iteration /
      linear-iteration totals are NOT all identical to the standard run's --
      i.e. the override demonstrably changed the timestep sequence, not just
      been silently absorbed by AdaptiveTimeStepping's own growth-factor cap
      (which is what happens on "easy", smoothly-converging decks such as
      SPE1 -- see the case comment below).

Note the (b) limiter run is deliberately NOT required to differ from
standard: on a deck where eta_time never exceeds Gamma_time*eta_sp anywhere
in the trajectory (true of both SPE9 decks tested 2026-09-04 -- spatial error
dominates throughout), a correctly-implemented limiter has nothing to shrink
and must be a complete no-op, by design. Only the unconstrained (c) run is
guaranteed to differ whenever the estimator computes a rescale at all, so it
is the one checked for "the override was computed but never actually reached
AdaptiveTimeStepping" -- this test's original regression target.

This is a diagnostic/regression check on the *mechanism*, not on solution
accuracy: (4) only requires the two trajectories to differ, in either
direction, by any amount; it is deliberately insensitive to the concrete
iteration counts (which are legitimately platform/compiler dependent) so it
stays robust while still catching the "override silently does nothing"
regression this stage's own manual testing found.

Usage:
    test_aposteriori_timestep_adaptivity.py --exe <flow_blackoil> \\
        --deck <path/to/DECK.DATA> --workdir <scratch dir>
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def run(exe, deck, outdir, extra_args):
    outdir.mkdir(parents=True, exist_ok=True)
    logpath = outdir / "run.log"
    cmd = [str(exe), str(deck), f"--output-dir={outdir}"] + extra_args
    with open(logpath, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    return proc.returncode, logpath.read_text(errors="ignore")


def grab_int(pattern, text, name):
    m = re.search(pattern, text)
    if not m:
        raise AssertionError(f"could not find '{name}' in simulator output")
    return int(m.group(1))


def grab_schedule_end_day(text):
    # "Starting time step N, stepsize D days, at day X/Y, date = ..." -- Y is
    # the total simulated horizon in days, constant across all such lines.
    matches = re.findall(r"at day [\d.]+/([\d.]+),", text)
    if not matches:
        raise AssertionError("could not find any 'Starting time step' lines")
    return float(matches[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True, type=Path)
    ap.add_argument("--deck", required=True, type=Path)
    ap.add_argument("--workdir", required=True, type=Path)
    args = ap.parse_args()

    if not args.exe.exists():
        print(f"SKIP: simulator {args.exe} not built", file=sys.stderr)
        return 0
    if not args.deck.exists():
        print(f"SKIP: deck {args.deck} not found (opm-tests not checked out?)",
              file=sys.stderr)
        return 0

    rc_std, log_std = run(args.exe, args.deck, args.workdir / "standard", [])
    # Production default: the override only ever LIMITS the native suggestion
    # (min(native, estimator)) -- see EnableAposterioriTimestepGrowthOverride's
    # doc comment. On a deck where eta_time never exceeds Gamma_time*eta_sp
    # (spatial error dominates throughout -- true of both SPE9 decks tested
    # 2026-09-04), this is CORRECTLY a complete no-op: the estimator has
    # nothing to shrink, and it must not be able to force growth on its own.
    # So this run is checked only for "fires safely", not for differing from
    # standard.
    rc_est, log_est = run(
        args.exe, args.deck, args.workdir / "steered",
        ["--enable-aposteriori-estimators=true",
         "--enable-aposteriori-timestep-control=true"])
    # Experimental bidirectional mode: the override is applied directly, in
    # both directions (bounded by --aposteriori-max-grow/-shrink), bypassing
    # the min(native, ...) limiter entirely. This is the mode guaranteed to
    # reach AdaptiveTimeStepping and change the trajectory whenever the
    # estimator computes a rescale at all -- it is the right one to check for
    # "the override was computed but never actually reached
    # AdaptiveTimeStepping" (this test's original regression target).
    rc_bidir, log_bidir = run(
        args.exe, args.deck, args.workdir / "bidir",
        ["--enable-aposteriori-estimators=true",
         "--enable-aposteriori-timestep-control=true",
         "--enable-aposteriori-timestep-growth-override=true"])

    assert rc_std == 0, f"standard run failed (exit {rc_std}); see {args.workdir}/standard/run.log"
    assert rc_est == 0, f"estimator-steered (limiter) run failed (exit {rc_est}); see {args.workdir}/steered/run.log"
    assert rc_bidir == 0, f"estimator-steered (bidirectional) run failed (exit {rc_bidir}); see {args.workdir}/bidir/run.log"

    # (2) the mechanism actually fired with a sane suggested dt, in both modes.
    def check_rescales(log, label):
        rescales = re.findall(
            r"rescale -> ([\d.eE+-]+) d, .*DRIVING next dt", log)
        assert rescales, f"no 'DRIVING next dt' line found in the {label} run -- the estimator never suggested an override"
        values = [float(x) for x in rescales]
        assert all(v > 0.0 for v in values), f"a suggested dt was <= 0 in the {label} run"
        print(f"[ok] {label}: {len(values)} space/time-balance rescales logged, "
              f"range [{min(values):.4g}, {max(values):.4g}] days")

    check_rescales(log_est, "limiter")
    check_rescales(log_bidir, "bidirectional")

    # (3) all three runs targeted the same simulated horizon (a full "Overall
    # Newton/Linear Iterations" summary, checked below, only ever gets printed
    # once the run has gone through the whole schedule, so this is a sanity
    # check on the deck/setup rather than on completion itself).
    end_std = grab_schedule_end_day(log_std)
    end_est = grab_schedule_end_day(log_est)
    end_bidir = grab_schedule_end_day(log_bidir)
    assert end_std == end_est == end_bidir, "the runs targeted different simulated end times"

    # (4) the bidirectional override demonstrably changed the timestep
    # sequence -- this is the "reaches AdaptiveTimeStepping" check. The
    # limiter run is deliberately NOT required to differ (see above).
    nsteps_std = grab_int(r"Number of timesteps:\s*(\d+)", log_std, "timestep count")
    nsteps_est = grab_int(r"Number of timesteps:\s*(\d+)", log_est, "timestep count")
    nsteps_bidir = grab_int(r"Number of timesteps:\s*(\d+)", log_bidir, "timestep count")
    newton_std = grab_int(r"Overall Newton Iterations:\s*(\d+)", log_std, "Newton its")
    newton_est = grab_int(r"Overall Newton Iterations:\s*(\d+)", log_est, "Newton its")
    newton_bidir = grab_int(r"Overall Newton Iterations:\s*(\d+)", log_bidir, "Newton its")
    linear_std = grab_int(r"Overall Linear Iterations:\s*(\d+)", log_std, "linear its")
    linear_est = grab_int(r"Overall Linear Iterations:\s*(\d+)", log_est, "linear its")
    linear_bidir = grab_int(r"Overall Linear Iterations:\s*(\d+)", log_bidir, "linear its")

    print(f"{'':20s}{'standard':>12s}{'limiter':>12s}{'bidirectional':>14s}")
    print(f"{'timesteps':20s}{nsteps_std:>12d}{nsteps_est:>12d}{nsteps_bidir:>14d}")
    print(f"{'Newton iterations':20s}{newton_std:>12d}{newton_est:>12d}{newton_bidir:>14d}")
    print(f"{'linear iterations':20s}{linear_std:>12d}{linear_est:>12d}{linear_bidir:>14d}")

    identical_bidir = (nsteps_std == nsteps_bidir
                        and newton_std == newton_bidir
                        and linear_std == linear_bidir)
    assert not identical_bidir, (
        "the bidirectional estimator-steered run produced an IDENTICAL "
        "timestep/Newton/linear trajectory to the standard run: the "
        "space/time-balance override was computed and logged but never "
        "actually reached AdaptiveTimeStepping (this is the regression this "
        "test exists to catch -- see the module docstring)."
    )

    print("PASS: the a posteriori time-step override measurably changed the "
          "simulated trajectory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
