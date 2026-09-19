#!/usr/bin/env python3
"""
preflight.py -- prove the agent is actually in the control loop BEFORE spending
hours of compute. Run this ahead of every long training campaign.

    python3 preflight.py --env cage
    python3 preflight.py --env approach

WHY THIS FILE EXISTS

84 cage training runs (~2.2 h) were thrown away on 19 Sept because the action
space was raw motor RPM, Box(0, 21713.7). PPO emits roughly [-3, +3], which
clips to [0, 3] RPM against a HOVER_RPM of 14476 -- 0.02% of hover thrust. The
drone fell out of the sky on every step no matter what the policy did, and every
run returned an identical ~17%.

Every other check in the project passed. The env built, reset, stepped,
returned finite observations, and trained without error. What nobody asked was
whether the POLICY COULD CHANGE THE OUTCOME.

The decisive symptom was visible in the results and nowhere else: a 0.1% spread
across 12 different hyperparameter settings. That is the signature of a policy
that is not in the loop, and check 2 below tests for it directly.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

PY = sys.executable


def check_resources(min_ram_mb=4000, min_disk_gb=5):
    """Never launch a campaign that could take the machine down."""
    import shutil as sh
    free_gb = sh.disk_usage(".").free / 1e9
    ram_mb = None
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                ram_mb = int(line.split()[1]) / 1024
                break
    gpu = "n/a"
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    print(f"  RAM available {ram_mb:.0f} MB | disk free {free_gb:.0f} GB | GPU {gpu}")
    assert ram_mb > min_ram_mb, f"only {ram_mb:.0f} MB RAM available"
    assert free_gb > min_disk_gb, f"only {free_gb:.0f} GB disk free"
    return True


def _train(script, arm, steps, out, extra):
    """One short training run. Returns its eval dict, or None if it failed."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    cmd = [PY, script, "--arm", arm, "--seed", "0", "--steps", str(steps),
           "--n-envs", "4", "--eval-episodes", "100", "--out", out] + extra
    if "--tb" in subprocess.run([PY, script, "--help"], capture_output=True,
                                text=True).stdout:
        cmd += ["--tb", ""]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    p = os.path.join(out, "ablation.json")
    if not os.path.exists(p):
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        return None
    return json.load(open(p))["results"][arm]


def check_policy_matters(script, extra, steps=150_000, metric="success_%"):
    """
    CHECK 2 -- THE ONE THAT WOULD HAVE CAUGHT THE BUG.

    Train twice with deliberately extreme, very different hyperparameters. If
    the policy has any influence at all, the two runs must land on different
    scores. Near-identical scores mean the outcome is determined by the
    environment alone and the agent is a spectator.
    """
    tmp = tempfile.mkdtemp()
    try:
        a = _train(script, "none", steps, os.path.join(tmp, "a"),
                   extra + ["--lr", "1e-5", "--ent-coef", "0.0",
                            "--net", "32", "32"])
        b = _train(script, "none", steps, os.path.join(tmp, "b"),
                   extra + ["--lr", "3e-3", "--ent-coef", "0.05",
                            "--net", "256", "256"])
        assert a and b, "a preflight training run crashed"
        va, vb = a[metric], b[metric]
        print(f"  tiny-net/slow-lr {va:6.2f}   big-net/fast-lr {vb:6.2f}"
              f"   spread {abs(va - vb):5.2f}")
        assert abs(va - vb) > 1.0, (
            f"two very different hyperparameter settings scored {va:.2f} and "
            f"{vb:.2f} -- a spread of {abs(va-vb):.2f}. The policy is NOT "
            "influencing the outcome. Check the action space scaling before "
            "spending any compute.")
        return max(va, vb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_learning_moves(script, extra, short=60_000, long=400_000,
                         metric="success_%"):
    """
    CHECK 3 -- more training must at least CHANGE the score.

    Deliberately weak, and weaker than check 2. At small step counts RL is
    noisy and success rates near zero, so requiring IMPROVEMENT here produces
    false alarms. This only catches a totally frozen pipeline. Check 2 is the
    gate that actually protects against the wasted-campaign failure mode --
    validated against the 19 Sept cage data, where 36 runs across 12 settings
    spanned 0.25 percentage points against this check's 1.0 threshold.
    """
    tmp = tempfile.mkdtemp()
    try:
        s = _train(script, "none", short, os.path.join(tmp, "s"), extra)
        l = _train(script, "none", long, os.path.join(tmp, "l"), extra)
        assert s and l, "a preflight training run crashed"
        print(f"  {short//1000}k steps {s[metric]:6.2f}   "
              f"{long//1000}k steps {l[metric]:6.2f}")
        assert abs(l[metric] - s[metric]) > 1.0, (
            f"score barely moved between {short} and {long} steps "
            f"({s[metric]:.2f} -> {l[metric]:.2f}) -- training is not learning")
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["cage", "approach"], default="cage")
    ap.add_argument("--rho", default="0.707")
    a = ap.parse_args()

    if a.env == "cage":
        import cage_env
        script, extra = "train_cage.py", ["--rho", a.rho]
        print("\n[1/4] env self-check (incl. flight authority)")
        cage_env.demo()
    else:
        import approach_env
        script, extra = "train.py", ["--rho", a.rho, "--t-max", "45"]
        print("\n[1/4] env self-check")
        approach_env.demo("sigma_pool.npz")

    print("\n[2/4] resources")
    check_resources()

    print("\n[3/4] does the POLICY change the outcome? "
          "(two extreme hyperparameter settings must differ)")
    check_policy_matters(script, extra)

    print("\n[4/4] does more training change the score?")
    check_learning_moves(script, extra)

    print("\nPREFLIGHT PASSED -- safe to launch the campaign\n")


if __name__ == "__main__":
    main()
