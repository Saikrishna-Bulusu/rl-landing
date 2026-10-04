#!/usr/bin/env python3
"""
urban_headroom.py -- checkpoint 1: how much room is there above a good rule?

Every agent flies the SAME episodes (same seed -> same pads, intruders, GeoAnchor
errors, marker decodes), so each comparison is paired episode by episode.

THE LADDER, read top to bottom
  rule             what a careful practitioner writes
  best_obs         the best planner using only what the sensors show now
  best_full        the best planner told every intruder's true future (oracle)

  best_obs  - rule       better DECISIONS from the same information. A learned
                         policy can certainly target this: nothing it needs is
                         missing from its observation.
  best_full - best_obs   knowing THIS flight's future. Past flights can supply
                         part of it only if intruders follow patterns -- which is
                         the corridor-vs-random question, and needs a decider that
                         reasons about time (checkpoint 2), see NOTE below.
  100 - best_full        failures no decider avoids with these sensors and rules

NOTE on the pattern question. A planner that imagines unseen intruders from the
prior (MPC info="prior") is also run, and it is FAR worse than the rule. Not
because the pattern is useless: inside a 20 s window, hovering always looks safer
than descending while a car might come -- but waiting only postpones that same
risk to the deadline. Pricing that properly needs value over the whole remaining
mission, which is what a trained policy learns and a myopic planner cannot.

GUARD: the best oracle must not lose to the rule (paired, 2 SE). An upper bound
that loses to its own baseline is broken, not informative.
"""
import argparse
import json
import math

import torch

import urban_agents as A
import urban_env as U

N, SEED = 4096, 2026


def paired(a, b):
    """Mean difference a-b in percentage points, with a 95% CI, per-episode paired."""
    d = a.float() - b.float()
    se = d.std(unbiased=True).item() / math.sqrt(len(d))
    return 100 * d.mean().item(), 196 * se


def failures(s):
    """Why the non-clean episodes failed, first reason only, % of all episodes."""
    oc, v = s["outcome"], s["viol_s"] > 0
    left = ~s["clean"]
    out = {}
    for name, m in [("bird strike", oc == 7), ("crash, unmapped", oc == 8), ("crash", s["crash"]), ("near mid-air (NMAC)", s["nmac"]), ("120 m ceiling", v[:, 0]),
                    ("helipad duty", v[:, 4]), ("drone separation", v[:, 5]),
                    ("people 30 m", v[:, 1]), ("vehicle 30 m", v[:, 2]), ("over a person", v[:, 3]),
                    ("timeout, people at pad", (oc == 6) & (s["people_at_arrival"] > 0)),
                    ("timeout, helipad duty", (oc == 6) & s["obliged_ever"]),
                    ("timeout, other", oc == 6), ("landed off pad", oc == 5)]:
        hit = left & m
        out[name] = 100 * hit.float().mean().item()
        left = left & ~m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="logan_map.npz", help="e.g. logan_map_lidar.npz")
    ap.add_argument("--nav", default="geoanchor", choices=("geoanchor", "gps"))
    ap.add_argument("--birds", action="store_true", help="a swooping magpie near half the pads")
    ap.add_argument("--fog", action="store_true", help="every flight in fog")
    ap.add_argument("--unmapped", action="store_true", help="structures the survey does not show")
    ap.add_argument("--cameras", action="store_true", help="magpie seen only by the forward and landing cameras")
    ap.add_argument("--hunter", action="store_true", help="the magpie attacks from behind and keeps steering")
    ap.add_argument("--rig", action="store_true", help="magpie seen by the six-camera rig's detector + tracker")
    ap.add_argument("--tag", default="", help="appended to the output name, e.g. _half for RIG_CALIB=rig_calib_half.json")
    args = ap.parse_args()
    assert not (args.cameras and args.rig)
    tag = "" if args.map == "logan_map.npz" else "_" + args.map.replace("logan_map_", "").replace(".npz", "")
    tag += ("_gps" if args.nav == "gps" else "") + ("_birds" if args.birds else "") + ("_fog" if args.fog else "") + ("_unmapped" if args.unmapped else "") + ("_cameras" if args.cameras else "") \
        + ("_hunter" if args.hunter else "") + ("_rig" if args.rig else "") + args.tag
    agents = {"rule": A.Rule(), "shield_obs": A.MPC("obs"), "safeplan_obs": A.MPC("obs", "safeplan"),
              "shield_full": A.MPC("full"), "safeplan_full": A.MPC("full", "safeplan"),
              "shield_prior": A.MPC("prior")}
    res = {}
    for world in ("corridor", "random"):
        print(f"\n=== {world} world, {N} paired episodes ===")
        S = {k: U.run(U.UrbanEnv(N, world, seed=SEED, map_path=f"{U.HERE}/{args.map}", nav=args.nav,
                                     p_swoop=U.P_SWOOP if args.birds else 0.0, p_fog=U.P_FOG if args.fog else 0.0,
                                     p_new=U.P_NEW if args.unmapped else 0.0,
                                     bird_sense="rig" if args.rig else "cameras" if args.cameras else "omni",
                                     magpie="hunter" if args.hunter else "straight"), a) for k, a in agents.items()}
        f = lambda x: 100 * x.float().mean().item()
        for k, s in S.items():
            on = s["bird_on"]
            print(f"  {k:<14} clean {f(s['clean']):5.1f}%   delivered {f(s['delivered']):5.1f}%   "
                  f"median {s['t_done'][s['clean']].median().item():4.0f} s" +
                  (f"   | with a magpie: clean {f(s['clean'][on]):5.1f}%, struck {f((s['outcome'] == 7)[on]):4.1f}%"
                   if args.birds else ""))
        best_obs = max(("shield_obs", "safeplan_obs"), key=lambda k: f(S[k]["clean"]))
        best_full = max(("shield_full", "safeplan_full"), key=lambda k: f(S[k]["clean"]))
        rule = S["rule"]["clean"]

        d, ci = paired(S[best_full]["clean"], rule)
        if d < -ci:
            raise AssertionError(f"{best_full} scores {d:+.1f} pp against the rule (95% CI +/-{ci:.1f}). "
                                 "An upper bound cannot lose to its baseline -- fix the oracle first.")
        print(f"\n  guard: {best_full} beats the rule by {d:+.1f} pp (95% CI +/-{ci:.1f}) -- the bound is valid")

        ladder = {
            "better decisions, same sensors": paired(S[best_obs]["clean"], rule),
            "knowing this flight's future": paired(S[best_full]["clean"], S[best_obs]["clean"]),
            "total headroom over the rule": paired(S[best_full]["clean"], rule),
        }
        print(f"\n  headroom ladder (clean-delivery points, paired 95% CI)")
        for k, (m, c) in ladder.items():
            print(f"    {k:<32} {m:+5.1f} pp  +/- {c:.1f}")
        print(f"    {'unavoidable with these sensors':<32} {100 - f(S[best_full]['clean']):5.1f} %")
        print(f"    {'myopic prior planner vs rule':<32} {paired(S['shield_prior']['clean'], rule)[0]:+5.1f} pp   (see NOTE)")

        print(f"\n  why episodes still fail (first reason, % of all episodes)")
        fr, fo = failures(S["rule"]), failures(S[best_full])
        for k in fr:
            if fr[k] > 0.05 or fo[k] > 0.05:
                print(f"    {k:<26} rule {fr[k]:4.1f}%   {best_full} {fo[k]:4.1f}%")
        viol = lambda s: s["crash"] | s["nmac"] | (s["viol_s"] > 0).any(-1)
        print(f"\n  any CASA/ASTM violation, NMAC or crash (% of episodes)")
        for k, s in S.items():
            print(f"    {k:<14} {f(viol(s)):4.1f}%")
        res[world] = {k: {"clean": f(s["clean"]), "delivered": f(s["delivered"]), "violation": f(viol(s)),
                          "median_t": s["t_done"][s["clean"]].median().item(),
                          **({"clean_magpie": f(s["clean"][s["bird_on"]]), "struck_magpie": f((s["outcome"] == 7)[s["bird_on"]]),
                              "clean_no_magpie": f(s["clean"][~s["bird_on"]])} if args.birds else {})}
                      for k, s in S.items()}
        res[world]["fail_best_obs"] = failures(S[best_obs])
        res[world]["ladder"] = {k: {"pp": m, "ci95": c} for k, (m, c) in ladder.items()}
        res[world]["fail_rule"], res[world]["fail_oracle"] = fr, fo
        res[world]["best_obs"], res[world]["best_full"] = best_obs, best_full

    # the pattern control: does the corridor change anything for these deciders?
    print("\n=== corridor minus random, per decider (the pattern would show up here) ===")
    for k in agents:
        print(f"  {k:<14} {res['corridor'][k]['clean'] - res['random'][k]['clean']:+5.1f} pp")
    json.dump(res, open(f"headroom{tag}.json", "w"), indent=1)
    print(f"\nsaved urban/headroom{tag}.json")


if __name__ == "__main__":
    main()
