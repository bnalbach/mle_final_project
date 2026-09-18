"""
Generally just run python evaluate.py --agent user_agent --rounds 100 --suite combat --timeout 0.5
Just put it in same dir as main.py



Automatic evaluation harness for the Bomberman DQN agent.

Runs the *trained* agent (self.train = False, greedy) inside the ORIGINAL
framework across several fixed scenarios / opponent line-ups and reports
behavioural metrics tuned to what each suite is meant to test:

    move    - does it move sensibly and not kill itself (no/weak opponents)
    coin    - does it collect coins efficiently (solo, and vs coin collectors)
    combat  - does it win, and how, against mixed opponents

Nothing here touches training. It builds a BombeRLeWorld exactly like main.py
does, steps it to the end, and reads the per-agent statistics the engine
already tracks (coins / kills / suicides / crates / bombs / moves / invalid),
plus a few things we measure ourselves (positions, action stream).

Usage
-----
    python evaluate.py                         # agent_v3 vs all suites, 30 rounds
    python evaluate.py --rounds 100
    python evaluate.py --suite combat
    python evaluate.py --agent agent_v3 --old-agent user_agent
    python evaluate.py --timeout 0.5           # enforce tournament think-time

Results are printed as a table and written to results/eval_<timestamp>.json.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

# --- make the framework importable and headless -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import settings as s                              # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402


# ------------------------------------------------------------------------------
# Suite definitions
# ------------------------------------------------------------------------------
def build_suites(my_agent, old_agent):
    """Return the list of evaluation configs.

    `focus` picks which headline metrics get highlighted for that config.
    The agent under test is ALWAYS the first entry in `agents`.
    """
    return [
        # --- movement sanity: something to do, (almost) no lethal pressure ----
        dict(name="move_solo_coinheaven", focus="move",
             scenario="coin-heaven", agents=[my_agent]),
        dict(name="move_vs_peaceful", focus="move",
             scenario="classic", agents=[my_agent, "peaceful_agent", "peaceful_agent"]),

        # --- coin collection --------------------------------------------------
        dict(name="coin_solo", focus="coin_solo",
             scenario="coin-heaven", agents=[my_agent]),
        dict(name="coin_vs_collectors", focus="coin_comp",
             scenario="coin-heaven",
             agents=[my_agent, "coin_collector_agent", "coin_collector_agent", "coin_collector_agent"]),

        # --- combat / mixed opponents (tournament conditions) -----------------
        dict(name="combat_3rule", focus="combat",
             scenario="classic",
             agents=[my_agent, "rule_based_agent", "rule_based_agent", "rule_based_agent"]),
        dict(name="combat_mixed", focus="combat",
             scenario="classic",
             agents=[my_agent, "rule_based_agent", "coin_collector_agent", "random_agent"]),
        dict(name="combat_vs_oldmodel", focus="combat",
             scenario="classic",
             agents=[my_agent, "rule_based_agent", "rule_based_agent", old_agent]),
    ]


SUITE_GROUPS = {
    "move":   {"move_solo_coinheaven", "move_vs_peaceful"},
    "coin":   {"coin_solo", "coin_vs_collectors"},
    "combat": {"combat_3rule", "combat_mixed", "combat_vs_oldmodel"},
}


# ------------------------------------------------------------------------------
# World construction (mirrors main.py, no GUI, no training)
# ------------------------------------------------------------------------------
def make_world_args(scenario, seed, log_dir):
    return WorldArgs(
        no_gui=True,
        fps=15,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,   # eval agents never train
        log_dir=log_dir,
        save_stats=False,
        match_name=None,
        seed=seed,
        silence_errors=False,
        scenario=scenario,
    )


def _round_metrics(world, me):
    """Extract one round's metrics for the agent-under-test `me`."""
    stat = me.statistics  # defaultdict(int): coins/kills/suicides/crates/bombs/moves/invalid
    actions = world.replay["actions"].get(me.name, [])
    n_act = max(1, len(actions))

    # opponents
    opponents = [a for a in world.agents if a is not me]
    total_coins = stat.get("coins", 0) + sum(a.statistics.get("coins", 0) for a in opponents)
    scores = [me.score] + [a.score for a in opponents]
    top = max(scores)
    n_top = sum(1 for sc in scores if sc == top)
    outright_win = (me.score == top and n_top == 1)
    shared_win = (me.score == top)
    rank = 1 + sum(1 for sc in scores if sc > me.score)

    return dict(
        score=me.score,
        coins=stat.get("coins", 0),
        kills=stat.get("kills", 0),
        suicides=stat.get("suicides", 0),
        crates=stat.get("crates", 0),
        bombs=stat.get("bombs", 0),
        moves=stat.get("moves", 0),
        invalid=stat.get("invalid", 0),
        survived=int(not me.dead),
        steps_alive=me._steps_alive,
        round_len=world.step,
        n_actions=len(actions),
        waits=actions.count("WAIT"),
        bomb_actions=actions.count("BOMB"),
        invalid_rate=stat.get("invalid", 0) / n_act,
        wait_rate=actions.count("WAIT") / n_act,
        unique_tiles=len(set(me._positions)),
        diversity=len(set(me._positions)) / max(1, len(me._positions)),
        oscillation=sum(1 for i in range(2, len(me._positions))
                        if me._positions[i] == me._positions[i - 2]) / max(1, len(me._positions)),
        coin_share=(stat.get("coins", 0) / total_coins) if total_coins else 0.0,
        total_coins=total_coins,
        outright_win=int(outright_win),
        shared_win=int(shared_win),
        rank=rank,
        n_opponents=len(opponents),
    )


def run_config(cfg, rounds, base_seed, timeout, log_dir):
    s.TIMEOUT = timeout  # measure decision quality, not CPU speed (set 0.5 for tournament realism)

    agents_spec = [(d, False) for d in cfg["agents"]]  # train=False for all
    world = BombeRLeWorld(make_world_args(cfg["scenario"], base_seed, log_dir), agents_spec)
    me = world.agents[0]
    assert me.code_name == cfg["agents"][0], "agent under test must be first"

    per_round = []
    for _ in range(rounds):
        world.new_round()
        me._positions = [(me.x, me.y)]
        me._steps_alive = 0
        while world.running:
            world.do_step()
            if not me.dead:
                me._positions.append((me.x, me.y))
                me._steps_alive += 1
        per_round.append(_round_metrics(world, me))

    return aggregate(per_round), per_round


def aggregate(per_round):
    keys = per_round[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in per_round]
        agg[k] = statistics.fmean(vals)
        if len(vals) > 1:
            agg[k + "_std"] = statistics.pstdev(vals)
    agg["n_rounds"] = len(per_round)
    return agg


# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------
def headline(cfg, agg):
    """Return the metrics that matter for this suite's purpose, as label->str."""
    f = cfg["focus"]
    if f == "move":
        return [
            ("survival_rate", f"{agg['survived']:.2f}"),
            ("suicides/round", f"{agg['suicides']:.2f}"),
            ("invalid_rate", f"{agg['invalid_rate']:.3f}"),
            ("wait_rate", f"{agg['wait_rate']:.3f}"),
            ("tile_diversity", f"{agg['diversity']:.2f}"),
            ("oscillation", f"{agg['oscillation']:.3f}"),
            ("coins", f"{agg['coins']:.1f}"),
        ]
    if f == "coin_solo":
        coin_count = s.SCENARIOS[cfg["scenario"]]["COIN_COUNT"]
        return [
            ("coins", f"{agg['coins']:.1f}/{coin_count}"),
            ("collect_rate", f"{agg['coins'] / coin_count:.2f}"),
            ("coins/step", f"{agg['coins'] / max(1, agg['steps_alive']):.3f}"),
            ("steps_alive", f"{agg['steps_alive']:.0f}"),
            ("suicides/round", f"{agg['suicides']:.2f}"),
            ("invalid_rate", f"{agg['invalid_rate']:.3f}"),
        ]
    if f == "coin_comp":
        return [
            ("coin_share", f"{agg['coin_share']:.2f}"),
            ("coins", f"{agg['coins']:.1f}"),
            ("win_rate", f"{agg['outright_win']:.2f}"),
            ("suicides/round", f"{agg['suicides']:.2f}"),
            ("survival_rate", f"{agg['survived']:.2f}"),
        ]
    # combat
    return [
        ("win_rate", f"{agg['outright_win']:.2f}"),
        ("mean_rank", f"{agg['rank']:.2f}"),
        ("score", f"{agg['score']:.2f}"),
        ("kills/round", f"{agg['kills']:.2f}"),
        ("suicides/round", f"{agg['suicides']:.2f}"),
        ("survival_rate", f"{agg['survived']:.2f}"),
        ("steps_survived", f"{agg['steps_alive']:.0f}"),
        ("coin_share", f"{agg['coin_share']:.2f}"),
    ]


def print_report(results):
    for name, (cfg, agg, _) in results.items():
        opp = ", ".join(cfg["agents"][1:]) or "none"
        print(f"\n=== {name}  [{cfg['scenario']}]  vs {opp}  (n={agg['n_rounds']}) ===")
        for label, val in headline(cfg, agg):
            print(f"    {label:<16} {val}")


# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate the trained Bomberman agent.")
    ap.add_argument("--agent", default="agent_v3", help="agent_code/<dir> under test")
    ap.add_argument("--old-agent", default="user_agent", help="'old model' opponent dir")
    ap.add_argument("--rounds", type=int, default=30, help="rounds per config")
    ap.add_argument("--seed", type=int, default=1, help="base RNG seed (board layout)")
    ap.add_argument("--suite", default="all", choices=["all", "move", "coin", "combat"])
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="per-step think-time budget (use 0.5 for tournament realism)")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    suites = build_suites(args.agent, args.old_agent)
    if args.suite != "all":
        wanted = SUITE_GROUPS[args.suite]
        suites = [c for c in suites if c["name"] in wanted]

    results = {}
    for cfg in suites:
        t0 = time.time()
        print(f"[running] {cfg['name']} ...", flush=True)
        agg, per_round = run_config(cfg, args.rounds, args.seed, args.timeout, log_dir)
        results[cfg["name"]] = (cfg, agg, per_round)
        print(f"[done]    {cfg['name']} in {time.time() - t0:.1f}s", flush=True)

    print_report(results)

    out = args.out or os.path.join(
        PROJECT_ROOT, "results", f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({name: {"config": cfg, "aggregate": agg, "rounds": per_round}
                   for name, (cfg, agg, per_round) in results.items()},
                  fh, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()