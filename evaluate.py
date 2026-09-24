"""
Automatic evaluation accross different scenarios.

main scenarios:
- move > does it move sensibly, not loop, not kill itself (no/weak opponents)
- coin > does it collect coins efficiently (solo, and vs coin collectors)
- combat > does it win, and how, against different opponents

Gathers resulting metrics in a table. Metrics include: coins, kills, suicides, crates, bombs, moves and invalid, 
plus things we measure ourselves (positions, action stream, loop detection).

Extras:
--gui           include gui 
--save-replay   write replays/<...>.pt per round -> replay with main.py
--config NAME   run a single named config (handy with --gui)
Per-config game logs (step-by-step) are written to logs/eval/<config>/game.log.

Usage:
e.g.
    python evaluate.py --agent user_agent --rounds 100 --suite combat --timeout 0.5 --out results/test_user_agent.json

Results print as a table and are written to results/eval_<timestamp>.json.
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from time import sleep, time as now

### make the framework importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
if "--gui" not in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import settings as s                                   
from environment import BombeRLeWorld, GUI, WorldArgs  
from fallbacks import pygame                           

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)


class Timekeeper:
    """Paces GUI frames to a wall-clock interval (copied from main.py)."""
    def __init__(self, interval):
        self.interval = interval
        self.next_time = None

    def is_due(self):
        return self.next_time is None or now() >= self.next_time

    def note(self):
        self.next_time = now() + self.interval

    def wait(self):
        if not self.is_due():
            sleep(self.next_time - now())


# ------------------------------------------------------------------------------
# Suite definitions
# ------------------------------------------------------------------------------
def build_suites(my_agent, old_agent):
    """`focus` picks the headline metrics. Agent under test is ALWAYS agents[0]."""
    return [
        dict(name="move_solo_coinheaven", focus="move",
             scenario="coin-heaven", agents=[my_agent]),
        dict(name="move_vs_peaceful", focus="move",
             scenario="classic", agents=[my_agent, "peaceful_agent", "peaceful_agent"]),

        dict(name="coin_solo", focus="coin_solo",
             scenario="coin-heaven", agents=[my_agent]),
        dict(name="coin_vs_collectors", focus="coin_comp",
             scenario="coin-heaven",
             agents=[my_agent, "coin_collector_agent", "coin_collector_agent", "coin_collector_agent"]),

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
# Loop / stuck detection
# ------------------------------------------------------------------------------
def detect_loops(seq, min_period=5, min_repeats=3):
    """Greedily find contiguous cycles: a motif of length >= min_period repeated
    >= min_repeats times back-to-back (e.g. a 5-move motif done 3x = 15 steps).

    Returns (n_loops, steps_in_loops). Works on actions or positions.
    """
    n = len(seq)
    in_loop = [False] * n
    count = 0
    i = 0
    while i < n:
        found = False
        max_p = (n - i) // min_repeats
        for p in range(min_period, max_p + 1):
            block = seq[i:i + p]
            reps = 1
            while seq[i + reps * p: i + (reps + 1) * p] == block:
                reps += 1
            if reps >= min_repeats:
                length = reps * p
                for k in range(i, i + length):
                    in_loop[k] = True
                count += 1
                i += length
                found = True
                break
        if not found:
            i += 1
    return count, sum(in_loop)


# ------------------------------------------------------------------------------
# World construction (mirrors main.py; no training)
# ------------------------------------------------------------------------------
def make_world_args(scenario, seed, log_dir, no_gui, save_replay, match_name):
    return WorldArgs(
        no_gui=no_gui,
        fps=15,
        turn_based=False,
        update_interval=0.1,
        save_replay=save_replay,          # True -> replays/<round_id>.pt per round
        replay=None,
        make_video=False,
        continue_without_training=True,   # eval agents never train
        log_dir=log_dir,
        save_stats=False,
        match_name=match_name,
        seed=seed,
        silence_errors=False,
        scenario=scenario,
    )


def _round_metrics(world, me, loop_period, loop_repeats):
    stat = me.statistics
    actions = world.replay["actions"].get(me.name, [])
    n_act = max(1, len(actions))

    opponents = [a for a in world.agents if a is not me]
    total_coins = stat.get("coins", 0) + sum(a.statistics.get("coins", 0) for a in opponents)
    scores = [me.score] + [a.score for a in opponents]
    top = max(scores)
    n_top = sum(1 for sc in scores if sc == top)
    outright_win = (me.score == top and n_top == 1)
    rank = 1 + sum(1 for sc in scores if sc > me.score)

    act_loops, act_loop_steps = detect_loops(actions, loop_period, loop_repeats)
    pos_loops, pos_loop_steps = detect_loops(me._positions, loop_period, loop_repeats)

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
        # --- loop / stuck metrics ---
        stuck=int(act_loops > 0 or pos_loops > 0),
        action_loops=act_loops,
        action_loop_rate=act_loop_steps / n_act,
        pos_loops=pos_loops,
        pos_loop_rate=pos_loop_steps / max(1, len(me._positions)),
        # --- competitive ---
        coin_share=(stat.get("coins", 0) / total_coins) if total_coins else 0.0,
        total_coins=total_coins,
        outright_win=int(outright_win),
        rank=rank,
        n_opponents=len(opponents),
    )


def run_config(cfg, args, log_dir):
    s.TIMEOUT = args.timeout  # measure decisions

    # Fresh, non-duplicated game log per config.
    logging.getLogger("BombeRLeWorld").handlers.clear()
    os.makedirs(log_dir, exist_ok=True)

    use_gui = args.gui
    match_name = cfg["name"] if args.save_replay else None
    world_args = make_world_args(cfg["scenario"], args.seed, log_dir,
                                 no_gui=not use_gui, save_replay=args.save_replay,
                                 match_name=match_name)

    agents_spec = [(d, False) for d in cfg["agents"]]
    world = BombeRLeWorld(world_args, agents_spec)
    me = world.agents[0]
    assert me.code_name == cfg["agents"][0], "agent under test must be first"

    gui = None
    timekeeper = None
    if use_gui:
        try:
            gui = GUI(world)
            timekeeper = Timekeeper(args.update_interval)
        except Exception as ex:  # real pygame not available
            print(f"[warn] could not start GUI ({ex}); running headless.")
            gui = None

    per_round = []
    quit_requested = False
    for _ in range(args.rounds):
        if quit_requested:
            break
        world.new_round()
        me._positions = [(me.x, me.y)]
        me._steps_alive = 0
        while world.running:
            if gui is not None:
                timekeeper.wait()
                if timekeeper.is_due():
                    timekeeper.note()
                    gui.render()
                    pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                        break
                    if event.type == pygame.KEYDOWN and event.key in ESCAPE_KEYS:
                        world.end_round()
                if quit_requested:
                    break
            if not world.running:
                break
            world.do_step()
            if not me.dead:
                me._positions.append((me.x, me.y))
                me._steps_alive += 1
        if world.running:          # if loop broke early (quit), close the round cleanly
            world.end_round()
        per_round.append(_round_metrics(world, me, args.loop_period, args.loop_repeats))

    return aggregate(per_round), per_round


def aggregate(per_round):
    agg = {}
    for k in per_round[0].keys():
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
    f = cfg["focus"]
    if f == "move":
        return [
            ("survival_rate", f"{agg['survived']:.2f}"),
            ("suicides/round", f"{agg['suicides']:.2f}"),
            ("stuck_rate", f"{agg['stuck']:.2f}"),
            ("action_loop_rate", f"{agg['action_loop_rate']:.3f}"),
            ("pos_loop_rate", f"{agg['pos_loop_rate']:.3f}"),
            ("invalid_rate", f"{agg['invalid_rate']:.3f}"),
            ("wait_rate", f"{agg['wait_rate']:.3f}"),
            ("tile_diversity", f"{agg['diversity']:.2f}"),
            ("coins", f"{agg['coins']:.1f}"),
        ]
    if f == "coin_solo":
        coin_count = s.SCENARIOS[cfg["scenario"]]["COIN_COUNT"]
        return [
            ("coins", f"{agg['coins']:.1f}/{coin_count}"),
            ("collect_rate", f"{agg['coins'] / coin_count:.2f}"),
            ("coins/step", f"{agg['coins'] / max(1, agg['steps_alive']):.3f}"),
            ("steps_alive", f"{agg['steps_alive']:.0f}"),
            ("stuck_rate", f"{agg['stuck']:.2f}"),
            ("suicides/round", f"{agg['suicides']:.2f}"),
            ("invalid_rate", f"{agg['invalid_rate']:.3f}"),
        ]
    if f == "coin_comp":
        return [
            ("coin_share", f"{agg['coin_share']:.2f}"),
            ("coins", f"{agg['coins']:.1f}"),
            ("win_rate", f"{agg['outright_win']:.2f}"),
            ("stuck_rate", f"{agg['stuck']:.2f}"),
            ("survival_rate", f"{agg['survived']:.2f}"),
        ]
    return [  # combat
        ("win_rate", f"{agg['outright_win']:.2f}"),
        ("mean_rank", f"{agg['rank']:.2f}"),
        ("score", f"{agg['score']:.2f}"),
        ("kills/round", f"{agg['kills']:.2f}"),
        ("suicides/round", f"{agg['suicides']:.2f}"),
        ("survival_rate", f"{agg['survived']:.2f}"),
        ("steps_survived", f"{agg['steps_alive']:.0f}"),
        ("stuck_rate", f"{agg['stuck']:.2f}"),
        ("coin_share", f"{agg['coin_share']:.2f}"),
    ]


def print_report(results):
    for name, (cfg, agg, _) in results.items():
        opp = ", ".join(cfg["agents"][1:]) or "none"
        print(f"\n=== {name}  [{cfg['scenario']}]  vs {opp}  (n={agg['n_rounds']}) ===")
        for label, val in headline(cfg, agg):
            print(f"    {label:<18} {val}")



def main():
    ap = argparse.ArgumentParser(description="Evaluate the trained Bomberman agent.")
    ap.add_argument("--agent", default="user_agent", help="agent_code/<dir> under test")
    ap.add_argument("--old-agent", default="user_agent", help="'old model' opponent dir")
    ap.add_argument("--rounds", type=int, default=30, help="rounds per config")
    ap.add_argument("--seed", type=int, default=1, help="base RNG seed (board layout)")
    ap.add_argument("--suite", default="all", choices=["all", "move", "coin", "combat"])
    ap.add_argument("--config", default=None, help="run only this named config")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="per-step think-time budget (use 0.5 for tournament realism)")
    ap.add_argument("--gui", action="store_true", help="watch live (needs real pygame + display)")
    ap.add_argument("--update-interval", type=float, default=0.1, help="GUI seconds/step")
    ap.add_argument("--save-replay", action="store_true", help="save replays/<config> | Round..pt")
    ap.add_argument("--loop-period", type=int, default=5, help="min motif length for a loop")
    ap.add_argument("--loop-repeats", type=int, default=3, help="min back-to-back repeats for a loop")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    suites = build_suites(args.agent, args.old_agent)
    if args.config:
        suites = [c for c in suites if c["name"] == args.config]
        if not suites:
            sys.exit(f"No config named {args.config!r}. "
                     f"Options: {[c['name'] for c in build_suites(args.agent, args.old_agent)]}")
    elif args.suite != "all":
        suites = [c for c in suites if c["name"] in SUITE_GROUPS[args.suite]]

    results = {}
    for cfg in suites:
        log_dir = os.path.join(PROJECT_ROOT, "logs", "eval", cfg["name"])
        t0 = time.time()
        print(f"[running] {cfg['name']} ...", flush=True)
        agg, per_round = run_config(cfg, args, log_dir)
        results[cfg["name"]] = (cfg, agg, per_round)
        print(f"[done]    {cfg['name']} in {time.time() - t0:.1f}s "
              f"(log: logs/eval/{cfg['name']}/game.log)", flush=True)

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