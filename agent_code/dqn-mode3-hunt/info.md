Crate reward 2.0 → 0.75. Crates were 100–160 pts/round vs a 5-pt kill — combat was noise. Now farming pays less.
Kill reward 5 → 15, opponent-eliminated 0.5 → 1.0.
New REWARD_BOMB_HITS_OPPONENT = 6.0 — paid at placement when the bomb is safe and its blast covers an opponent (dense offensive signal, not the rare terminal kill).
New REWARD_MOVED_TOWARD_OPPONENT = 0.15 — small, only when armed and not already in danger, so it learns to close in.
PENALTY_GOT_KILLED −60 → −30 — dying while engaging an opponent shouldn't be as punishing as blowing yourself up (that stays −60), or it'll be too timid to fight.
INIT_EPSILON env override — lets train_b3.sh set exploration per stage (0.50 → 0.35 → 0.25 → 0.15) so it re-explores each harder opponent set. Slower decay (0.9997) for the long run.

base: model_b1
train: bash train_b3.sh
python evaluate.py --agent dqn-mode3-hunt --suite combat --rounds 100 --seed 1 --timeout 0.5 --out results/b3_pre.json    
<!-- python compare.py results/b3_pre.json results/b3_post.json -->

possible improvements: run against other of my models (add it to train_b3.sh)

eval:
=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.60
    action_loop_rate   0.283
    pos_loop_rate      0.262
    invalid_rate       0.001
    wait_rate          0.020
    tile_diversity     0.47
    coins              50.0

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.68
    action_loop_rate   0.264
    pos_loop_rate      0.243
    invalid_rate       0.000
    wait_rate          0.042
    tile_diversity     0.23
    coins              6.3

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              50.0/50
    collect_rate       1.00
    coins/step         0.166
    steps_alive        302
    stuck_rate         0.60
    suicides/round     0.00
    invalid_rate       0.001

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.21
    coins              10.6
    win_rate           0.06
    stuck_rate         0.97
    survival_rate      0.97

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.29
    mean_rank          1.97
    score              3.76
    kills/round        0.18
    suicides/round     0.35
    survival_rate      0.50
    steps_survived     275
    stuck_rate         0.33
    coin_share         0.32

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.33
    mean_rank          1.72
    score              3.92
    kills/round        0.13
    suicides/round     0.36
    survival_rate      0.56
    steps_survived     295
    stuck_rate         0.45
    coin_share         0.37

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.25
    mean_rank          2.16
    score              3.67
    kills/round        0.24
    suicides/round     0.38
    survival_rate      0.42
    steps_survived     249
    stuck_rate         0.36
    coin_share         0.27

results: 
- bad: win_rate down, score down, suicides up
- good: kills up (was the main goal)