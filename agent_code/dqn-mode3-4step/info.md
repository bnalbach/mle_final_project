base: dqn-mode3-hunt

training: train_b4.sh
changes: 
- callbacks.py: / (same as dqn-mode3-hunt/model_b3)
- training.py: adapted from dqn-mode3-hunt/model_b3


1. learn to hunt against killable, non-bombing targets (the missing stage)
stage hunt_safe   0.30  8000 peaceful_agent coin_collector_agent coin_collector_agent
2. real danger: opponents that bomb back
stage danger_2rule 0.20 10000 rule_based_agent rule_based_agent
3. tournament density
stage full_3rule   0.12 16000 rule_based_agent rule_based_agent rule_based_agent
4. robustness: mixed field closer to the real tournament
stage mixed_3opp   0.10 12000 rule_based_agent coin_collector_agent "$OLD"

-> saves each submodel

eval: (all)
=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.95
    action_loop_rate   0.819
    pos_loop_rate      0.818
    invalid_rate       0.000
    wait_rate          0.000
    tile_diversity     0.15
    coins              24.3

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.90
    action_loop_rate   0.698
    pos_loop_rate      0.700
    invalid_rate       0.000
    wait_rate          0.165
    tile_diversity     0.06
    coins              1.7

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              24.3/50
    collect_rate       0.49
    coins/step         0.063
    steps_alive        387
    stuck_rate         0.95
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.21
    coins              10.5
    win_rate           0.06
    stuck_rate         1.00
    survival_rate      1.00

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.23
    mean_rank          2.50
    score              3.41
    kills/round        0.27
    suicides/round     0.25
    survival_rate      0.42
    steps_survived     248
    stuck_rate         0.38
    coin_share         0.23

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.35
    mean_rank          1.74
    score              4.17
    kills/round        0.22
    suicides/round     0.10
    survival_rate      0.74
    steps_survived     327
    stuck_rate         0.54
    coin_share         0.38

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.18
    mean_rank          2.53
    score              2.85
    kills/round        0.19
    suicides/round     0.26
    survival_rate      0.47
    steps_survived     273
    stuck_rate         0.34
    coin_share         0.21

