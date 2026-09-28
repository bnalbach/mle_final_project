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
    survival_rate      0.99
    suicides/round     0.01
    stuck_rate         0.87
    action_loop_rate   0.674
    pos_loop_rate      0.672
    invalid_rate       0.000
    wait_rate          0.171
    tile_diversity     0.07
    coins              1.9

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              24.3/50
    collect_rate       0.49
    coins/step         0.063
    steps_alive        387
    stuck_rate         0.95
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.20
    coins              10.1
    win_rate           0.07
    stuck_rate         1.00
    survival_rate      1.00

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.26
    mean_rank          2.25
    score              3.43
    kills/round        0.26
    suicides/round     0.28
    survival_rate      0.47
    steps_survived     257
    stuck_rate         0.35
    coin_share         0.25

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.28
    mean_rank          1.92
    score              3.72
    kills/round        0.22
    suicides/round     0.17
    survival_rate      0.58
    steps_survived     294
    stuck_rate         0.52
    coin_share         0.31

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.15
    mean_rank          2.55
    score              2.85
    kills/round        0.21
    suicides/round     0.11
    survival_rate      0.40
    steps_survived     255
    stuck_rate         0.40
    coin_share         0.20