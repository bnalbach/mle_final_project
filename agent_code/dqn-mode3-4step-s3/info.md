=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.93
    action_loop_rate   0.777
    pos_loop_rate      0.777
    invalid_rate       0.000
    wait_rate          0.001
    tile_diversity     0.18
    coins              28.0

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.93
    action_loop_rate   0.715
    pos_loop_rate      0.715
    invalid_rate       0.000
    wait_rate          0.168
    tile_diversity     0.07
    coins              1.8

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              28.0/50
    collect_rate       0.56
    coins/step         0.073
    steps_alive        381
    stuck_rate         0.93
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.17
    coins              8.4
    win_rate           0.04
    stuck_rate         0.98
    survival_rate      0.98

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.20
    mean_rank          2.37
    score              3.32
    kills/round        0.23
    suicides/round     0.21
    survival_rate      0.44
    steps_survived     248
    stuck_rate         0.33
    coin_share         0.25

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.31
    mean_rank          1.92
    score              3.76
    kills/round        0.22
    suicides/round     0.13
    survival_rate      0.65
    steps_survived     316
    stuck_rate         0.57
    coin_share         0.32

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.14
    mean_rank          2.49
    score              2.64
    kills/round        0.16
    suicides/round     0.16
    survival_rate      0.52
    steps_survived     261
    stuck_rate         0.48
    coin_share         0.21