=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.52
    action_loop_rate   0.186
    pos_loop_rate      0.150
    invalid_rate       0.000
    wait_rate          0.005
    tile_diversity     0.54
    coins              49.8

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      0.97
    suicides/round     0.03
    stuck_rate         0.50
    action_loop_rate   0.088
    pos_loop_rate      0.064
    invalid_rate       0.000
    wait_rate          0.061
    tile_diversity     0.33
    coins              8.4

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              49.8/50
    collect_rate       1.00
    coins/step         0.187
    steps_alive        266
    stuck_rate         0.52
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.21
    coins              10.6
    win_rate           0.04
    stuck_rate         0.98
    survival_rate      0.98

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.30
    mean_rank          2.03
    score              3.42
    kills/round        0.11
    suicides/round     0.25
    survival_rate      0.65
    steps_survived     311
    stuck_rate         0.58
    coin_share         0.32

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.38
    mean_rank          1.71
    score              3.70
    kills/round        0.08
    suicides/round     0.26
    survival_rate      0.70
    steps_survived     325
    stuck_rate         0.56
    coin_share         0.37

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.24
    mean_rank          2.14
    score              3.41
    kills/round        0.22
    suicides/round     0.27
    survival_rate      0.57
    steps_survived     295
    stuck_rate         0.46
    coin_share         0.26