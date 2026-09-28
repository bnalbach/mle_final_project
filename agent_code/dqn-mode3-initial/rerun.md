rerun evaluation.py wtih same settings on different machine

=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.13
    action_loop_rate   0.040
    pos_loop_rate      0.035
    invalid_rate       0.000
    wait_rate          0.000
    tile_diversity     0.77
    coins              50.0

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      0.98
    suicides/round     0.02
    stuck_rate         0.44
    action_loop_rate   0.040
    pos_loop_rate      0.007
    invalid_rate       0.000
    wait_rate          0.036
    tile_diversity     0.35
    coins              8.9

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              50.0/50
    collect_rate       1.00
    coins/step         0.327
    steps_alive        153
    stuck_rate         0.13
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.23
    coins              11.7
    win_rate           0.14
    stuck_rate         0.98
    survival_rate      0.98

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.40
    mean_rank          1.78
    score              4.01
    kills/round        0.13
    suicides/round     0.32
    survival_rate      0.57
    steps_survived     301
    stuck_rate         0.49
    coin_share         0.37

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.37
    mean_rank          1.68
    score              4.28
    kills/round        0.16
    suicides/round     0.27
    survival_rate      0.59
    steps_survived     300
    stuck_rate         0.50
    coin_share         0.39

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.34
    mean_rank          1.90
    score              3.95
    kills/round        0.22
    suicides/round     0.36
    survival_rate      0.48
    steps_survived     275
    stuck_rate         0.38
    coin_share         0.32