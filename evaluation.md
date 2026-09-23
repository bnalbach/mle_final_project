# dqn-mode1-initial
=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         1.00
    action_loop_rate   0.913
    pos_loop_rate      0.913
    invalid_rate       0.000
    wait_rate          0.003
    tile_diversity     0.07
    coins              15.4

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      0.23
    suicides/round     0.77
    stuck_rate         0.23
    action_loop_rate   0.182
    pos_loop_rate      0.182
    invalid_rate       0.019
    wait_rate          0.333
    tile_diversity     0.24
    coins              1.0

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              15.4/50
    collect_rate       0.31
    coins/step         0.039
    steps_alive        400
    stuck_rate         1.00
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.17
    coins              8.6
    win_rate           0.01
    stuck_rate         0.99
    survival_rate      0.98

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.00
    mean_rank          3.56
    score              0.65
    kills/round        0.01
    suicides/round     0.83
    survival_rate      0.01
    steps_survived     49
    stuck_rate         0.13
    coin_share         0.07

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.02
    mean_rank          2.68
    score              0.97
    kills/round        0.02
    suicides/round     0.82
    survival_rate      0.04
    steps_survived     69
    stuck_rate         0.20
    coin_share         0.10

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.01
    mean_rank          2.80
    score              0.74
    kills/round        0.01
    suicides/round     0.85
    survival_rate      0.01
    steps_survived     55
    stuck_rate         0.12
    coin_share         0.08

Wrote /home/jonathan/Documents/SS26/mle/final-project-test-folder/bomberman_rl/results/eval_20260923_150519.json


# dqn-mode2-initial
=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      0.97
    suicides/round     0.03
    stuck_rate         0.97
    action_loop_rate   0.758
    pos_loop_rate      0.754
    invalid_rate       0.006
    wait_rate          0.206
    tile_diversity     0.16
    coins              25.1

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      0.07
    suicides/round     0.93
    stuck_rate         0.06
    action_loop_rate   0.024
    pos_loop_rate      0.023
    invalid_rate       0.010
    wait_rate          0.291
    tile_diversity     0.31
    coins              1.4

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              25.1/50
    collect_rate       0.50
    coins/step         0.064
    steps_alive        390
    stuck_rate         0.97
    suicides/round     0.03
    invalid_rate       0.006

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.17
    coins              8.8
    win_rate           0.05
    stuck_rate         0.83
    survival_rate      0.83

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.04
    mean_rank          3.24
    score              1.00
    kills/round        0.04
    suicides/round     0.83
    survival_rate      0.01
    steps_survived     63
    stuck_rate         0.05
    coin_share         0.09

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.05
    mean_rank          2.56
    score              1.49
    kills/round        0.06
    suicides/round     0.85
    survival_rate      0.04
    steps_survived     74
    stuck_rate         0.05
    coin_share         0.13

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.07
    mean_rank          2.62
    score              1.53
    kills/round        0.11
    suicides/round     0.84
    survival_rate      0.02
    steps_survived     69
    stuck_rate         0.03
    coin_share         0.11

Wrote /home/jonathan/Documents/SS26/mle/final-project-test-folder/bomberman_rl/results/eval_20260923_150745.json



# dqn_version2-backup-after-coin
=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.58
    action_loop_rate   0.376
    pos_loop_rate      0.357
    invalid_rate       0.000
    wait_rate          0.041
    tile_diversity     0.52
    coins              44.8

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      0.57
    suicides/round     0.43
    stuck_rate         0.57
    action_loop_rate   0.566
    pos_loop_rate      0.565
Wrote /home/jonathan/Documents/SS26/mle/final-project-test-folder/bomberman_rl/results/eval_20260923_150519.json

    invalid_rate       0.000
    wait_rate          0.485
    tile_diversity     0.16
    coins              0.1

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              44.8/50
    collect_rate       0.90
    coins/step         0.167
    steps_alive        267
    stuck_rate         0.58
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.22
    coins              11.2
    win_rate           0.09
    stuck_rate         0.88
    survival_rate      0.83

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.00
    mean_rank          3.76
    score              0.11
    kills/round        0.00
    suicides/round     0.78
    survival_rate      0.00
    steps_survived     66
    stuck_rate         0.54
    coin_share         0.01

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.00
    mean_rank          2.85
    score              0.26
    kills/round        0.03
    suicides/round     0.75
    survival_rate      0.04
    steps_survived     92
    stuck_rate         0.55
    coin_share         0.01

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.01
    mean_rank          2.92
    score              0.20
    kills/round        0.00
    suicides/round     0.75
    survival_rate      0.00
    steps_survived     86
    stuck_rate         0.56
    coin_share         0.02

Wrote /home/jonathan/Documents/SS26/mle/final-project-test-folder/bomberman_rl/results/eval_20260923_160531.json


# dqn-mode2-selfkill
=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.02
    action_loop_rate   0.015
    pos_loop_rate      0.015
    invalid_rate       0.000
    wait_rate          0.000
    tile_diversity     0.82
    coins              49.9

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      0.09
    suicides/round     0.91
    stuck_rate         0.08
    action_loop_rate   0.036
    pos_loop_rate      0.036
    invalid_rate       0.000
    wait_rate          0.202
    tile_diversity     0.28
    coins              2.4

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              49.9/50
    collect_rate       1.00
    coins/step         0.381
    steps_alive        131
    stuck_rate         0.02
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.24
    coins              12.2
    win_rate           0.19
    stuck_rate         1.00
    survival_rate      1.00

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.06
    mean_rank          2.84
    score              1.83
    kills/round        0.05
    suicides/round     0.62
    survival_rate      0.04
    steps_survived     79
    stuck_rate         0.03
    coin_share         0.18

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.11
    mean_rank          2.41
    score              2.07
    kills/round        0.02
    suicides/round     0.67
    survival_rate      0.07
    steps_survived     91
    stuck_rate         0.06
    coin_share         0.22

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.05
    mean_rank          2.53
    score              2.07
    kills/round        0.04
    suicides/round     0.61
    survival_rate      0.02
    steps_survived     80
    stuck_rate         0.02
    coin_share         0.21
Wrote /home/jonathan/Documents/SS26/mle/final-project-test-folder/bomberman_rl/results/eval_20260923_160800.json


# dqn-mode2-crate_balance

=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.40
    action_loop_rate   0.114
    pos_loop_rate      0.063
    invalid_rate       0.008
    wait_rate          0.000
    tile_diversity     0.59
    coins              49.7

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      0.94
    suicides/round     0.06
    stuck_rate         0.80
    action_loop_rate   0.131
    pos_loop_rate      0.060
    invalid_rate       0.001
    wait_rate          0.045
    tile_diversity     0.33
    coins              8.8

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              49.7/50
    collect_rate       0.99
    coins/step         0.219
    steps_alive        227
    stuck_rate         0.40
    suicides/round     0.00
    invalid_rate       0.008

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.20
    coins              9.9
    win_rate           0.04
    stuck_rate         0.98
    survival_rate      0.98

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.16
    mean_rank          2.36
    score              3.27
    kills/round        0.17
    suicides/round     0.30
    survival_rate      0.15
    steps_survived     174
    stuck_rate         0.33
    coin_share         0.27

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.28
    mean_rank          1.96
    score              3.80
    kills/round        0.11
    suicides/round     0.32
    survival_rate      0.29
    steps_survived     227
    stuck_rate         0.41
    coin_share         0.36

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.24
    mean_rank          2.04
    score              5.04
    kills/round        0.29
    suicides/round     0.40
    survival_rate      0.21
    steps_survived     202
    stuck_rate         0.32
    coin_share         0.40

Wrote /home/jonathan/Documents/SS26/mle/final-project-test-folder/bomberman_rl/results/eval_20260923_162718.json


# dqn-mode2-bomb

=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.40
    action_loop_rate   0.114
    pos_loop_rate      0.063
    invalid_rate       0.008
    wait_rate          0.000
    tile_diversity     0.59
    coins              49.7

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      0.96
    suicides/round     0.04
    stuck_rate         0.83
    action_loop_rate   0.116
    pos_loop_rate      0.053
    invalid_rate       0.001
    wait_rate          0.046
    tile_diversity     0.34
    coins              8.8

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              49.7/50
    collect_rate       0.99
    coins/step         0.219
    steps_alive        227
    stuck_rate         0.40
    suicides/round     0.00
    invalid_rate       0.008

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.19
    coins              9.4
    win_rate           0.05
    stuck_rate         1.00
    survival_rate      1.00

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.18
    mean_rank          2.25
    score              3.38
    kills/round        0.16
    suicides/round     0.30
    survival_rate      0.28
    steps_survived     208
    stuck_rate         0.41
    coin_share         0.29

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.32
    mean_rank          1.77
    score              4.30
    kills/round        0.16
    suicides/round     0.36
    survival_rate      0.28
    steps_survived     221
    stuck_rate         0.43
    coin_share         0.39

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.18
    mean_rank          2.10
    score              4.87
    kills/round        0.29
    suicides/round     0.35
    survival_rate      0.21
    steps_survived     215
    stuck_rate         0.43
    coin_share         0.38
Wrote /home/jonathan/Documents/SS26/mle/final-project-test-folder/bomberman_rl/results/eval_20260923_162824.json


# dqn-mode3-initial
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
    stuck_rate         0.51
    action_loop_rate   0.063
    pos_loop_rate      0.026
    invalid_rate       0.000
    wait_rate          0.034
    tile_diversity     0.34
    coins              8.8

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
    coins              11.6
    win_rate           0.16
    stuck_rate         0.98
    survival_rate      0.98

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.27
    mean_rank          2.05
    score              3.62
    kills/round        0.13
    suicides/round     0.41
    survival_rate      0.53
    steps_survived     278
    stuck_rate         0.46
    coin_share         0.33

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.50
    mean_rank          1.55
    score              5.00
    kills/round        0.22
    suicides/round     0.35
    survival_rate      0.55
    steps_survived     293
    stuck_rate         0.40
    coin_share         0.43

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.14
    mean_rank          2.09
    score              4.59
    kills/round        0.18
    suicides/round     0.39
    survival_rate      0.42
    steps_survived     265
    stuck_rate         0.32
    coin_share         0.41

Wrote /home/jonathan/Documents/SS26/mle/final-project-test-folder/bomberman_rl/results/eval_20260923_165340.json

# dqn-mode3-multi_opponents

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
    survival_rate      0.98
    suicides/round     0.02
    stuck_rate         0.50
    action_loop_rate   0.050
    pos_loop_rate      0.020
    invalid_rate       0.000
    wait_rate          0.052
    tile_diversity     0.34
    coins              8.8

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              49.8/50
    collect_rate       1.00
    coins/step         0.187
    steps_alive        266
    stuck_rate         0.52
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.22
    coins              10.8
    win_rate           0.10
    stuck_rate         0.96
    survival_rate      0.96

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.27
    mean_rank          1.99
    score              3.53
    kills/round        0.11
    suicides/round     0.38
    survival_rate      0.55
    steps_survived     291
    stuck_rate         0.51
    coin_share         0.33

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.34
    mean_rank          1.75
    score              3.85
    kills/round        0.09
    suicides/round     0.20
    survival_rate      0.69
    steps_survived     328
    stuck_rate         0.53
    coin_share         0.38

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.28
    mean_rank          1.87
    score              5.20
    kills/round        0.30
    suicides/round     0.34
    survival_rate      0.53
    steps_survived     289
    stuck_rate         0.32
    coin_share         0.41

Wrote /home/jonathan/Documents/SS26/mle/final-project-test-folder/bomberman_rl/results/eval_20260923_165649.json



