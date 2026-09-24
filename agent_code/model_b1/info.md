uses model_b0 as base

training:
python main.py play --no-gui --agents user_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds 3000

eval:
=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.30
    mean_rank          1.94
    score              3.70
    kills/round        0.14
    suicides/round     0.36
    survival_rate      0.58
    steps_survived     298
    stuck_rate         0.52
    coin_share         0.33

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.44
    mean_rank          1.61
    score              4.23
    kills/round        0.11
    suicides/round     0.28
    survival_rate      0.66
    steps_survived     317
    stuck_rate         0.51
    coin_share         0.41

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.38
    mean_rank          1.98
    score              4.09
    kills/round        0.31
    suicides/round     0.35
    survival_rate      0.45
    steps_survived     274
    stuck_rate         0.52
    coin_share         0.28