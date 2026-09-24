USE mode3_model as .pt base! callbacks.py from b3 and for training.py use new adapted one (baed on b3).

training with train_b4.sh

1. learn to hunt against killable, non-bombing targets (the missing stage)
stage hunt_safe   0.30  8000 peaceful_agent coin_collector_agent coin_collector_agent
2. real danger: opponents that bomb back
stage danger_2rule 0.20 10000 rule_based_agent rule_based_agent
3. tournament density
stage full_3rule   0.12 16000 rule_based_agent rule_based_agent rule_based_agent
4. robustness: mixed field closer to the real tournament
stage mixed_3opp   0.10 12000 rule_based_agent coin_collector_agent "$OLD"

saves each model

performances: 
hunt_safe

danger_2rule

full_3rule
=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100)===
    win_rate           0.28
    mean_rank          2.27
    score              3.67
    kills/round        0.29
    suicides/round     0.18
    survival_rate      0.41
    steps_survived     255
    stuck_rate         0.30
    coin_share         0.26

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100)===
    win_rate           0.26
    mean_rank          2.05
    score              3.46
    kills/round        0.19
    suicides/round     0.14
    survival_rate      0.59
    steps_survived     304
    stuck_rate         0.59
    coin_share         0.30

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100)===
    win_rate           0.28
    mean_rank          2.29
    score              3.72
    kills/round        0.35
    suicides/round     0.17
    survival_rate      0.57
    steps_survived     288
    stuck_rate         0.53
    coin_share         0.24

mixed_3opp
=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.01
    mean_rank          3.83
    score              0.05
    kills/round        0.01
    suicides/round     0.75
    survival_rate      0.01
    steps_survived     45
    stuck_rate         0.37
    coin_share         0.00

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.00
    mean_rank          2.98
    score              0.00
    kills/round        0.00
    suicides/round     1.00
    survival_rate      0.00
    steps_survived     5
    stuck_rate         0.00
    coin_share         0.00

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.00
    mean_rank          2.95
    score              0.06
    kills/round        0.01
    suicides/round     0.53
    survival_rate      0.01
    steps_survived     110
    stuck_rate         0.77
    coin_share         0.00
