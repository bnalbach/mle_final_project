# Possible Implementation Improvements
- for training, train the model in different scenarios whihc get progressively harder (only movement train, then coin collection, then combat - see evaluation.py for scenarios)

- if multiple bombs on the field move in the direction where the least bombs are (exact oppisite direction)
- ALSO look at movement of directions of other agents (e.g. calculate avg direction the enemy is moving in based on their last 5-7 steps) --> use it for deciding where to lay bombs
  - cceate in `danger_cells`
- *prob. way too complex: pattern detection in enemy movement to adjust wn behavior based on the opponent*
- account for agents that may block you as you try to escape from a bomb
  - also usable in reverse -> block of other agents trying to escape from a bomb
- isn't `KILLED_SELF` a bit of a suboptimal way to penalize suicide? --> many ways can lead to acidental suicide, so rather target the path/behavior patterns that lead there than just the result --> add additional penalties for it 


Main Problems: 
- too many suicides
- *too few kills*

e.g. mode3_model (seed 1)
(MLE) ~/Code/mle_final_project$ python evaluate.py --agent user_agent --rounds 100 --suite combat --timeout 0.5
[running] combat_3rule ...
[done]    combat_3rule in 120.9s
[running] combat_mixed ...
[done]    combat_mixed in 104.4s
[running] combat_vs_oldmodel ...
[done]    combat_vs_oldmodel in 187.2s

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate         0.33
    mean_rank        1.94
    score            3.93
    kills/round      0.15
    suicides/round   0.30
    survival_rate    0.57
    steps_survived   285
    coin_share       0.35

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate         0.42
    mean_rank        1.67
    score            4.02
    kills/round      0.10
    suicides/round   0.31
    survival_rate    0.59
    steps_survived   300
    coin_share       0.39

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate         0.28
    mean_rank        1.98
    score            3.64
    kills/round      0.19
    suicides/round   0.24
    survival_rate    0.58
    steps_survived   288
    coin_share       0.30


model_b1
=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.33
    mean_rank          2.03
    score              4.01
    kills/round        0.18
    suicides/round     0.34
    survival_rate      0.55
    steps_survived     290
    stuck_rate         0.49
    coin_share         0.35

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.48
    mean_rank          1.47
    score              4.62
    kills/round        0.14
    suicides/round     0.29
    survival_rate      0.63
    steps_survived     318
    stuck_rate         0.56
    coin_share         0.44

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.26
    mean_rank          2.11
    score              3.61
    kills/round        0.22
    suicides/round     0.39
    survival_rate      0.39
    steps_survived     256
    stuck_rate         0.44
    coin_share         0.28

model_b2
=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.29
    mean_rank          2.06
    score              3.24
    kills/round        0.04
    suicides/round     0.38
    survival_rate      0.51
    steps_survived     266
    stuck_rate         0.44
    coin_share         0.34

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.44
    mean_rank          1.60
    score              4.59
    kills/round        0.18
    suicides/round     0.29
    survival_rate      0.58
    steps_survived     296
    stuck_rate         0.49
    coin_share         0.41

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.30
    mean_rank          2.00
    score              3.70
    kills/round        0.19
    suicides/round     0.31
    survival_rate      0.41
    steps_survived     255
    stuck_rate         0.31
    coin_share         0.31


model_b3
=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.30
    mean_rank          2.12
    score              3.74
    kills/round        0.19
    suicides/round     0.43
    survival_rate      0.47
    steps_survived     269
    stuck_rate         0.39
    coin_share         0.31

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.38
    mean_rank          1.71
    score              3.75
    kills/round        0.10
    suicides/round     0.37
    survival_rate      0.59
    steps_survived     300
    stuck_rate         0.49
    coin_share         0.36

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.29
    mean_rank          2.03
    score              4.11
    kills/round        0.30
    suicides/round     0.44
    survival_rate      0.38
    steps_survived     252
    stuck_rate         0.22
    coin_share         0.29