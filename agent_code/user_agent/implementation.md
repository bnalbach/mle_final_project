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

e.g.
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