base model: dqn-mode3-initial
(this model: initially named model_b1)

script changes:
- callbacks.py:
- training.py:

training run:
python main.py play --no-gui --agents dqn-mode3-explore rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario classic --n-rounds 3000

eval:
=== move_solo_coinheaven  [coin-heaven]  vs none  (n=100) ===
    survival_rate      1.00
    suicides/round     0.00
    stuck_rate         0.39
    action_loop_rate   0.154
    pos_loop_rate      0.121
    invalid_rate       0.000
    wait_rate          0.002
    tile_diversity     0.60
    coins              49.1

=== move_vs_peaceful  [classic]  vs peaceful_agent, peaceful_agent  (n=100) ===
    survival_rate      0.97
    suicides/round     0.03
    stuck_rate         0.64
    action_loop_rate   0.107
    pos_loop_rate      0.071
    invalid_rate       0.000
    wait_rate          0.049
    tile_diversity     0.33
    coins              8.6

=== coin_solo  [coin-heaven]  vs none  (n=100) ===
    coins              49.1/50
    collect_rate       0.98
    coins/step         0.224
    steps_alive        219
    stuck_rate         0.39
    suicides/round     0.00
    invalid_rate       0.000

=== coin_vs_collectors  [coin-heaven]  vs coin_collector_agent, coin_collector_agent, coin_collector_agent  (n=100) ===
    coin_share         0.21
    coins              10.8
    win_rate           0.08
    stuck_rate         0.97
    survival_rate      0.97

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.28
    mean_rank          2.04
    score              3.68
    kills/round        0.18
    suicides/round     0.33
    survival_rate      0.53
    steps_survived     286
    stuck_rate         0.49
    coin_share         0.31

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.50
    mean_rank          1.62
    score              4.34
    kills/round        0.11
    suicides/round     0.33
    survival_rate      0.56
    steps_survived     300
    stuck_rate         0.48
    coin_share         0.42

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.27
    mean_rank          2.21
    score              3.54
    kills/round        0.19
    suicides/round     0.29
    survival_rate      0.49
    steps_survived     272
    stuck_rate         0.52
    coin_share         0.29