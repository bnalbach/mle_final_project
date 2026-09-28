base model: dqn-mode3-explore
(this model: initially model_b2)

fix being stuck in loops: Deterministic argmax ties made it oscillate; now ties prefer an unvisited tile (12-step memory) and randomize otherwise

train: /; just callbacks.py changed
test: python evaluate.py --agent agent_b2 --suite combat --rounds 100 --seed 1 --timeout 0.5

eval:
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
    survival_rate      0.97
    suicides/round     0.03
    stuck_rate         0.43
    action_loop_rate   0.044
    pos_loop_rate      0.021
    invalid_rate       0.000
    wait_rate          0.036
    tile_diversity     0.35
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
    coin_share         0.22
    coins              11.2
    win_rate           0.09
    stuck_rate         0.99
    survival_rate      0.99

=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.31
    mean_rank          2.01
    score              3.52
    kills/round        0.11
    suicides/round     0.29
    survival_rate      0.64
    steps_survived     317
    stuck_rate         0.58
    coin_share         0.33

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.49
    mean_rank          1.58
    score              4.55
    kills/round        0.14
    suicides/round     0.27
    survival_rate      0.66
    steps_survived     321
    stuck_rate         0.48
    coin_share         0.43

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.27
    mean_rank          2.11
    score              3.70
    kills/round        0.23
    suicides/round     0.37
    survival_rate      0.45
    steps_survived     283
    stuck_rate         0.44
    coin_share         0.28