Crate reward 2.0 → 0.75. Crates were 100–160 pts/round vs a 5-pt kill — combat was noise. Now farming pays less.
Kill reward 5 → 15, opponent-eliminated 0.5 → 1.0.
New REWARD_BOMB_HITS_OPPONENT = 6.0 — paid at placement when the bomb is safe and its blast covers an opponent (dense offensive signal, not the rare terminal kill).
New REWARD_MOVED_TOWARD_OPPONENT = 0.15 — small, only when armed and not already in danger, so it learns to close in.
PENALTY_GOT_KILLED −60 → −30 — dying while engaging an opponent shouldn't be as punishing as blowing yourself up (that stays −60), or it'll be too timid to fight.
INIT_EPSILON env override — lets train_b3.sh set exploration per stage (0.50 → 0.35 → 0.25 → 0.15) so it re-explores each harder opponent set. Slower decay (0.9997) for the long run.

base: model_b1
train:
python evaluate.py --agent user_agent --suite combat --rounds 100 --seed 1 --timeout 0.5 --out results/b3_pre.json    # baseline = b1 weights
bash train_b3.sh    # overnight; saves every round, Ctrl-C anytime keeps latest
python evaluate.py --agent user_agent --suite combat --rounds 100 --seed 1 --timeout 0.5 --out results/b3_post.json
python compare.py results/b3_pre.json results/b3_post.json

possible improvements: run against other of my models (add it to train_b3.sh)

pre:
=== combat_3rule  [classic]  vs rule_based_agent, rule_based_agent, rule_based_agent  (n=100) ===
    win_rate           0.41
    mean_rank          1.89
    score              4.20
    kills/round        0.21
    suicides/round     0.43
    survival_rate      0.48
    steps_survived     268
    stuck_rate         0.44
    coin_share         0.35

=== combat_mixed  [classic]  vs rule_based_agent, coin_collector_agent, random_agent  (n=100) ===
    win_rate           0.49
    mean_rank          1.53
    score              4.50
    kills/round        0.14
    suicides/round     0.28
    survival_rate      0.63
    steps_survived     318
    stuck_rate         0.50
    coin_share         0.42

=== combat_vs_oldmodel  [classic]  vs rule_based_agent, rule_based_agent, user_agent  (n=100) ===
    win_rate           0.26
    mean_rank          2.11
    score              3.67
    kills/round        0.24
    suicides/round     0.31
    survival_rate      0.45
    steps_survived     268
    stuck_rate         0.34
    coin_share         0.27


post:
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


compare:
=== combat_3rule ===
    metric            before     after     delta
    outright_win       0.410     0.300    -0.110  -
    rank               1.890     2.120    +0.230  -
    score              4.200     3.740    -0.460  -
    kills              0.210     0.190    -0.020  -
    suicides           0.430     0.430    +0.000
    survived           0.480     0.470    -0.010  -
    steps_alive      267.540   268.520    +0.980  +
    stuck              0.440     0.390    -0.050  +
    coin_share         0.352     0.310    -0.042  -

=== combat_mixed ===
    metric            before     after     delta
    outright_win       0.490     0.380    -0.110  -
    rank               1.530     1.710    +0.180  -
    score              4.500     3.750    -0.750  -
    kills              0.140     0.100    -0.040  -
    suicides           0.280     0.370    +0.090  -
    survived           0.630     0.590    -0.040  -
    steps_alive      318.310   300.190   -18.120  -
    stuck              0.500     0.490    -0.010  +
    coin_share         0.422     0.362    -0.060  -

=== combat_vs_oldmodel ===
    metric            before     after     delta
    outright_win       0.260     0.290    +0.030  +
    rank               2.110     2.030    -0.080  +
    score              3.670     4.110    +0.440  +
    kills              0.240     0.300    +0.060  +
    suicides           0.310     0.440    +0.130  -
    survived           0.450     0.380    -0.070  -
    steps_alive      267.660   252.060   -15.600  -
    stuck              0.340     0.220    -0.120  +
    coin_share         0.275     0.290    +0.015  +


results: 
- bad: win_rate down, score down, suicides up
- good: kills up (was the main goal)