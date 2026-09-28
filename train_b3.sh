#!/usr/bin/env bash

# Usage bash train_b3.sh
set -e
AGENT=user_agent 

run () {  # $1 = INIT_EPSILON, $2 = n_rounds, rest = opponents
  local eps="$1"; local rounds="$2"; shift 2
  echo ">>> stage: eps=$eps rounds=$rounds opponents: $*"
  INIT_EPSILON="$eps" python main.py play --no-gui --scenario classic \
      --train 1 --n-rounds "$rounds" --agents "$AGENT" "$@"
}

# 1. non-bombing opponent: learn to approach + bomb it w/o dying
run 0.50 4000 coin_collector_agent
# 2. 1 real fighter: real kills and real danger appear
run 0.35 6000 rule_based_agent
# 3. 2 fighters
run 0.25 6000 rule_based_agent rule_based_agent
# 4. full tournament density
run 0.15 8000 rule_based_agent rule_based_agent rule_based_agent

echo ">>> done. Evaluate with:  python evaluate.py --agent $AGENT --suite combat --rounds 100 --timeout 0.5"
