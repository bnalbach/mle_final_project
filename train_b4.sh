#!/usr/bin/env bash
# b4 final run to teach hunting
set -e
AGENT=user_agent
OLD=training_partner          # old model (b4) as a sparring opponent
mkdir -p snapshots results

stage () {
  local name="$1"; local eps="$2"; local rounds="$3"; shift 3
  echo ">>> $name: eps=$eps rounds=$rounds vs $*"
  INIT_EPSILON="$eps" python main.py play --no-gui --scenario classic \
      --train 1 --n-rounds "$rounds" --agents "$AGENT" "$@"
  cp agent_code/$AGENT/dqn_model.pt snapshots/b4_${name}.pt
  echo ">>> eval after $name"
  python evaluate.py --agent $AGENT --suite combat --rounds 50 --seed 1 \
      --timeout 0.5 --out results/b4_${name}.json
}

# 2. save hunt
stage hunt_safe   0.30  8000 peaceful_agent coin_collector_agent coin_collector_agent
# 2. real danger
stage danger_2rule 0.20 10000 rule_based_agent rule_based_agent
# 3. tournament
stage full_3rule   0.12 16000 rule_based_agent rule_based_agent rule_based_agent
# 4. tournament mixed
stage mixed_3opp   0.10 12000 rule_based_agent coin_collector_agent "$OLD"

echo ">>> DONE. Compare each snapshot against b1 and keep the best:"
for st in hunt_safe danger_2rule full_3rule mixed_3opp; do
  echo "    python compare.py results/b1.json results/b4_${st}.json"
done
echo "    # to use a snapshot:  cp snapshots/b4_<best>.pt agent_code/$AGENT/dqn_model.pt"