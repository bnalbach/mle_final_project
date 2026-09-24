fix being stuck in loops: Deterministic argmax ties made it oscillate; now ties prefer an unvisited tile (12-step memory) and randomize otherwise

base: model_b1
train: /; just callbacks.py changed
test: python evaluate.py --agent agent_b2 --suite combat --rounds 100 --seed 1 --timeout 0.5

