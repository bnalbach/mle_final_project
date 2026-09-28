# Network Architecture & Training Setup

## Network Architecture

**Type**: Fully-connected feedforward neural network (Multi-Layer Perceptron / MLP) — not a CNN or RNN, since the state is a flat engineered feature vector rather than raw grid/image input.

| Layer | Type | Size |
|---|---|---|
| Input | — | Feature vector length (~84–88 dims after escape-direction features) |
| Hidden 1 | `nn.Linear` | 256 units |
| Activation 1 | `nn.ReLU` | — |
| Hidden 2 | `nn.Linear` | 256 units |
| Activation 2 | `nn.ReLU` | — |
| Output | `nn.Linear` | 6 units (one Q-value per action: UP, DOWN, LEFT, RIGHT, WAIT, BOMB) |

- **2 hidden layers**, **256 neurons each**
- **ReLU** activations between all linear layers
- **No activation on the output layer** (Q-values are unbounded real numbers)

## Learning Algorithm

**Base algorithm**: Deep Q-Network (DQN), specifically the **Double DQN** variant (`USE_DOUBLE_DQN = True`)

- Action **selection** for the bootstrapped target uses the online/policy network (`self.policy_net`)
- Action **evaluation** (Q-value for that selected action) uses a separate frozen target network (`self.target_net`)
- Target network synced from the policy network every `TARGET_UPDATE_EVERY = 500` gradient steps
- This decoupling reduces the overestimation bias from using `max` over noisy Q-estimates for both selection and evaluation

## Optimizer

**Adam** (`torch.optim.Adam`), applied to `self.policy_net.parameters()`:

- Learning rate: `LEARNING_RATE = 1e-4`
- Gradient clipping: `torch.nn.utils.clip_grad_norm_`, max norm `GRAD_CLIP_NORM = 10.0`
- Clipping applied after `loss.backward()`, before `optimizer.step()`

## Loss Function

**Smooth L1 loss** (Huber loss), via `nn.SmoothL1Loss()`

- Quadratic for small errors, linear for large ones
- More robust than plain MSE to reward outliers (e.g. `PENALTY_DIED = -15.0` combined with stacked penalties)

## Target Computation

Standard temporal-difference target with discounting:

```
y = r + gamma * Q_target(s', argmax_a' Q_policy(s', a')) * (1 - done)
```

- Discount factor: `GAMMA = 0.95`
- Invalid actions in the next state masked out before argmax/max (add `-1e9` to their Q-values)
- Terminal transitions (`done=1`) zero out the bootstrapped term, leaving just the immediate reward

## Experience Replay

- **Replay buffer**: fixed-size cyclic buffer (`collections.deque`, `maxlen=50_000`)
- Stores `(state, action, reward, next_state, done, next_valid_action_mask)` tuples
- **Batch size**: 64, sampled uniformly at random via `random.sample` (uniform replay, not prioritized)
- **Warm-up**: no gradient updates until buffer has at least `MIN_REPLAY_SIZE = 1,000` transitions

## Exploration Strategy

**Epsilon-greedy**, decayed exponentially per round (not per step):

- `EPSILON_START = 1.0`
- `EPSILON_MIN = 0.05`
- `EPSILON_DECAY = 0.9995` per round

Custom domain-biased exploration layered on top of vanilla epsilon-greedy:

- Excludes `BOMB` from random exploration if no safe escape exists
- Excludes `WAIT` from random exploration if a safe move is available and no bomb threatens the current tile

## Device

Auto-detected priority: `cuda` -> Intel Arc `xpu` -> `cpu`, via `_select_device()`. All tensors and both networks moved to that device via `.to(DEVICE)`.

## Summary Table

| Component | Choice |
|---|---|
| Network type | MLP, 2 hidden layers x 256 units, ReLU |
| Algorithm | Double DQN |
| Optimizer | Adam, lr = 1e-4 |
| Loss | Smooth L1 (Huber) |
| Discount (gamma) | 0.95 |
| Replay buffer | Uniform, capacity 50,000, batch 64 |
| Target network sync | Every 500 gradient steps |
| Exploration | Epsilon-greedy (1.0 -> 0.05, decay 0.9995/round) with domain-specific action masking |
| Gradient clipping | Max norm 10.0 |
