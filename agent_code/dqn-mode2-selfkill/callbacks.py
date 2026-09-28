import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import settings as s

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]

MOVE_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "WAIT": (0, 0),
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "dqn_model.pt")

BOMB_RANGE = s.BOMB_POWER

# Number of steps a bomb takes to explode after being placed. Used by
# _direction_has_safe_reachable_after_bomb (see fix below) to make sure a
# direction is only reported "safe" if a safe tile is reachable IN TIME,
# not just reachable at all regardless of how long it takes.
BOMB_TIMER = getattr(s, "BOMB_TIMER", 4)

# How many recent own positions to remember for the "recently visited"
# features below. Kept short on purpose: this is meant to break immediate
# back-and-forth oscillation (A -> B -> A -> B ...), not to encode a long
# trajectory history.
POSITION_HISTORY_LENGTH = 4


def _select_device():
    """Prefer CUDA, then Intel Arc (xpu), then fall back to CPU.

    Intel Arc integrated/discrete graphics are exposed via PyTorch's native
    'xpu' backend (PyTorch >= 2.5, installed from the xpu wheel index).
    torch.cuda.is_available() is always False on Intel hardware, so it's
    checked first only for portability if this code ever runs elsewhere.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")


DEVICE = _select_device()


# --------------------------------------------------------------------------
# Neural network Q-function (replaces the per-action linear weight vectors)
# --------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Small MLP mapping the engineered feature vector to Q-values for all actions.

    This is a direct generalisation of the old linear model
    (Q(s, a) = weights[a] @ features): instead of one weight vector per
    action, we have one shared network with one output neuron per action,
    which lets the model represent non-linear interactions between features
    (e.g. "danger AND no escape AND crate nearby") that a linear model cannot.
    """

    def __init__(self, input_dim, num_actions=len(ACTIONS), hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def _inside(field, position):
    x, y = position
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1]


def _bombs_with_timers(game_state):
    return [(tuple(bomb[0]), int(bomb[1])) for bomb in game_state.get("bombs", [])]


def _bomb_positions(game_state):
    return [position for position, _ in _bombs_with_timers(game_state)]


def _bomb_blast_cells(field, bomb_position):
    cells = {tuple(bomb_position)}
    x, y = bomb_position

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for distance in range(1, BOMB_RANGE + 1):
            target = (x + dx * distance, y + dy * distance)

            if not _inside(field, target):
                break

            cells.add(target)

            if field[target[0], target[1]] != 0:
                break

    return cells


def _all_blast_cells(game_state):
    cells = set()
    field = game_state["field"]

    for bomb_position in _bomb_positions(game_state):
        cells.update(_bomb_blast_cells(field, bomb_position))

    return cells


def _danger_cells(game_state):
    cells = _all_blast_cells(game_state)
    explosion_map = game_state.get("explosion_map")

    if explosion_map is not None:
        cells.update(zip(*np.where(explosion_map > 0)))

    return cells


def _minimum_bomb_timer_at_position(game_state, position):
    field = game_state["field"]
    timers = []

    for bomb_position, timer in _bombs_with_timers(game_state):
        if position in _bomb_blast_cells(field, bomb_position):
            timers.append(timer)

    return min(timers) if timers else 99


def _bomb_would_hit_crate(game_state):
    field = game_state["field"]
    x, y = game_state["self"][3]

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for distance in range(1, BOMB_RANGE + 1):
            target = (x + dx * distance, y + dy * distance)

            if not _inside(field, target):
                break

            value = field[target[0], target[1]]

            if value == 1:
                return True

            if value == -1:
                break

    return False


def _reachable_distances(field, start, blocked=None):
    blocked = set() if blocked is None else set(blocked)

    distances = {start: 0}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()

        for dx, dy in MOVE_DELTAS.values():
            if dx == 0 and dy == 0:
                continue

            target = (x + dx, y + dy)

            if not _inside(field, target):
                continue

            if target in distances:
                continue

            if target in blocked:
                continue

            if field[target[0], target[1]] != 0:
                continue

            distances[target] = distances[(x, y)] + 1
            queue.append(target)

    return distances


def _reachable_with_first_step(field, start, blocked=None):
    """BFS distance map that ALSO tracks the first-step direction.

    Returns (distances, first_step): first_step maps each reachable position
    to the direction ("UP"/"DOWN"/"LEFT"/"RIGHT") of the first move taken
    from `start` along the shortest walkable path to it (None for `start`
    itself). This lets callers use the direction of the actual path to a
    target instead of the straight-line direction, which can point directly
    into a wall/crate when the real path requires a detour.
    """
    blocked = set() if blocked is None else set(blocked)

    distances = {start: 0}
    first_step = {start: None}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()

        for direction, (dx, dy) in MOVE_DELTAS.items():
            if dx == 0 and dy == 0:
                continue

            target = (x + dx, y + dy)

            if not _inside(field, target):
                continue

            if target in distances:
                continue

            if target in blocked:
                continue

            if field[target[0], target[1]] != 0:
                continue

            distances[target] = distances[(x, y)] + 1
            first_step[target] = direction if (x, y) == start else first_step[(x, y)]
            queue.append(target)

    return distances, first_step


def _bomb_has_safe_escape(game_state):
    field = game_state["field"]
    position = game_state["self"][3]

    dangerous = _danger_cells(game_state) | _bomb_blast_cells(field, position)

    others = {item[3] for item in game_state.get("others", [])}

    reachable = _reachable_distances(field, position, others)

    return any(target != position and target not in dangerous for target in reachable)


def _direction_has_safe_reachable(game_state, position, direction):
    field = game_state["field"]
    dx, dy = MOVE_DELTAS[direction]

    first_step = (position[0] + dx, position[1] + dy)

    if not _inside(field, first_step):
        return False

    if field[first_step[0], first_step[1]] != 0:
        return False

    dangerous = _danger_cells(game_state)

    if first_step in dangerous:
        return False

    others = {item[3] for item in game_state.get("others", [])}

    reachable = _reachable_distances(field, first_step, others)

    return any(tile not in dangerous for tile in reachable)


def _direction_has_safe_reachable_after_bomb(game_state, position, direction, max_steps=None):
    """True if stepping `direction` from `position` (right after placing a
    bomb there) leads to a tile from which a SAFE tile is reachable WITHIN
    max_steps further moves.

    FIX (time-blind safety check): previously this ran an unbounded BFS and
    returned True as soon as ANY safe tile was reachable at ANY distance --
    it never compared that distance to how long the just-placed bomb takes
    to explode (BOMB_TIMER). That meant a direction could be reported "safe"
    even if the nearest actual safe tile was, say, 6 steps away while the
    bomb explodes in 4 -- the agent would walk that way, still be inside the
    blast radius when it goes off, and die, despite every safety feature
    having said "this way is fine".

    max_steps defaults to BOMB_TIMER - 1: after this first step, the agent
    has BOMB_TIMER - 1 further steps before the bomb goes off (the placement
    step itself, plus this first step, plus max_steps more = BOMB_TIMER total
    steps elapsed at detonation), so it must reach a safe tile within that
    many additional moves, not merely "eventually".
    """
    if max_steps is None:
        max_steps = BOMB_TIMER - 1

    field = game_state["field"]
    dx, dy = MOVE_DELTAS[direction]

    first_step = (position[0] + dx, position[1] + dy)

    if not _inside(field, first_step):
        return False

    if field[first_step[0], first_step[1]] != 0:
        return False

    # Existing danger plus the blast of the bomb we would place now
    dangerous = _danger_cells(game_state)
    dangerous |= _bomb_blast_cells(field, position)

    if first_step in dangerous:
        return False

    others = {item[3] for item in game_state.get("others", [])}

    reachable = _reachable_distances(field, first_step, others)

    return any(
        tile not in dangerous and distance <= max_steps
        for tile, distance in reachable.items()
    )


def valid_actions(game_state):
    field = game_state["field"]
    _, _, bombs_left, position = game_state["self"]

    others = {item[3] for item in game_state.get("others", [])}

    actions = ["WAIT"]

    if bombs_left:
        actions.append("BOMB")

    for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
        dx, dy = MOVE_DELTAS[action]
        target = (position[0] + dx, position[1] + dy)

        if not _inside(field, target):
            continue

        if field[target[0], target[1]] != 0:
            continue

        if target in others:
            continue

        actions.append(action)

    return actions


def _nearest_coin(game_state):
    position = game_state["self"][3]
    distances = _reachable_distances(game_state["field"], position)

    candidates = [
        (distances[coin], coin)
        for coin in game_state.get("coins", [])
        if coin in distances
    ]

    return min(candidates) if candidates else (99, None)


def _nearest_crate_target(game_state):
    field = game_state["field"]
    position = game_state["self"][3]

    distances = _reachable_distances(field, position)

    candidates = []

    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x, y] != 1:
                continue

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                target = (x + dx, y + dy)

                if target in distances:
                    candidates.append((distances[target], target))

    return min(candidates) if candidates else (99, None)


def _distance_to_nearest_bomb(game_state):
    position = game_state["self"][3]
    distances = _reachable_distances(game_state["field"], position)

    values = [distances[bomb] for bomb in _bomb_positions(game_state) if bomb in distances]

    return min(values) if values else 99


def _distance_to_nearest_crate(game_state):
    return _nearest_crate_target(game_state)[0]


_DIRECTION_TO_CODE = {"LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4}


def _path_direction_distance(game_state, position, target, distance):
    """Direction+level features based on the ACTUAL shortest walkable path
    to `target`, replacing the old straight-line-direction computation.
    """
    if target is None:
        return 0, 0

    field = game_state["field"]
    _, first_step = _reachable_with_first_step(field, position)

    step_direction = first_step.get(target)
    direction = _DIRECTION_TO_CODE.get(step_direction, 0)

    if distance <= 2:
        level = 1
    elif distance <= 5:
        level = 2
    elif distance < 99:
        level = 3
    else:
        level = 0

    return direction, level


def _one_hot(value, size):
    result = [0.0] * size

    if 0 <= value < size:
        result[value] = 1.0

    return result


def _recently_visited_directions(position, recent_positions):
    """For each of the four movement directions, check whether stepping that
    way would land on a tile the agent occupied within the last
    POSITION_HISTORY_LENGTH steps.
    """
    recent_set = set(recent_positions)
    flags = []

    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        dx, dy = MOVE_DELTAS[direction]
        neighbour = (position[0] + dx, position[1] + dy)
        flags.append(float(neighbour in recent_set))

    return flags


def state_to_features(game_state, recent_positions=None):
    field = game_state["field"]
    _, _, bombs_left, position = game_state["self"]

    valid = set(valid_actions(game_state))

    features = [1.0]

    for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
        features.append(float(action not in valid))

    danger = _danger_cells(game_state)

    for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
        dx, dy = MOVE_DELTAS[action]
        target = (position[0] + dx, position[1] + dy)
        features.append(float(target in danger))

    timer = _minimum_bomb_timer_at_position(game_state, position)
    danger_level = 3 if timer <= 1 else 2 if timer == 2 else 1 if timer < 99 else 0

    features.extend(_one_hot(danger_level, 4))
    features.extend(_one_hot(min(timer, 3), 4))

    explosion_value = game_state.get("explosion_map", np.zeros_like(field))[
        position[0], position[1]
    ]

    features.extend(_one_hot(min(int(explosion_value), 3), 4))

    features.append(float(_bomb_would_hit_crate(game_state)))
    features.append(float(_bomb_has_safe_escape(game_state)))

    safe_mask = 0

    for index, action in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
        dx, dy = MOVE_DELTAS[action]
        target = (position[0] + dx, position[1] + dy)
        safe_mask |= int(target not in danger) << index

    features.extend(_one_hot(safe_mask, 16))

    coin_distance, coin = _nearest_coin(game_state)
    crate_distance, crate = _nearest_crate_target(game_state)

    for target, distance in [(coin, coin_distance), (crate, crate_distance)]:
        direction, level = _path_direction_distance(game_state, position, target, distance)
        features.extend(_one_hot(direction, 5))
        features.extend(_one_hot(level, 4))

    features.extend(_one_hot(int(bool(bombs_left)), 2))

    # Directional safe-reachability features (for general movement: is this
    # direction safe given *current* dangers, ignoring any bomb we might
    # place ourselves right now).
    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        leads_safe = _direction_has_safe_reachable(game_state, position, direction)
        features.append(float(leads_safe))

    # Directional escape features *after dropping a bomb here*. Now
    # time-aware (see _direction_has_safe_reachable_after_bomb docstring):
    # only reports True if a safe tile is reachable before the bomb we'd
    # place right now actually goes off.
    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        safe_after_bomb = _direction_has_safe_reachable_after_bomb(
            game_state, position, direction
        )
        features.append(float(safe_after_bomb))

    # Short-term movement-history features.
    if recent_positions is None:
        recent_positions = []

    features.extend(_recently_visited_directions(position, recent_positions))

    return np.asarray(features, dtype=np.float32)


def _dummy_game_state():
    return {
        "field": np.zeros((1, 1), dtype=int),
        "self": ("agent", 0, False, (0, 0)),
        "coins": [],
        "bombs": [],
        "others": [],
        "explosion_map": np.zeros((1, 1)),
    }


def setup(self):
    """Load (or freshly initialise) the Q-network used for action selection."""
    self.epsilon = 0.0

    self.recent_positions = deque(maxlen=POSITION_HISTORY_LENGTH)

    input_dim = len(state_to_features(_dummy_game_state()))

    self.policy_net = QNetwork(input_dim).to(DEVICE)

    if os.path.isfile(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])

    self.policy_net.eval()

    self.logger.info("Using device: %s", DEVICE)


def q_values(self, features):
    """Single forward pass -> Q-values for every action (replaces weights[a] @ features)."""
    with torch.no_grad():
        features_tensor = torch.as_tensor(features, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        outputs = self.policy_net(features_tensor).squeeze(0).cpu().numpy()

    return {action: float(outputs[i]) for i, action in enumerate(ACTIONS)}


def act(self, game_state):
    if not hasattr(self, "recent_positions"):
        self.recent_positions = deque(maxlen=POSITION_HISTORY_LENGTH)

    position = game_state["self"][3]

    features = state_to_features(game_state, recent_positions=self.recent_positions)
    actions = valid_actions(game_state)

    if self.train and np.random.random() < self.epsilon:
        if "BOMB" in actions and not _bomb_has_safe_escape(game_state):
            actions = [a for a in actions if a != "BOMB"]

        no_immediate_threat = _minimum_bomb_timer_at_position(game_state, position) == 99
        has_safe_move = any(
            _direction_has_safe_reachable(game_state, position, direction)
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]
        )

        if "WAIT" in actions and no_immediate_threat and has_safe_move and len(actions) > 1:
            actions = [a for a in actions if a != "WAIT"]

        chosen_action = np.random.choice(actions)
        self.recent_positions.append(position)
        return chosen_action

    values = q_values(self, features)

    best_value = max(values[action] for action in actions)
    best_actions = [action for action in actions if values[action] == best_value]

    if self.train:
        chosen_action = np.random.choice(best_actions)
    else:
        chosen_action = next(action for action in ACTIONS if action in best_actions)

    self.recent_positions.append(position)
    return chosen_action
