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

# Number of steps a bomb takes to explode after being placed.
BOMB_TIMER = getattr(s, "BOMB_TIMER", 4)

POSITION_HISTORY_LENGTH = 4


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")


DEVICE = _select_device()


class QNetwork(nn.Module):
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
    """Cells reachable by a bomb's blast, matching the OFFICIAL engine
    (items.py Bomb.get_blast_coords): propagation stops ONLY at a wall
    (field value == -1), NOT at a crate. A crate in the path is destroyed
    but does NOT block the blast from continuing further in that
    direction, up to BOMB_RANGE tiles total.

    FIX: the previous version stopped at ANY non-zero cell (`!= 0`), which
    incorrectly stopped at crates too -- exactly like a wall. That made
    every one of this codebase's safety checks (_danger_cells,
    _minimum_bomb_timer_at_position, _position_can_survive,
    _direction_has_safe_reachable_after_bomb, ...) UNDERESTIMATE the true
    blast radius whenever a crate stood between the bomb and a tile
    further away: that tile was wrongly classified as safe, when the real
    engine's explosion actually reaches straight through the crate (up to
    BOMB_RANGE tiles, or until a wall) and would still hit it.
    """
    cells = {tuple(bomb_position)}
    x, y = bomb_position

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for distance in range(1, BOMB_RANGE + 1):
            target = (x + dx * distance, y + dy * distance)

            if not _inside(field, target):
                break

            if field[target[0], target[1]] == -1:  # only WALLS stop the blast
                break

            cells.add(target)

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


def _maximum_bomb_timer_at_position(game_state, position):
    """Like _minimum_bomb_timer_at_position but returns the LARGEST timer
    among bombs threatening `position`, i.e. how many steps remain until
    the LAST bomb that could still hit this tile goes off.
    """
    field = game_state["field"]
    timers = []

    for bomb_position, timer in _bombs_with_timers(game_state):
        if position in _bomb_blast_cells(field, bomb_position):
            timers.append(timer)

    return max(timers) if timers else 99


def _bomb_would_hit_crate(game_state):
    """True if a bomb placed at the agent's current position would hit at
    least one crate anywhere along its (now correctly wall-only-stopped)
    blast path.
    """
    field = game_state["field"]
    x, y = game_state["self"][3]

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for distance in range(1, BOMB_RANGE + 1):
            target = (x + dx * distance, y + dy * distance)

            if not _inside(field, target):
                break

            value = field[target[0], target[1]]

            if value == -1:
                break

            if value == 1:
                return True

    return False


def _count_crates_bomb_would_hit(game_state):
    """Counts how many crates a bomb placed at the agent's current position
    would destroy. Matches the official engine's blast propagation: a
    crate does NOT stop the blast (only a wall does), so multiple crates
    stacked in the same direction (within BOMB_RANGE) are ALL hit and
    counted -- not just the first one.

    Used to credit REWARD_SAFE_CRATE_HIT immediately at bomb-placement
    time (see train.py's _bomb_placement_reward), instead of waiting for
    the delayed CRATE_DESTROYED event. Since the agent can only ever have
    one bomb active at a time, this prediction is exact in TRAINING_MODE
    1/2 (no opponents who could destroy the same crates first).
    """
    field = game_state["field"]
    x, y = game_state["self"][3]

    count = 0

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for distance in range(1, BOMB_RANGE + 1):
            target = (x + dx * distance, y + dy * distance)

            if not _inside(field, target):
                break

            value = field[target[0], target[1]]

            if value == -1:
                break

            if value == 1:
                count += 1

    return count


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
    """Return whether `direction` starts a route escaping a bomb placed now.

    BFS examines every free tile reachable before detonation. Tiles in the
    new bomb's future blast zone may be traversed before it explodes, but the
    final destination must be outside that blast zone.
    """
    if max_steps is None:
        # Placing the bomb consumes one engine tick. The resulting game state
        # exposes timer BOMB_TIMER - 1, so after the separately evaluated first
        # escape move only BOMB_TIMER - 2 additional moves remain.
        max_steps = BOMB_TIMER - 2

    field = game_state["field"]
    dx, dy = MOVE_DELTAS[direction]
    first_step = (position[0] + dx, position[1] + dy)

    if not _inside(field, first_step):
        return False
    if field[first_step[0], first_step[1]] != 0:
        return False

    existing_danger = _danger_cells(game_state)
    own_blast = _bomb_blast_cells(field, position)

    # Existing danger is unsafe now. own_blast is only unsafe at detonation.
    if first_step in existing_danger:
        return False

    others = {item[3] for item in game_state.get("others", [])}
    reachable = _reachable_distances(field, first_step, others)

    return any(
        tile not in existing_danger
        and tile not in own_blast
        and distance <= max_steps
        for tile, distance in reachable.items()
    )


def _bomb_has_any_safe_direction(game_state, position):
    """True if at least one of UP/DOWN/LEFT/RIGHT is a safe escape after
    placing a bomb at `position` right now.
    """
    return any(
        _direction_has_safe_reachable_after_bomb(game_state, position, direction)
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]
    )


def _position_can_survive(game_state, position, max_steps):
    """True if a tile outside ALL current danger zones is reachable from
    `position` within `max_steps` moves.
    """
    field = game_state["field"]

    dangerous = _danger_cells(game_state)

    if position not in dangerous:
        return True

    others = {item[3] for item in game_state.get("others", [])}

    reachable = _reachable_distances(field, position, others)

    return any(
        tile not in dangerous and distance <= max_steps
        for tile, distance in reachable.items()
    )



def _can_escape_active_bombs(game_state, position):
    """Whether `position` still has a route out before the earliest active
    bomb threatening it detonates.
    """
    deadline = _minimum_bomb_timer_at_position(game_state, position)
    if deadline == 99:
        return True
    return _position_can_survive(game_state, position, deadline)


def _direction_preserves_active_escape(game_state, position, direction):
    """Whether moving `direction` now preserves a timely route away from an
    already active bomb threatening `position`.
    """
    deadline = _minimum_bomb_timer_at_position(game_state, position)
    if deadline == 99:
        return _direction_has_safe_reachable(game_state, position, direction)

    if direction not in MOVE_DELTAS or direction == "WAIT":
        return False

    field = game_state["field"]
    dx, dy = MOVE_DELTAS[direction]
    first_step = (position[0] + dx, position[1] + dy)

    if not _inside(field, first_step) or field[first_step[0], first_step[1]] != 0:
        return False

    explosion_map = game_state.get("explosion_map")
    if explosion_map is not None and explosion_map[first_step[0], first_step[1]] > 0:
        return False

    dangerous = _danger_cells(game_state)
    others = {item[3] for item in game_state.get("others", [])}
    reachable = _reachable_distances(field, first_step, others)

    return any(
        tile not in dangerous and distance <= deadline - 1
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

    # Preserve the existing four feature slots. In an active bomb threat,
    # they directly encode whether each immediate direction preserves a timely
    # escape route; otherwise they retain the generic movement-safety meaning.
    active_bomb_threat = _minimum_bomb_timer_at_position(game_state, position) < 99

    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        if active_bomb_threat:
            leads_safe = _direction_preserves_active_escape(
                game_state, position, direction
            )
        else:
            leads_safe = _direction_has_safe_reachable(
                game_state, position, direction
            )
        features.append(float(leads_safe))

    for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
        safe_after_bomb = _direction_has_safe_reachable_after_bomb(
            game_state, position, direction
        )
        features.append(float(safe_after_bomb))

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
