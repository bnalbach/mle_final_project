import os
import pickle
from collections import deque

import numpy as np


ACTIONS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    #"WAIT",
]

MOVE_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    #"WAIT": (0, 0),
}

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "q_table.pkl",
)


def _inside(field, position):
    x, y = position
    return (
        0 <= x < field.shape[0]
        and 0 <= y < field.shape[1]
    )


def valid_actions(game_state):
    field = game_state["field"]
    _, _, _, position = game_state["self"]
    others = {
        item[3]
        for item in game_state["others"]
    }

    x, y = position
    actions = []

    for action in ACTIONS:
        dx, dy = MOVE_DELTAS[action]
        target = (x + dx, y + dy)

        if action == "WAIT":
            actions.append(action)
            continue

        if not _inside(field, target):
            continue

        tx, ty = target

        if field[tx, ty] != 0:
            continue

        if target in others:
            continue

        actions.append(action)

    return actions


def _reachable_distances(field, start):
    distances = {start: 0}
    queue = deque([start])

    while queue:
        x, y = queue.popleft()

        for dx, dy in MOVE_DELTAS.values():
            if dx == 0 and dy == 0:
                continue

            neighbor = (x + dx, y + dy)

            if not _inside(field, neighbor):
                continue

            if neighbor in distances:
                continue

            nx, ny = neighbor

            if field[nx, ny] != 0:
                continue

            distances[neighbor] = distances[(x, y)] + 1
            queue.append(neighbor)

    return distances


def _nearest_coin_information(game_state):
    field = game_state["field"]
    _, _, _, position = game_state["self"]
    coins = game_state["coins"]

    if not coins:
        return 0, 0, 0

    distances = _reachable_distances(field, position)

    reachable_coins = [
        (distances[coin], coin)
        for coin in coins
        if coin in distances
    ]

    if not reachable_coins:
        return 0, 0, 0

    distance, target = min(reachable_coins)

    x, y = position
    tx, ty = target

    dx = np.clip(tx - x, -5, 5)
    dy = np.clip(ty - y, -5, 5)

    return int(dx), int(dy), min(distance, 10)


def _distance_to_nearest_coin(field, start, coins):
    if not coins:
        return 0

    distances = _reachable_distances(field, start)

    reachable_distances = [
        distances[coin]
        for coin in coins
        if coin in distances
    ]

    if not reachable_distances:
        return 99

    return min(reachable_distances)


def state_to_features(game_state):
    field = game_state["field"]
    _, _, _, position = game_state["self"]
    coins = game_state["coins"]

    x, y = position
    valid = set(valid_actions(game_state))

    blocked = []
    action_distances = []

    for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
        dx, dy = MOVE_DELTAS[action]
        target = (x + dx, y + dy)

        is_valid = action in valid
        blocked.append(int(not is_valid))

        if not is_valid:
            action_distances.append(99)
            continue

        distance = _distance_to_nearest_coin(
            field,
            target,
            coins,
        )

        action_distances.append(min(distance, 10))

    coin_dx, coin_dy, coin_distance = (
        _nearest_coin_information(game_state)
    )

    return (
        tuple(blocked),
        tuple(action_distances),
        coin_dx,
        coin_dy,
        coin_distance,
        min(len(coins), 5),
    )


def setup(self):
    self.logger.info(
        "Setting up coin-collection Q-learning agent"
    )

    if self.train:
        self.q_table = {}
    else:
        try:
            with open(MODEL_PATH, "rb") as file:
                self.q_table = pickle.load(file)

            self.logger.info(
                "Loaded trained Q-table with %d states",
                len(self.q_table),
            )

        except FileNotFoundError:
            self.logger.warning(
                "No trained model found; using empty Q-table"
            )
            self.q_table = {}


def _q_values(self, state):
    if state not in self.q_table:
        self.q_table[state] = {
            action: 0.0
            for action in ACTIONS
        }

    return self.q_table[state]


def act(self, game_state):
    state = state_to_features(game_state)
    actions = valid_actions(game_state)
    q_values = _q_values(self, state)

    if self.train and np.random.random() < self.epsilon:
        return np.random.choice(actions)

    best_value = max(
        q_values[action]
        for action in actions
    )

    best_actions = [
        action
        for action in actions
        if q_values[action] == best_value
    ]

    if self.train:
        return np.random.choice(best_actions)

    return next(
        action
        for action in ACTIONS
        if action in best_actions
    )