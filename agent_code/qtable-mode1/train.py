import pickle

import events as e

from .callbacks import (
    ACTIONS,
    MODEL_PATH,
    _nearest_coin_information,
    _q_values,
    state_to_features,
    valid_actions,
)


# ============================================================
# Learning hyperparameters
# ============================================================

ALPHA = 0.15
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


# ============================================================
# Reward and penalty configuration
# ============================================================

PENALTY_STEP = -0.01

REWARD_COIN_COLLECTED = 10.0
REWARD_ALL_COINS_COLLECTED = 0.0

REWARD_MOVED_CLOSER = 1.0
PENALTY_MOVED_FARTHER = -2.0
PENALTY_NO_DISTANCE_PROGRESS = -0.2

PENALTY_INVALID_ACTION = -5.0
PENALTY_WAIT = -0.2

PENALTY_REVISITED_POSITION = -0.1
PENALTY_IMMEDIATE_BACKTRACK = -0.5

REWARD_SURVIVED_WITHOUT_COINS = 0.0
PENALTY_SURVIVED_WITH_COINS_REMAINING = 0.0


# ============================================================
# Training setup
# ============================================================

def setup_training(self):
    self.q_table = {}
    self.epsilon = EPSILON_START
    self.training_rounds = 0

    self.coins_collected_this_round = 0
    self.total_reward_this_round = 0.0

    self.position_history = []


# ============================================================
# State and reward helpers
# ============================================================

def _distance_to_coin(game_state):
    if not game_state["coins"]:
        return 0

    _, _, distance = _nearest_coin_information(
        game_state
    )

    return distance


def reward_from_transition(
    self,
    old_game_state,
    new_game_state,
    events,
    self_action=None,
    terminal_transition=False,
):
    reward = 0.0

    new_position = new_game_state["self"][3]

    was_revisited = (
        new_position in self.position_history
    )

    was_immediate_backtrack = (
        len(self.position_history) >= 2
        and new_position == self.position_history[-2]
    )

    # Update history for normal transitions only.
    # The final end_of_round callback passes identical states,
    # so it should not create an artificial revisit.
    if not terminal_transition:
        self.position_history.append(new_position)
        self.position_history = self.position_history[-6:]

    # A small cost encourages faster coin collection.
    if not terminal_transition:
        reward += PENALTY_STEP

    # Actual coin collection has priority over shaping rewards.
    if e.COIN_COLLECTED in events:
        reward += REWARD_COIN_COLLECTED

        if len(new_game_state["coins"]) == 0:
            reward += REWARD_ALL_COINS_COLLECTED

        return reward

    # Terminal transition without a coin collection.
    if terminal_transition:
        return reward

    if e.INVALID_ACTION in events:
        reward += PENALTY_INVALID_ACTION

    old_distance = _distance_to_coin(
        old_game_state
    )

    new_distance = _distance_to_coin(
        new_game_state
    )

    if new_distance < old_distance:
        reward += REWARD_MOVED_CLOSER

    elif new_distance > old_distance:
        reward += PENALTY_MOVED_FARTHER

    else:
        reward += PENALTY_NO_DISTANCE_PROGRESS

    if (
        self_action == "WAIT"
        or e.WAITED in events
    ) and new_game_state["coins"]:
        reward += PENALTY_WAIT

    #if was_revisited:
    #    reward += PENALTY_REVISITED_POSITION

    if was_immediate_backtrack:
        reward += PENALTY_IMMEDIATE_BACKTRACK

    return reward


# ============================================================
# Q-learning update
# ============================================================

def _update_q_value(
    self,
    old_state,
    action,
    reward,
    new_state,
    new_game_state=None,
    terminal=False,
):
    # WAIT is only an emergency fallback and is not learned.
    if action not in ACTIONS:
        return

    old_q_values = _q_values(
        self,
        old_state,
    )

    current_q = old_q_values[action]

    if terminal:
        target = reward

    else:
        next_q_values = _q_values(
            self,
            new_state,
        )

        next_actions = valid_actions(
            new_game_state
        )

        if next_actions:
            best_next_value = max(
                next_q_values[next_action]
                for next_action in next_actions
            )

            target = (
                reward
                + GAMMA * best_next_value
            )

        else:
            target = reward

    old_q_values[action] += ALPHA * (
        target - current_q
    )


# ============================================================
# Training callbacks
# ============================================================

def game_events_occurred(
    self,
    old_game_state,
    self_action,
    new_game_state,
    events,
):
    old_state = state_to_features(
        old_game_state
    )

    new_state = state_to_features(
        new_game_state
    )

    reward = reward_from_transition(
        self,
        old_game_state,
        new_game_state,
        events,
        self_action=self_action,
        terminal_transition=False,
    )

    if e.COIN_COLLECTED in events:
        self.coins_collected_this_round += 1

    self.total_reward_this_round += reward

    terminal = (
        len(new_game_state["coins"]) == 0
    )

    _update_q_value(
        self,
        old_state,
        self_action,
        reward,
        new_state,
        new_game_state=new_game_state,
        terminal=terminal,
    )


def end_of_round(
    self,
    last_game_state,
    last_action,
    events,
):
    last_state = state_to_features(
        last_game_state
    )

    # Do not apply distance or loop shaping to the artificial
    # final transition, because old and new states are identical.
    reward = reward_from_transition(
        self,
        last_game_state,
        last_game_state,
        events,
        self_action=last_action,
        terminal_transition=True,
    )

    if e.SURVIVED_ROUND in events:
        if last_game_state["coins"]:
            reward += (
                PENALTY_SURVIVED_WITH_COINS_REMAINING
            )
        else:
            reward += REWARD_SURVIVED_WITHOUT_COINS

    self.total_reward_this_round += reward

    _update_q_value(
        self,
        last_state,
        last_action,
        reward,
        last_state,
        terminal=True,
    )

    self.training_rounds += 1

    self.epsilon = max(
        EPSILON_MIN,
        self.epsilon * EPSILON_DECAY,
    )

    self.logger.info(
        "Round %d; epsilon=%.4f; states=%d; "
        "coins=%d; reward=%.2f",
        self.training_rounds,
        self.epsilon,
        len(self.q_table),
        self.coins_collected_this_round,
        self.total_reward_this_round,
    )

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(self.q_table, file)

    self.coins_collected_this_round = 0
    self.total_reward_this_round = 0.0
    self.position_history = []