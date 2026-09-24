import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import events as e

from .callbacks import (
    ACTIONS,
    DEVICE,
    MODEL_PATH,
    MOVE_DELTAS,
    POSITION_HISTORY_LENGTH,
    QNetwork,
    _bomb_would_hit_crate,
    _direction_has_safe_reachable_after_bomb,
    _distance_to_nearest_bomb,
    _distance_to_nearest_crate,
    _minimum_bomb_timer_at_position,
    _nearest_coin,
    _nearest_crate_target,
    state_to_features,
    valid_actions,
)

# --------------------------------------------------------------------------
# Curriculum training mode
# --------------------------------------------------------------------------
# 1 = coins only: crate-, bomb- and opponent-related rewards/penalties are
#     disabled (set to 0) so the agent learns pure movement/navigation and
#     coin-seeking without any competing signal. Use this on a coins-only
#     map first (e.g. --scenario coin-heaven).
# 2 = coins + crates: bomb-placement and crate-destruction rewards are
#     enabled on top of mode 1, so the agent learns when/where to bomb
#     without opponent-related signals yet.
# 3 = full game: opponent-related rewards (KILLED_OPPONENT, OPPONENT_ELIMINATED,
#     GOT_KILLED distinction) are enabled on top of mode 2.
#
# Change this single constant to switch curriculum stage; everything else
# (network, replay buffer, optimizer, feature extraction) stays identical
# across modes, so a checkpoint trained in mode 1 loads and keeps training
# seamlessly when you switch to mode 2 or 3.
TRAINING_MODE = 1

# --------------------------------------------------------------------------
# DQN hyperparameters (identical across all curriculum modes)
# --------------------------------------------------------------------------
GAMMA = 0.95
LEARNING_RATE = 1e-4

REPLAY_BUFFER_CAPACITY = 50_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 1_000          # steps to collect before training starts
TARGET_UPDATE_EVERY = 500        # sync target network every N training steps
GRAD_CLIP_NORM = 10.0

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

USE_DOUBLE_DQN = True

# --------------------------------------------------------------------------
# Reward/penalty constants always active (survival, movement, coins)
# --------------------------------------------------------------------------
PENALTY_STEP = -0.001

REWARD_COIN_COLLECTED = 5.0
REWARD_COIN_DIRECTION = 0.1
REWARD_COIN_FOUND = 0.5

REWARD_MOVED_CLOSER = 1.0
PENALTY_MOVED_AWAY_FROM_COIN = -0.5  # was -2.0, too dominant given frequent
                                      # nearest-coin target switches on maps
                                      # with many coins
PENALTY_NO_COIN_PROGRESS = -0.05

PENALTY_INVALID_ACTION = -5.0
PENALTY_WAIT = -0.2

PENALTY_KILLED_SELF = -15.0
REWARD_SURVIVED_ROUND = 2.0

PENALTY_REVISITED_POSITION = -0.0
REVISIT_PENALTY_GROWTH = -0.15
MAX_REVISIT_COUNT = 6

PENALTY_ENTERED_CORNER = -0.30
PENALTY_CORNER_BACKTRACK = -0.35

PENALTY_IMMEDIATE_BACKTRACK = -0.3
BACKTRACK_PENALTY_GROWTH = -0.3
MAX_BACKTRACK_STREAK = 6
PENALTY_DETECTED_LOOP = -0.5

# NEW: movement-diversity penalty. was_backtrack only catches the strict
# A -> B -> A pattern (new_position equals the position from exactly 2 steps
# ago), and was_revisited only counts how often a single tile reappears --
# neither directly penalises a small oscillation loop like A -> B -> C -> B
# that keeps the agent confined to just 2-3 tiles overall. This measures
# how many *distinct* tiles were visited in a short recent window and
# penalises low diversity directly, catching that class of behaviour
# regardless of the exact revisit pattern.
DIVERSITY_WINDOW = 4
MIN_UNIQUE_TILES_FOR_NO_PENALTY = 3  # need at least 3 distinct tiles among
                                      # the last DIVERSITY_WINDOW + 1 positions
PENALTY_LOW_MOVEMENT_DIVERSITY = -0.4

# --------------------------------------------------------------------------
# Reward/penalty constants gated by TRAINING_MODE >= 2 (crates and bombs)
# --------------------------------------------------------------------------
if TRAINING_MODE >= 2:
    REWARD_MOVED_TOWARD_CRATE = 0.05
    REWARD_CRATE_DIRECTION = 0.01
    PENALTY_MOVED_AWAY_FROM_CRATE = -0.25
    PENALTY_NO_CRATE_PROGRESS = -0.02

    REWARD_BOMB_TARGETED = 2.0
    PENALTY_USELESS_BOMB = -8.0
    PENALTY_BOMB_NEAR_COIN = -5.0
    PENALTY_BOMB_NO_RESULT = -2.0
    REWARD_BOMB_WITH_SAFE_ESCAPE = 0.0
    PENALTY_BOMB_NO_SAFE_ESCAPE = -5.0

    REWARD_MOVED_AWAY_FROM_BOMB = 0.15
    PENALTY_MOVED_TOWARD_BOMB = -0.5
    PENALTY_NO_BOMB_DISTANCE_PROGRESS = -0.05

    REWARD_MOVED_OUT_OF_BLAST = 1.0
    REWARD_ESCAPED_SECOND_LAST_TURN = 2.0
    REWARD_ESCAPED_LAST_TURN = 5.0

    PENALTY_MOVED_INTO_BLAST = -3.0
    PENALTY_ENTERED_LAST_TURN_BLAST = -6.0
    PENALTY_STAYING_IN_BLAST = -1.0

    REWARD_CRATE_DESTROYED = 5.0
else:
    # Mode 1: all crate/bomb-related signals neutralised. Bomb-danger
    # avoidance (blast-timer tracking) is still computed since bombs can
    # still exist on a coins-only map via other agents, but none of it
    # contributes reward in this mode so it can't distract from coin-seeking.
    REWARD_MOVED_TOWARD_CRATE = 0.0
    REWARD_CRATE_DIRECTION = 0.0
    PENALTY_MOVED_AWAY_FROM_CRATE = 0.0
    PENALTY_NO_CRATE_PROGRESS = 0.0

    REWARD_BOMB_TARGETED = 0.0
    PENALTY_USELESS_BOMB = 0.0
    PENALTY_BOMB_NEAR_COIN = 0.0
    PENALTY_BOMB_NO_RESULT = 0.0
    REWARD_BOMB_WITH_SAFE_ESCAPE = 0.0
    PENALTY_BOMB_NO_SAFE_ESCAPE = 0.0

    REWARD_MOVED_AWAY_FROM_BOMB = 0.0
    PENALTY_MOVED_TOWARD_BOMB = 0.0
    PENALTY_NO_BOMB_DISTANCE_PROGRESS = 0.0

    REWARD_MOVED_OUT_OF_BLAST = 0.0
    REWARD_ESCAPED_SECOND_LAST_TURN = 0.0
    REWARD_ESCAPED_LAST_TURN = 0.0

    PENALTY_MOVED_INTO_BLAST = 0.0
    PENALTY_ENTERED_LAST_TURN_BLAST = 0.0
    PENALTY_STAYING_IN_BLAST = 0.0

    REWARD_CRATE_DESTROYED = 0.0

# --------------------------------------------------------------------------
# Reward/penalty constants gated by TRAINING_MODE >= 3 (opponents)
# --------------------------------------------------------------------------
if TRAINING_MODE >= 3:
    REWARD_KILLED_OPPONENT = 5.0
    REWARD_OPPONENT_ELIMINATED = 0.5
    PENALTY_GOT_KILLED = -10.0
else:
    # Modes 1 and 2: no opponent-elimination incentive yet, and death by an
    # opponent is folded into the same penalty as a self-inflicted death so
    # the agent isn't taught anything opponent-specific before this stage.
    REWARD_KILLED_OPPONENT = 0.0
    REWARD_OPPONENT_ELIMINATED = 0.0
    PENALTY_GOT_KILLED = PENALTY_KILLED_SELF


class ReplayBuffer:
    """Fixed-size cyclic buffer of transitions, sampled uniformly at random.

    Breaks the correlation between consecutive Bomberman steps that would
    otherwise destabilise training if we updated on transitions in the
    order they occur (as the old linear model's online update did).
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_index, reward, next_state, done, next_valid_mask):
        self.buffer.append((state, action_index, reward, next_state, done, next_valid_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid_masks = zip(*batch)

        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
            np.stack(next_valid_masks),
        )

    def __len__(self):
        return len(self.buffer)


def _action_mask(actions_available):
    mask = np.zeros(len(ACTIONS), dtype=np.float32)

    for action in actions_available:
        mask[ACTIONS.index(action)] = 1.0

    return mask


def setup_training(self):
    self.coins_collected_this_round = 0
    self.total_reward_this_round = 0.0
    self.position_history = []
    self.consecutive_backtracks = 0
    self.pending_bomb = False

    # Mirrors self.recent_positions from callbacks.setup(), but train.py
    # needs its own before/after snapshots per transition to build the
    # "recent_positions at the time of old_game_state / new_game_state"
    # views used when re-deriving features for replay storage.
    if not hasattr(self, "recent_positions"):
        self.recent_positions = deque(maxlen=POSITION_HISTORY_LENGTH)

    self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    self.target_net = QNetwork(self.policy_net.net[0].in_features).to(DEVICE)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    self.loss_fn = nn.SmoothL1Loss()  # Huber loss: robust to reward outliers (e.g. -15 on death)

    self.gradient_steps = 0
    self.policy_net.train()

    # Restore epsilon and the round counter from the checkpoint if present,
    # so stopping and restarting training (including switching curriculum
    # mode) continues the exploration decay schedule instead of resetting.
    self.epsilon = EPSILON_START
    self.training_rounds = 0

    if os.path.isfile(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self.epsilon = checkpoint.get("epsilon", EPSILON_START)
        self.training_rounds = checkpoint.get("training_rounds", 0)
        self.logger.info(
            "Resumed training state: round=%d, epsilon=%.4f",
            self.training_rounds,
            self.epsilon,
        )

    self.logger.info("Curriculum TRAINING_MODE=%d", TRAINING_MODE)
    self.logger.info("Training on device: %s", DEVICE)
    if DEVICE.type == "xpu":
        self.logger.info(
            "Intel XPU device name: %s",
            torch.xpu.get_device_name(DEVICE.index or 0),
        )


def _distance_to_coin(game_state):
    return _nearest_coin(game_state)[0]


def _number_of_destroyed_crates(events):
    return sum(event == e.CRATE_DESTROYED for event in events)


def _number_of_killed_opponents(events):
    killed_opponent_event = getattr(e, "KILLED_OPPONENT", None)
    if killed_opponent_event is None:
        return 0
    return sum(event == killed_opponent_event for event in events)


def _number_of_coins_found(events):
    coin_found_event = getattr(e, "COIN_FOUND", None)
    if coin_found_event is None:
        return 0
    return sum(event == coin_found_event for event in events)


def _number_of_opponents_eliminated(events):
    opponent_eliminated_event = getattr(e, "OPPONENT_ELIMINATED", None)
    if opponent_eliminated_event is None:
        return 0
    return sum(event == opponent_eliminated_event for event in events)


def _killed_self(events):
    return getattr(e, "KILLED_SELF", None) in events


def _got_killed(events):
    return getattr(e, "GOT_KILLED", None) in events


def _survived_round(events):
    survived_event = getattr(e, "SURVIVED_ROUND", None)
    return survived_event is not None and survived_event in events


def _points_toward(game_state, action, target):
    if target is None:
        return False

    if action not in MOVE_DELTAS:
        return False

    x, y = game_state["self"][3]
    dx, dy = MOVE_DELTAS[action]

    target_dx = target[0] - x
    target_dy = target[1] - y

    if abs(target_dx) >= abs(target_dy):
        if target_dx == 0:
            return False

        return dx == (-1 if target_dx < 0 else 1) and dy == 0

    if target_dy == 0:
        return False

    return dy == (-1 if target_dy < 0 else 1) and dx == 0


def _corner_blocked_sides(game_state, position):
    field = game_state["field"]
    blocked_sides = 0

    x, y = position

    for dx, dy in MOVE_DELTAS.values():
        nx = x + dx
        ny = y + dy

        if nx < 0 or ny < 0 or nx >= field.shape[0] or ny >= field.shape[1]:
            blocked_sides += 1
            continue

        if field[nx, ny] != 0:
            blocked_sides += 1

    return blocked_sides


def _is_corner(game_state, position):
    return _corner_blocked_sides(game_state, position) >= 2


def _has_low_movement_diversity(position_history, new_position):
    """True if the last DIVERSITY_WINDOW positions plus new_position contain
    fewer than MIN_UNIQUE_TILES_FOR_NO_PENALTY distinct tiles.

    Unlike was_backtrack (only catches the strict A -> B -> A pattern) and
    was_revisited (only counts repeats of a single tile), this catches any
    tight oscillation confined to few tiles, e.g. A -> B -> C -> B -> C,
    regardless of the exact order in which tiles repeat.
    """
    recent_window = position_history[-DIVERSITY_WINDOW:] + [new_position]

    if len(recent_window) < DIVERSITY_WINDOW + 1:
        return False

    return len(set(recent_window)) < MIN_UNIQUE_TILES_FOR_NO_PENALTY


def reward_from_transition(
    self,
    old_game_state,
    new_game_state,
    events,
    self_action=None,
    terminal_transition=False,
):
    reward = 0.0

    old_position = old_game_state["self"][3]
    new_position = new_game_state["self"][3]

    recent_visit_count = self.position_history.count(new_position)
    was_revisited = recent_visit_count > 0

    was_backtrack = (
        len(self.position_history) >= 2
        and new_position == self.position_history[-2]
    )

    low_diversity = _has_low_movement_diversity(self.position_history, new_position)

    if not terminal_transition:
        self.position_history.append(new_position)

    self.position_history = self.position_history[-6:]

    reward += PENALTY_STEP

    # --------------------------------------------------------------------
    # Coin events -- always active, all curriculum modes.
    # --------------------------------------------------------------------
    if e.COIN_COLLECTED in events:
        reward += REWARD_COIN_COLLECTED
        self.coins_collected_this_round += 1

    coins_found = _number_of_coins_found(events)
    if coins_found > 0:
        reward += coins_found * REWARD_COIN_FOUND

    # --------------------------------------------------------------------
    # Crate / bomb events -- values are 0.0 in TRAINING_MODE 1, so this
    # block is always safe to evaluate but has no effect until mode >= 2.
    # --------------------------------------------------------------------
    crates_destroyed = _number_of_destroyed_crates(events)

    opponents_killed = _number_of_killed_opponents(events)
    if opponents_killed > 0:
        reward += opponents_killed * REWARD_KILLED_OPPONENT

    opponents_eliminated = _number_of_opponents_eliminated(events)
    if opponents_eliminated > 0:
        reward += opponents_eliminated * REWARD_OPPONENT_ELIMINATED

    bomb_exploded_event = getattr(e, "BOMB_EXPLODED", None)
    bomb_exploded = bomb_exploded_event is not None and bomb_exploded_event in events

    coins_visible_before_bomb = bool(old_game_state.get("coins", []))

    if coins_visible_before_bomb:
        coin_distance_before_bomb = _distance_to_coin(old_game_state)
    else:
        coin_distance_before_bomb = 99

    if self_action == "BOMB":
        bomb_targets_crate = _bomb_would_hit_crate(old_game_state)
        self.pending_bomb = True

        if bomb_targets_crate:
            reward += REWARD_BOMB_TARGETED
        else:
            reward += PENALTY_USELESS_BOMB

        if coin_distance_before_bomb <= 1:
            reward += PENALTY_BOMB_NEAR_COIN

        has_any_safe_direction = any(
            _direction_has_safe_reachable_after_bomb(old_game_state, old_position, direction)
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]
        )

        if has_any_safe_direction:
            reward += REWARD_BOMB_WITH_SAFE_ESCAPE
        else:
            reward += PENALTY_BOMB_NO_SAFE_ESCAPE

        if crates_destroyed > 0:
            reward += crates_destroyed * REWARD_CRATE_DESTROYED

        self.pending_bomb = False

    elif bomb_exploded and self.pending_bomb:
        reward += PENALTY_BOMB_NO_RESULT
        self.pending_bomb = False

    # --------------------------------------------------------------------
    # Death handling -- KILLED_SELF always penalised; GOT_KILLED equals
    # PENALTY_KILLED_SELF in modes 1/2 (no opponent-specific distinction
    # yet) and is set to its own, lighter value in mode 3.
    # --------------------------------------------------------------------
    if _killed_self(events):
        reward += PENALTY_KILLED_SELF

    if _got_killed(events):
        reward += PENALTY_GOT_KILLED

    if terminal_transition:
        if _survived_round(events):
            reward += REWARD_SURVIVED_ROUND

        return reward

    if e.INVALID_ACTION in events:
        reward += PENALTY_INVALID_ACTION

    coins_visible = bool(new_game_state.get("coins", []))

    if coins_visible:
        old_coin_distance = _distance_to_coin(old_game_state)
        new_coin_distance = _distance_to_coin(new_game_state)

        if new_coin_distance < old_coin_distance:
            reward += REWARD_MOVED_CLOSER

            _, target = _nearest_coin(old_game_state)

            if _points_toward(old_game_state, self_action, target):
                reward += REWARD_COIN_DIRECTION

        elif new_coin_distance > old_coin_distance:
            reward += PENALTY_MOVED_AWAY_FROM_COIN
        else:
            reward += PENALTY_NO_COIN_PROGRESS

    waited_event = getattr(e, "WAITED", None)

    if self_action == "WAIT" or (waited_event is not None and waited_event in events):
        reward += PENALTY_WAIT

    # --------------------------------------------------------------------
    # Crate-distance shaping -- no-ops in mode 1 (all constants are 0.0).
    # --------------------------------------------------------------------
    old_crate_distance = _distance_to_nearest_crate(old_game_state)
    new_crate_distance = _distance_to_nearest_crate(new_game_state)

    if new_crate_distance < old_crate_distance:
        reward += REWARD_MOVED_TOWARD_CRATE

        _, target = _nearest_crate_target(old_game_state)

        if _points_toward(old_game_state, self_action, target):
            reward += REWARD_CRATE_DIRECTION

    elif new_crate_distance > old_crate_distance:
        reward += PENALTY_MOVED_AWAY_FROM_CRATE
    elif new_crate_distance < 99:
        reward += PENALTY_NO_CRATE_PROGRESS

    # --------------------------------------------------------------------
    # Bomb-distance / blast-zone shaping -- no-ops in mode 1.
    # --------------------------------------------------------------------
    if self_action != "BOMB":
        old_bomb_distance = _distance_to_nearest_bomb(old_game_state)
        new_bomb_distance = _distance_to_nearest_bomb(new_game_state)

        if old_bomb_distance < 99 and new_bomb_distance < 99:
            if new_bomb_distance > old_bomb_distance:
                reward += REWARD_MOVED_AWAY_FROM_BOMB
            elif new_bomb_distance < old_bomb_distance:
                reward += PENALTY_MOVED_TOWARD_BOMB
            else:
                reward += PENALTY_NO_BOMB_DISTANCE_PROGRESS

    old_timer = _minimum_bomb_timer_at_position(old_game_state, old_position)
    new_timer = _minimum_bomb_timer_at_position(new_game_state, new_position)

    if self_action != "BOMB":
        moved_out = old_timer < 99 and new_timer == 99
        entered = old_timer == 99 and new_timer < 99

        if moved_out:
            if old_timer <= 1:
                reward += REWARD_ESCAPED_LAST_TURN
            elif old_timer == 2:
                reward += REWARD_ESCAPED_SECOND_LAST_TURN
            else:
                reward += REWARD_MOVED_OUT_OF_BLAST
        elif entered:
            if new_timer <= 1:
                reward += PENALTY_ENTERED_LAST_TURN_BLAST
            else:
                reward += PENALTY_MOVED_INTO_BLAST
        elif new_timer <= 1:
            reward += PENALTY_STAYING_IN_BLAST

    old_is_corner = _is_corner(old_game_state, old_position)
    new_is_corner = _is_corner(new_game_state, new_position)

    made_progress = e.COIN_COLLECTED in events or new_crate_distance < old_crate_distance

    if self_action in MOVE_DELTAS and old_timer == 99 and new_timer == 99:
        if new_is_corner and not old_is_corner and not made_progress:
            reward += PENALTY_ENTERED_CORNER

        if old_is_corner and new_is_corner and was_backtrack:
            reward += PENALTY_CORNER_BACKTRACK

    if was_revisited:
        revisit_level = min(recent_visit_count, MAX_REVISIT_COUNT)
        reward += PENALTY_REVISITED_POSITION + (revisit_level - 1) * REVISIT_PENALTY_GROWTH

        if recent_visit_count >= 2:
            reward += PENALTY_DETECTED_LOOP

    if was_backtrack:
        self.consecutive_backtracks = min(self.consecutive_backtracks + 1, MAX_BACKTRACK_STREAK)
        reward += PENALTY_IMMEDIATE_BACKTRACK + (self.consecutive_backtracks - 1) * BACKTRACK_PENALTY_GROWTH
    else:
        self.consecutive_backtracks = 0

    # NEW: penalise low movement diversity directly, independent of the
    # exact oscillation pattern (catches A-B-C-B loops that was_backtrack
    # misses, without double-penalising the strict A-B-A case captured
    # above -- both can fire together for a true 2-tile pendulum, which is
    # intentional since that is the worst case).
    if low_diversity:
        reward += PENALTY_LOW_MOVEMENT_DIVERSITY

    return reward


def _optimize_model(self):
    """Sample a batch from replay and take one gradient step (Double DQN target)."""
    if len(self.replay_buffer) < max(BATCH_SIZE, MIN_REPLAY_SIZE):
        return

    states, actions, rewards, next_states, dones, next_valid_masks = self.replay_buffer.sample(BATCH_SIZE)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
    actions_t = torch.as_tensor(actions, dtype=torch.int64, device=DEVICE)
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE)
    next_valid_masks_t = torch.as_tensor(next_valid_masks, dtype=torch.float32, device=DEVICE)

    current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Mask invalid actions out of the next-state max by pushing them to -inf
        invalid_penalty = (1.0 - next_valid_masks_t) * -1e9

        if USE_DOUBLE_DQN:
            next_online_q = self.policy_net(next_states_t) + invalid_penalty
            best_next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_states_t).gather(1, best_next_actions).squeeze(1)
        else:
            next_target_q = (self.target_net(next_states_t) + invalid_penalty).max(dim=1).values

        targets = rewards_t + GAMMA * next_target_q * (1.0 - dones_t)

    loss = self.loss_fn(current_q, targets)

    self.optimizer.zero_grad()
    loss.backward()
    if self.gradient_steps % 100 == 0:
        self.logger.info("grad_step=%d loss=%.4f", self.gradient_steps, loss.item())
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRAD_CLIP_NORM)
    self.optimizer.step()

    self.gradient_steps += 1

    if self.gradient_steps % TARGET_UPDATE_EVERY == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())


def _store_transition(
    self,
    old_game_state,
    self_action,
    reward,
    new_game_state,
    terminal,
    recent_positions_before,
    recent_positions_after,
):
    """recent_positions_before/after are snapshots (lists) of self.recent_positions
    taken immediately before and after act() ran for this transition, so the
    stored features match exactly what the network saw/will see when act()
    is called with the corresponding game states -- keeping training and
    inference consistent for the movement-history feature.
    """
    old_features = state_to_features(old_game_state, recent_positions=recent_positions_before)
    action_index = ACTIONS.index(self_action)

    if terminal:
        next_features = old_features  # unused when done=1, but keep shapes consistent
        next_mask = np.zeros(len(ACTIONS), dtype=np.float32)
    else:
        next_features = state_to_features(new_game_state, recent_positions=recent_positions_after)
        next_mask = _action_mask(valid_actions(new_game_state))

    self.replay_buffer.push(old_features, action_index, reward, next_features, float(terminal), next_mask)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    # self.recent_positions was already updated by act() for old_game_state
    # (the position *before* this action was taken is what act() appended).
    # For old_game_state's features we want the history as it stood when
    # act() was called for it, i.e. excluding old_game_state's own position;
    # for new_game_state's features we want the history including it.
    recent_positions_after = list(self.recent_positions)
    recent_positions_before = recent_positions_after[:-1] if recent_positions_after else []

    reward = reward_from_transition(
        self,
        old_game_state,
        new_game_state,
        events,
        self_action,
        False,
    )

    _store_transition(
        self,
        old_game_state,
        self_action,
        reward,
        new_game_state,
        terminal=False,
        recent_positions_before=recent_positions_before,
        recent_positions_after=recent_positions_after,
    )
    _optimize_model(self)

    self.total_reward_this_round += reward


def end_of_round(self, last_game_state, last_action, events):
    recent_positions_after = list(self.recent_positions)
    recent_positions_before = recent_positions_after[:-1] if recent_positions_after else []

    reward = reward_from_transition(
        self,
        last_game_state,
        last_game_state,
        events,
        last_action,
        True,
    )

    _store_transition(
        self,
        last_game_state,
        last_action,
        reward,
        last_game_state,
        terminal=True,
        recent_positions_before=recent_positions_before,
        recent_positions_after=recent_positions_after,
    )
    _optimize_model(self)

    self.total_reward_this_round += reward
    self.training_rounds += 1

    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    self.logger.info(
        "Round %d; mode=%d; epsilon=%.4f; coins=%d; reward=%.2f; buffer=%d; grad_steps=%d",
        self.training_rounds,
        TRAINING_MODE,
        self.epsilon,
        self.coins_collected_this_round,
        self.total_reward_this_round,
        len(self.replay_buffer),
        self.gradient_steps,
    )

    torch.save(
        {
            "model_state_dict": self.policy_net.state_dict(),
            "training_rounds": self.training_rounds,
            "epsilon": self.epsilon,
            "training_mode": TRAINING_MODE,
        },
        MODEL_PATH,
    )

    self.coins_collected_this_round = 0
    self.total_reward_this_round = 0.0
    self.position_history = []
    self.consecutive_backtracks = 0
    self.pending_bomb = False
    self.recent_positions.clear()
