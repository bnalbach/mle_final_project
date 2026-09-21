import os
import random
import shutil
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
    _bomb_has_any_safe_direction,
    _bomb_would_hit_crate,
    _can_escape_active_bombs,
    _count_crates_bomb_would_hit,
    _direction_has_safe_reachable_after_bomb,
    _distance_to_nearest_bomb,
    _distance_to_nearest_crate,
    _maximum_bomb_timer_at_position,
    _minimum_bomb_timer_at_position,
    _nearest_coin,
    _nearest_crate_target,
    _position_can_survive,
    _reachable_with_first_step,
    load_compatible_model_state,
    _nearest_opponent,
    _bomb_would_hit_position,
    state_to_features,
    valid_actions,
)

# --------------------------------------------------------------------------
# Curriculum training mode
# --------------------------------------------------------------------------
TRAINING_MODE = 3

CURRICULUM_TRANSITION_EPSILON = 0.05

# --------------------------------------------------------------------------
# DQN hyperparameters (identical across all curriculum modes)
# --------------------------------------------------------------------------
GAMMA = 0.95
LEARNING_RATE = 1e-4

REPLAY_BUFFER_CAPACITY = 50_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 1_000
TARGET_UPDATE_EVERY = 500
GRAD_CLIP_NORM = 10.0

EPSILON_START = 0.5
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9997

USE_DOUBLE_DQN = True

# --------------------------------------------------------------------------
# Debug instrumentation for PENALTY_LOST_ESCAPE_ROUTE / survivability.
# --------------------------------------------------------------------------
DEBUG_LOG_EVERY_N_DEATHS = 10

# --------------------------------------------------------------------------
# Reward/penalty constants always active (survival, movement, coins)
# --------------------------------------------------------------------------
PENALTY_STEP = -0.001

REWARD_COIN_COLLECTED = 10.0
REWARD_COIN_DIRECTION = 0.2
REWARD_COIN_FOUND = 0.2

REWARD_MOVED_CLOSER = 1.5
PENALTY_MOVED_AWAY_FROM_COIN = -0.75
PENALTY_NO_COIN_PROGRESS = -0.1

PENALTY_INVALID_ACTION = -5.0
PENALTY_WAIT = -0.2

PENALTY_KILLED_SELF = -60.0
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

DIVERSITY_WINDOW = 4
MIN_UNIQUE_TILES_FOR_NO_PENALTY = 3
PENALTY_LOW_MOVEMENT_DIVERSITY = -0.4

# --------------------------------------------------------------------------
# Reward/penalty constants gated by TRAINING_MODE >= 2 (crates and bombs)
# --------------------------------------------------------------------------
if TRAINING_MODE >= 2:
    REWARD_MOVED_TOWARD_CRATE = 0.05
    REWARD_CRATE_DIRECTION = 0.01
    PENALTY_MOVED_AWAY_FROM_CRATE = -0.25
    PENALTY_NO_CRATE_PROGRESS = -0.02

    # --------------------------------------------------------------------
    # Bomb-placement reward.
    #
    # The crate reward is predicted and paid IMMEDIATELY at placement time
    # (via _count_crates_bomb_would_hit, exact under TRAINING_MODE 1/2
    # since only one bomb can ever be active at once, and now correctly
    # counting ALL crates the blast would hit -- including multiple crates
    # stacked in the same direction, matching the official engine's blast
    # propagation which only stops at walls, not crates). It is GATED on
    # safety: an unsafe bomb gets only the mild baseline penalty plus a
    # harsher unsafe-specific penalty and NO crate reward, no matter how
    # many crates it would hit -- so it can no longer be bailed out by a
    # high crate count. A safe bomb gets the baseline penalty plus the
    # full per-crate reward.
    # --------------------------------------------------------------------
    PENALTY_BOMB_BASELINE = 0  # mild, applies to every bomb placement
    REWARD_SAFE_CRATE_HIT = 0.75  # per crate, only paid if the bomb was safe
    PENALTY_BOMB_NO_SAFE_ESCAPE = -12.0  # additional, on top of baseline,
    # if no safe escape exists.
    PENALTY_BOMB_NEAR_COIN = -5.0

    REWARD_MOVED_AWAY_FROM_BOMB = 0.15
    PENALTY_MOVED_TOWARD_BOMB = -0.5
    PENALTY_NO_BOMB_DISTANCE_PROGRESS = -0.05

    REWARD_MOVED_OUT_OF_BLAST = 1.0
    REWARD_ESCAPED_SECOND_LAST_TURN = 2.0
    REWARD_ESCAPED_LAST_TURN = 5.0

    PENALTY_MOVED_INTO_BLAST = -3.0
    PENALTY_ENTERED_LAST_TURN_BLAST = -6.0
    PENALTY_STAYING_IN_BLAST = -1.0

    REWARD_CHOSE_SAFE_DIRECTION = 0.5
    PENALTY_IGNORED_SAFE_DIRECTION = -4.0

    PENALTY_LOST_ESCAPE_ROUTE = -20.0
else:
    REWARD_MOVED_TOWARD_CRATE = 0.0
    REWARD_CRATE_DIRECTION = 0.0
    PENALTY_MOVED_AWAY_FROM_CRATE = 0.0
    PENALTY_NO_CRATE_PROGRESS = 0.0

    PENALTY_BOMB_BASELINE = 0.0
    REWARD_SAFE_CRATE_HIT = 0.0
    PENALTY_BOMB_NO_SAFE_ESCAPE = 0.0
    PENALTY_BOMB_NEAR_COIN = 0.0

    REWARD_MOVED_AWAY_FROM_BOMB = 0.0
    PENALTY_MOVED_TOWARD_BOMB = 0.0
    PENALTY_NO_BOMB_DISTANCE_PROGRESS = 0.0

    REWARD_MOVED_OUT_OF_BLAST = 0.0
    REWARD_ESCAPED_SECOND_LAST_TURN = 0.0
    REWARD_ESCAPED_LAST_TURN = 0.0

    PENALTY_MOVED_INTO_BLAST = 0.0
    PENALTY_ENTERED_LAST_TURN_BLAST = 0.0
    PENALTY_STAYING_IN_BLAST = 0.0

    REWARD_CHOSE_SAFE_DIRECTION = 0.0
    PENALTY_IGNORED_SAFE_DIRECTION = 0.0
    PENALTY_LOST_ESCAPE_ROUTE = 0.0

# --------------------------------------------------------------------------
# Reward/penalty constants gated by TRAINING_MODE >= 3 (opponents)
# --------------------------------------------------------------------------
if TRAINING_MODE >= 3:
    REWARD_KILLED_OPPONENT = 15.0
    REWARD_OPPONENT_ELIMINATED = 1.0
    PENALTY_GOT_KILLED = -30.0          # dying WHILE engaging is less bad than self-suicide (-60)
    REWARD_BOMB_HITS_OPPONENT = 6.0     # safe bomb whose blast covers an opponent (paid at placement)
    REWARD_MOVED_TOWARD_OPPONENT = 0.15 # armed + safe + closed distance to nearest opponent
else:
    REWARD_KILLED_OPPONENT = 0.0
    REWARD_OPPONENT_ELIMINATED = 0.0
    PENALTY_GOT_KILLED = PENALTY_KILLED_SELF
    REWARD_BOMB_HITS_OPPONENT = 0.0
    REWARD_MOVED_TOWARD_OPPONENT = 0.0


class ReplayBuffer:
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
    self.bombs_dropped_this_round = 0
    self.crates_destroyed_this_round = 0
    self.useless_bombs_this_round = 0
    self.killed_self_this_round = 0
    self.got_killed_this_round = 0
    self.position_history = []
    self.consecutive_backtracks = 0

    self.survivability_trail_this_round = []
    self.last_survivable_before = None
    self.death_count_for_debug_sampling = getattr(self, "death_count_for_debug_sampling", 0)

    if not hasattr(self, "recent_positions"):
        self.recent_positions = deque(maxlen=POSITION_HISTORY_LENGTH)

    self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

    self.target_net = QNetwork(self.policy_net.net[0].in_features).to(DEVICE)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
    self.loss_fn = nn.SmoothL1Loss()

    self.gradient_steps = 0
    self.policy_net.train()

    self.epsilon = EPSILON_START
    self.training_rounds = 0

    if os.path.isfile(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self.epsilon = checkpoint.get("epsilon", EPSILON_START)
        self.training_rounds = checkpoint.get("training_rounds", 0)
        checkpoint_mode = checkpoint.get("training_mode", TRAINING_MODE)

        self.logger.info(
            "Resumed training state: round=%d, epsilon=%.4f, checkpoint_mode=%d",
            self.training_rounds,
            self.epsilon,
            checkpoint_mode,
        )

        if checkpoint_mode != TRAINING_MODE:
            old_epsilon = self.epsilon
            self.epsilon = max(self.epsilon, CURRICULUM_TRANSITION_EPSILON)
            self.logger.info(
                "Curriculum mode changed (%d -> %d): boosting epsilon %.4f -> %.4f",
                checkpoint_mode,
                TRAINING_MODE,
                old_epsilon,
                self.epsilon,
            )

    # Per-stage exploration override for the curriculum script (train_b3.sh).
    _init_eps = os.environ.get("INIT_EPSILON")
    if _init_eps is not None:
        self.epsilon = float(_init_eps)
        self.logger.info("INIT_EPSILON override -> epsilon=%.4f", self.epsilon)

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


def _bomb_actually_dropped(events):
    bomb_dropped_event = getattr(e, "BOMB_DROPPED", None)
    return bomb_dropped_event is not None and bomb_dropped_event in events


def _points_toward(game_state, action, target):
    if target is None:
        return False

    if action not in MOVE_DELTAS:
        return False

    field = game_state["field"]
    position = game_state["self"][3]

    _, first_step = _reachable_with_first_step(field, position)

    return first_step.get(target) == action


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
    recent_window = position_history[-DIVERSITY_WINDOW:] + [new_position]

    if len(recent_window) < DIVERSITY_WINDOW + 1:
        return False

    return len(set(recent_window)) < MIN_UNIQUE_TILES_FOR_NO_PENALTY


def _bomb_placement_reward(old_game_state, new_game_state, new_position):
    """Computes the full bomb-placement reward immediately, combining the
    predicted crate count (now correctly counting multiple crates per
    direction -- see _count_crates_bomb_would_hit's docstring for the
    blast-propagation fix) with a hard safety gate:

      reward = PENALTY_BOMB_BASELINE                      (always, mild)
             + (REWARD_SAFE_CRATE_HIT * predicted_crates)  (ONLY if safe)
             + PENALTY_BOMB_NO_SAFE_ESCAPE                 (ONLY if unsafe)

    Returns (reward, predicted_crates, is_safe).
    """
    predicted_crates = _count_crates_bomb_would_hit(old_game_state)
    is_safe = _can_escape_active_bombs(new_game_state, new_position)

    reward = PENALTY_BOMB_BASELINE

    if is_safe:
        reward += predicted_crates * REWARD_SAFE_CRATE_HIT
    else:
        reward += PENALTY_BOMB_NO_SAFE_ESCAPE

    return reward, predicted_crates, is_safe


def _active_bomb_escape_reward(
    self,
    old_game_state,
    old_position,
    new_game_state,
    new_position,
    self_action,
):
    """Assign feedback to the action that preserves or loses an escape route.

    The states immediately before and after the same action are compared. A
    penalty therefore belongs to the exact movement (or WAIT) that changes a
    previously survivable situation into an unsurvivable one.
    """
    deadline = _minimum_bomb_timer_at_position(old_game_state, old_position)
    if deadline == 99:
        return 0.0

    was_survivable = _can_escape_active_bombs(old_game_state, old_position)
    is_survivable = _can_escape_active_bombs(new_game_state, new_position)

    if was_survivable and not is_survivable:
        reward = PENALTY_LOST_ESCAPE_ROUTE
    elif was_survivable and is_survivable and self_action != "WAIT":
        reward = REWARD_CHOSE_SAFE_DIRECTION
    else:
        reward = 0.0

    self.survivability_trail_this_round.append(
        (deadline, was_survivable, is_survivable, self_action, reward)
    )
    return reward


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

    old_timer = _minimum_bomb_timer_at_position(old_game_state, old_position)

    if e.COIN_COLLECTED in events:
        reward += REWARD_COIN_COLLECTED
        self.coins_collected_this_round += 1

    coins_found = _number_of_coins_found(events)
    if coins_found > 0:
        reward += coins_found * REWARD_COIN_FOUND

    # Bookkeeping only -- no reward here. The reward is paid immediately
    # at bomb-placement time (see _bomb_placement_reward).
    crates_destroyed = _number_of_destroyed_crates(events)
    if crates_destroyed > 0:
        self.crates_destroyed_this_round += crates_destroyed

    opponents_killed = _number_of_killed_opponents(events)
    if opponents_killed > 0:
        reward += opponents_killed * REWARD_KILLED_OPPONENT

    opponents_eliminated = _number_of_opponents_eliminated(events)
    if opponents_eliminated > 0:
        reward += opponents_eliminated * REWARD_OPPONENT_ELIMINATED

    coins_visible_before_bomb = bool(old_game_state.get("coins", []))

    if coins_visible_before_bomb:
        coin_distance_before_bomb = _distance_to_coin(old_game_state)
    else:
        coin_distance_before_bomb = 99

    if self_action == "BOMB" and _bomb_actually_dropped(events):
        bomb_reward, predicted_crates, is_safe = _bomb_placement_reward(
            old_game_state, new_game_state, new_position
        )
        reward += bomb_reward

        self.bombs_dropped_this_round += 1

        if predicted_crates == 0:
            self.useless_bombs_this_round += 1

        # Offensive shaping: a SAFE bomb whose blast would catch an opponent.
        if is_safe:
            _, _opp = _nearest_opponent(old_game_state)
            if _opp is not None and _bomb_would_hit_position(
                old_game_state["field"], old_position, _opp
            ):
                reward += REWARD_BOMB_HITS_OPPONENT

        if coin_distance_before_bomb <= 1:
            reward += PENALTY_BOMB_NEAR_COIN

        safe_directions_before = {
            direction: _direction_has_safe_reachable_after_bomb(
                old_game_state,
                old_position,
                direction,
            )
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]
        }

        pre_safe = any(safe_directions_before.values())

        safe_after_drop = _can_escape_active_bombs(
            new_game_state,
            new_position,
        )

        if pre_safe != safe_after_drop:
            self.logger.warning(
                "BOMB SAFETY MISMATCH position=%s predicted_crates=%d "
                "pre_safe=%s safe_dirs_before=%s safe_after_drop=%s "
                "bombs_before=%s bombs_after=%s",
                old_position,
                predicted_crates,
                pre_safe,
                safe_directions_before,
                safe_after_drop,
                old_game_state.get("bombs", []),
                new_game_state.get("bombs", []),
            )

    if _killed_self(events):
        reward += PENALTY_KILLED_SELF
        self.killed_self_this_round += 1
    elif _got_killed(events):
        reward += PENALTY_GOT_KILLED
        self.got_killed_this_round += 1

    if terminal_transition:
        if _survived_round(events):
            reward += REWARD_SURVIVED_ROUND

        return reward

    if e.INVALID_ACTION in events:
        reward += PENALTY_INVALID_ACTION

    coins_visible = bool(new_game_state.get("coins", []))

    if coins_visible and old_timer == 99:
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

    old_crate_distance = _distance_to_nearest_crate(old_game_state)
    new_crate_distance = _distance_to_nearest_crate(new_game_state)

    if old_timer == 99:
        if new_crate_distance < old_crate_distance:
            reward += REWARD_MOVED_TOWARD_CRATE

            _, target = _nearest_crate_target(old_game_state)

            if _points_toward(old_game_state, self_action, target):
                reward += REWARD_CRATE_DIRECTION

        elif new_crate_distance > old_crate_distance:
            reward += PENALTY_MOVED_AWAY_FROM_CRATE
        elif new_crate_distance < 99:
            reward += PENALTY_NO_CRATE_PROGRESS

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

    # Offensive shaping: close in on the nearest opponent while armed and not
    # under an active bomb threat (old_game_state['self'][2] is bombs_left).
    if self_action in MOVE_DELTAS and old_timer == 99 and old_game_state["self"][2]:
        _old_opp_dist, _ = _nearest_opponent(old_game_state)
        _new_opp_dist, _ = _nearest_opponent(new_game_state)
        if _old_opp_dist < 99 and _new_opp_dist < 99 and _new_opp_dist < _old_opp_dist:
            reward += REWARD_MOVED_TOWARD_OPPONENT

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

    if self_action in MOVE_DELTAS:
        reward += _active_bomb_escape_reward(
            self,
            old_game_state,
            old_position,
            new_game_state,
            new_position,
            self_action,
        )

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

    if low_diversity:
        reward += PENALTY_LOW_MOVEMENT_DIVERSITY

    return reward


def _optimize_model(self):
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
    old_features = state_to_features(old_game_state, recent_positions=recent_positions_before)
    action_index = ACTIONS.index(self_action)

    if terminal:
        next_features = old_features
        next_mask = np.zeros(len(ACTIONS), dtype=np.float32)
    else:
        next_features = state_to_features(new_game_state, recent_positions=recent_positions_after)
        next_mask = _action_mask(valid_actions(new_game_state))

    self.replay_buffer.push(old_features, action_index, reward, next_features, float(terminal), next_mask)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
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


def _log_survivability_trail(self, events):
    if not _killed_self(events):
        return

    self.death_count_for_debug_sampling += 1

    if self.death_count_for_debug_sampling % DEBUG_LOG_EVERY_N_DEATHS != 0:
        return

    trail = self.survivability_trail_this_round

    if not trail:
        self.logger.info(
            "[survivability-debug] death #%d: no bomb-threat steps recorded this round.",
            self.death_count_for_debug_sampling,
        )
        return

    last_steps = trail[-6:]
    any_penalty_fired = any(reward == PENALTY_LOST_ESCAPE_ROUTE for _, _, _, _, reward in trail)
    ever_flipped_to_unsurvivable = any(before and not after for _, before, after, _, _ in trail)

    self.logger.info(
        "[survivability-debug] death #%d: %d bomb-threat steps, "
        "ever unsurvivable: %s, PENALTY_LOST_ESCAPE_ROUTE fired: %s. "
        "Last steps (deadline, before, after, action, reward): %s",
        self.death_count_for_debug_sampling,
        len(trail),
        ever_flipped_to_unsurvivable,
        any_penalty_fired,
        last_steps,
    )

    if not ever_flipped_to_unsurvivable:
        self.logger.info(
            "[survivability-debug] death #%d: SUSPECTED BLIND SPOT -- "
            "no recorded action changed a survivable state into an unsurvivable one, yet the agent still died.",
            self.death_count_for_debug_sampling,
        )
    elif not any_penalty_fired:
        self.logger.info(
            "[survivability-debug] death #%d: FLIPPED TO UNSURVIVABLE BUT PENALTY NEVER "
            "FIRED -- the transition happened before the visible trail window, or on the "
            "very first bomb-threat step of the round (no prior reading to compare against).",
            self.death_count_for_debug_sampling,
        )


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

    deaths_this_round = self.killed_self_this_round + self.got_killed_this_round

    _log_survivability_trail(self, events)

    self.logger.info(
        "Round %d; mode=%d; epsilon=%.4f; coins=%d; reward=%.2f; buffer=%d; grad_steps=%d; "
        "bombs=%d; crates=%d; useless_bombs=%d; deaths=%d; killed_self=%d; got_killed=%d",
        self.training_rounds,
        TRAINING_MODE,
        self.epsilon,
        self.coins_collected_this_round,
        self.total_reward_this_round,
        len(self.replay_buffer),
        self.gradient_steps,
        self.bombs_dropped_this_round,
        self.crates_destroyed_this_round,
        self.useless_bombs_this_round,
        deaths_this_round,
        self.killed_self_this_round,
        self.got_killed_this_round,
    )

    if os.path.isfile(MODEL_PATH):
        try:
            shutil.copy(MODEL_PATH, MODEL_PATH + ".bak")
        except OSError:
            pass

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
    self.bombs_dropped_this_round = 0
    self.crates_destroyed_this_round = 0
    self.useless_bombs_this_round = 0
    self.killed_self_this_round = 0
    self.got_killed_this_round = 0
    self.position_history = []
    self.consecutive_backtracks = 0
    self.survivability_trail_this_round = []
    self.last_survivable_before = None
    self.recent_positions.clear()
