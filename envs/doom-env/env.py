"""DOOM RL Simulator Environment for Neurosymbolic RL training in Theta."""
from __future__ import annotations
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch as th
    from nudge.env import NudgeBaseEnv
except ImportError:
    th = None
    NudgeBaseEnv = object


class DoomSimGym:
    """Lightweight 2.5D DOOM tactical simulation gymnasium environment."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.max_steps = 300
        self.current_step = 0

        # State representation
        self.px = 2.0
        self.py = 2.0
        self.angle = 0.0
        self.hp = 100
        self.armor = 0
        self.ammo = 50

        # Demon target
        self.ex = 6.0
        self.ey = 2.0
        self.e_hp = 60
        self.e_alive = True

        # Item pickup
        self.item_x = 4.0
        self.item_y = 5.0
        self.item_collected = False

        # Door / Exit
        self.door_x = 8.0
        self.door_y = 2.0
        self.has_key = False
        self.exit_cleared = False

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            self.rng = random.Random(seed)
        self.current_step = 0
        self.px, self.py = 2.0, 2.0
        self.angle = 0.0
        self.hp = 100
        self.armor = 25
        self.ammo = 50

        self.ex, self.ey = 6.0, 2.0
        self.e_hp = 60
        self.e_alive = True

        self.item_x, self.item_y = 4.0, 5.0
        self.item_collected = False

        self.has_key = False
        self.exit_cleared = False

        return self._get_obs(), {}

    def _get_obs(self) -> Any:
        # Distance & angle to enemy
        dx_e = self.ex - self.px
        dy_e = self.ey - self.py
        dist_e = math.hypot(dx_e, dy_e)
        ang_e = (math.atan2(dy_e, dx_e) - self.angle + math.pi) % (2 * math.pi) - math.pi

        # Distance to item
        dx_i = self.item_x - self.px
        dy_i = self.item_y - self.py
        dist_i = math.hypot(dx_i, dy_i) if not self.item_collected else 10.0

        # Distance to exit
        dx_d = self.door_x - self.px
        dy_d = self.door_y - self.py
        dist_d = math.hypot(dx_d, dy_d)

        vals = [
            self.px / 10.0,
            self.py / 10.0,
            self.angle / (2 * math.pi),
            self.hp / 100.0,
            self.armor / 100.0,
            self.ammo / 50.0,
            dist_e / 10.0 if self.e_alive else 0.0,
            ang_e / math.pi if self.e_alive else 0.0,
            dist_i / 10.0,
            1.0 if not self.item_collected else 0.0,
            dist_d / 10.0,
            1.0 if self.exit_cleared else 0.0,
        ]
        return np.array(vals, dtype=np.float32) if np is not None else vals

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = -0.01  # time penalty

        # Action decoding: 0=Fwd, 1=Back, 2=TurnL, 3=TurnR, 4=Fire, 5=Use
        if action == 0:  # Forward
            self.px = max(1.0, min(9.0, self.px + math.cos(self.angle) * 0.4))
            self.py = max(1.0, min(9.0, self.py + math.sin(self.angle) * 0.4))
        elif action == 1:  # Backward
            self.px = max(1.0, min(9.0, self.px - math.cos(self.angle) * 0.4))
            self.py = max(1.0, min(9.0, self.py - math.sin(self.angle) * 0.4))
        elif action == 2:  # Turn Left
            self.angle = (self.angle - 0.25) % (2 * math.pi)
        elif action == 3:  # Turn Right
            self.angle = (self.angle + 0.25) % (2 * math.pi)
        elif action == 4:  # Fire Weapon
            if self.ammo > 0 and self.e_alive:
                self.ammo -= 1
                dx = self.ex - self.px
                dy = self.ey - self.py
                dist = math.hypot(dx, dy)
                target_ang = math.atan2(dy, dx)
                diff = abs((target_ang - self.angle + math.pi) % (2 * math.pi) - math.pi)
                if diff < 0.25 and dist < 7.0:
                    self.e_hp -= 25
                    reward += 20.0
                    if self.e_hp <= 0:
                        self.e_alive = False
                        reward += 100.0
        elif action == 5:  # Use Door / Exit
            if math.hypot(self.door_x - self.px, self.door_y - self.py) < 1.2:
                self.exit_cleared = True
                reward += 200.0

        # Item pickup check
        if not self.item_collected and math.hypot(self.item_x - self.px, self.item_y - self.py) < 0.8:
            self.item_collected = True
            self.hp = min(100, self.hp + 30)
            reward += 30.0

        # Demon attacks player if alive
        if self.e_alive:
            dist = math.hypot(self.ex - self.px, self.ey - self.py)
            if dist < 4.0 and self.rng.random() < 0.35:
                dmg = 12
                if self.armor > 0:
                    self.armor = max(0, self.armor - 6)
                    dmg -= 6
                self.hp -= dmg
                reward -= float(dmg)

        done = self.hp <= 0 or self.exit_cleared
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, done, truncated, {"exit_cleared": self.exit_cleared, "hp": self.hp}

    def close(self):
        pass


class NudgeEnv(NudgeBaseEnv):
    """Nudge / NeSyRL wrapper for DOOM RL simulator."""
    name = "doom"
    pred2action = {
        "forward": 0,
        "backward": 1,
        "turn_left": 2,
        "turn_right": 3,
        "fire": 4,
        "use": 5,
    }
    pred_names: Sequence

    def __init__(self, mode: str = "ppo", render_mode="rgb_array", seed=None, **kwargs):
        super().__init__(mode) if hasattr(super(), "__init__") else None
        self.env = DoomSimGym(seed=seed)
        self.n_actions = 6
        self.n_raw_actions = 6
        self.n_objects = 4  # [agent, enemy, item, door]
        self.n_features = 4
        self.seed = seed

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        logic_state = self.extract_logic_state(obs).unsqueeze(0) if th is not None else obs
        neural_state = self.extract_neural_state(obs)
        return logic_state, neural_state

    def step(self, action, is_mapped: bool = False):
        if hasattr(action, "item"):
            action = action.item()
        obs, reward, done, truncation, info = self.env.step(int(action))
        logic_state = self.extract_logic_state(obs).unsqueeze(0) if th is not None else obs
        neural_state = self.extract_neural_state(obs)
        return (logic_state, neural_state), reward, done, truncation, info

    def extract_logic_state(self, obs: Any) -> Any:
        if th is None:
            return obs
        state = th.zeros((self.n_objects, self.n_features), dtype=th.float32)
        # Obj 0: Agent [x, y, angle, hp]
        state[0] = th.tensor(obs[0:4], dtype=th.float32)
        # Obj 1: Enemy [dist, angle, alive, hp_ratio]
        state[1] = th.tensor([obs[6], obs[7], 1.0 if obs[6] > 0 else 0.0, obs[6]], dtype=th.float32)
        # Obj 2: Item [dist, available, 0, 0]
        state[2] = th.tensor([obs[8], obs[9], 0.0, 0.0], dtype=th.float32)
        # Obj 3: Exit Door [dist, cleared, 0, 0]
        state[3] = th.tensor([obs[10], obs[11], 0.0, 0.0], dtype=th.float32)
        return state

    def extract_neural_state(self, obs: Any) -> Any:
        if th is None:
            return obs
        return th.tensor(obs, dtype=th.float32).unsqueeze(0)

    def get_action_meanings(self) -> List[str]:
        return ["forward", "backward", "turn_left", "turn_right", "fire", "use"]

    def close(self):
        self.env.close()
