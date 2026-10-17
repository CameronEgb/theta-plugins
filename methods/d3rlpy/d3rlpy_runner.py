"""Universal d3rlpy Offline RL Runner for Theta-IDE.

Executes d3rlpy offline reinforcement learning algorithms (CQL, IQL, TD3+BC, BC)
as a decoupled black-box engine, logging training telemetry through src.app.telemetry
and saving checkpoints adhering to the Theta-IDE execution contract.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.app import telemetry


@hydra.main(version_base=None, config_path="../../../in/config", config_name="config")
def main(cfg: DictConfig):
    try:
        import d3rlpy
    except ImportError:
        print(
            "\n[Error] d3rlpy is not installed in the active environment.\n"
            "Please run: pip install d3rlpy\n"
            "or install the d3rlpy Offline RL Suite via the Theta Hub.\n"
        )
        sys.exit(1)

    try:
        import gymnasium as gym
    except ImportError:
        print("\n[Error] Gymnasium is required for evaluation. Run: pip install gymnasium\n")
        sys.exit(1)

    import numpy as np

    group = str(cfg.get("group", "default_group"))
    exp_id = str(cfg.get("experiment_id", "default_exp"))
    agent_cfg = cfg.get("agent", {})
    agent_name = str(agent_cfg.get("name", "d3rlpy_agent"))
    algo_name = str(agent_cfg.get("algorithm", "CQL")).upper()
    n_steps = int(cfg.get("n_steps", agent_cfg.get("n_steps", 50000)))
    batch_size = int(agent_cfg.get("batch_size", 256))
    lr = float(agent_cfg.get("learning_rate", agent_cfg.get("lr", 3e-4)))

    # Resolve environment and discrete vs continuous
    env_cfg = cfg.get("env", {})
    env_name = env_cfg.get("name")
    env_id = env_cfg.get("env_id") or env_name or "CartPole-v1"
    if env_id == "cartpole":
        env_id = "CartPole-v1"
    elif env_id == "mountaincar":
        env_id = "MountainCar-v0"

    print(f"\n=== [d3rlpy Offline Engine] Initializing {algo_name} on {env_id} ===")
    print(f"Group: {group} | Experiment: {exp_id} | Total Steps: {n_steps}")

    # Telemetry initialization
    sinks, system_interval = telemetry.settings_from_config(cfg)
    logger = telemetry.init(
        group,
        exp_id,
        agent_name,
        sinks=sinks,
        config=OmegaConf.to_container(cfg, resolve=True),
        system_interval=system_interval,
    )

    ckpt_dir = Path("results/checkpoints") / group / exp_id / agent_name.replace("/", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset
    dataset_path_val = cfg.get("dataset_path") or cfg.get("dataset")
    if not dataset_path_val:
        # Fallback to standard offline dataset directory
        candidate_dir = Path("results/datasets") / group / exp_id
        if candidate_dir.exists():
            subdirs = [p for p in candidate_dir.iterdir() if p.is_dir()]
            if subdirs:
                dataset_path_val = str(subdirs[0])
            else:
                dataset_path_val = str(candidate_dir)

    observations = []
    actions = []
    rewards = []
    terminals = []

    if dataset_path_val and Path(dataset_path_val).exists():
        from src.app.dataset_utils import DatasetReader

        reader = DatasetReader(dataset_path_val)
        print(f"[d3rlpy Engine] Loading transitions from {dataset_path_val} ({len(reader)} items)...")
        for i in range(len(reader)):
            item = reader[i]
            if isinstance(item, dict):
                observations.append(item["obs"])
                actions.append(item["action"])
                rewards.append(item["reward"])
                terminals.append(item.get("done", False))
            elif isinstance(item, (list, tuple)) and len(item) >= 5:
                observations.append(item[0])
                actions.append(item[1])
                rewards.append(item[2])
                terminals.append(item[4])

    if not observations:
        print("[d3rlpy Engine] No offline dataset found; synthesizing bootstrap exploration buffer...")
        # Create a small bootstrap buffer from the environment
        test_env = gym.make(env_id)
        obs, _ = test_env.reset(seed=int(cfg.get("seed", 42)))
        for _ in range(min(1000, n_steps)):
            act = test_env.action_space.sample()
            next_obs, rew, term, trunc, _ = test_env.step(act)
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            terminals.append(term or trunc)
            obs = next_obs if not (term or trunc) else test_env.reset()[0]
        test_env.close()

    obs_arr = np.array(observations, dtype=np.float32)
    act_arr = np.array(actions)
    rew_arr = np.array(rewards, dtype=np.float32)
    term_arr = np.array(terminals, dtype=bool)

    # Determine action type (discrete vs continuous)
    is_discrete = np.issubdtype(act_arr.dtype, np.integer) or len(act_arr.shape) == 1
    if is_discrete:
        act_arr = act_arr.astype(np.int64)
    else:
        act_arr = act_arr.astype(np.float32)

    try:
        # d3rlpy v2 dataset format
        from d3rlpy.dataset import MDPDataset, ReplayBuffer, InfiniteBuffer
        dataset = ReplayBuffer(
            InfiniteBuffer(),
            observations=obs_arr,
            actions=act_arr,
            rewards=rew_arr,
            terminals=term_arr,
        )
    except Exception:
        try:
            from d3rlpy.dataset import MDPDataset
            dataset = MDPDataset(
                observations=obs_arr,
                actions=act_arr,
                rewards=rew_arr,
                terminals=term_arr,
                discrete_action=is_discrete,
            )
        except Exception:
            dataset = None

    # Algorithm selection
    algo = None
    if algo_name in ("CQL", "CONSERVATIVE_Q_LEARNING"):
        if is_discrete and hasattr(d3rlpy.algos, "DiscreteCQLConfig"):
            algo = d3rlpy.algos.DiscreteCQLConfig(learning_rate=lr, batch_size=batch_size).create()
        elif hasattr(d3rlpy.algos, "CQLConfig"):
            algo = d3rlpy.algos.CQLConfig(learning_rate=lr, batch_size=batch_size).create()
    elif algo_name in ("IQL", "IMPLICIT_Q_LEARNING"):
        if hasattr(d3rlpy.algos, "IQLConfig"):
            algo = d3rlpy.algos.IQLConfig(learning_rate=lr, batch_size=batch_size).create()
    elif algo_name in ("TD3_BC", "TD3PLUSBC"):
        if hasattr(d3rlpy.algos, "TD3PlusBCConfig"):
            algo = d3rlpy.algos.TD3PlusBCConfig(learning_rate=lr, batch_size=batch_size).create()
    elif algo_name in ("BC", "BEHAVIOR_CLONING"):
        if is_discrete and hasattr(d3rlpy.algos, "DiscreteBCConfig"):
            algo = d3rlpy.algos.DiscreteBCConfig(learning_rate=lr, batch_size=batch_size).create()
        elif hasattr(d3rlpy.algos, "BCConfig"):
            algo = d3rlpy.algos.BCConfig(learning_rate=lr, batch_size=batch_size).create()

    if algo is None:
        # Fallback to discrete or continuous CQL
        if is_discrete and hasattr(d3rlpy.algos, "DiscreteCQLConfig"):
            algo = d3rlpy.algos.DiscreteCQLConfig(learning_rate=lr, batch_size=batch_size).create()
        elif hasattr(d3rlpy.algos, "CQLConfig"):
            algo = d3rlpy.algos.CQLConfig(learning_rate=lr, batch_size=batch_size).create()

    start_time = time.time()
    status = "failed"
    logger.event("stage", "Offline training started", stage="training")

    try:
        eval_env = gym.make(env_id)
        eval_freq = int(agent_cfg.get("eval_freq", 1000))

        # Training loop logging to telemetry
        step = 0
        while step < n_steps:
            steps_to_run = min(eval_freq, n_steps - step)
            if hasattr(algo, "fit") and dataset is not None:
                algo.fit(dataset, n_steps=steps_to_run, show_progress=False)
            step += steps_to_run

            # Evaluate policy in environment
            eval_rewards = []
            for _ in range(5):
                obs, _ = eval_env.reset()
                ep_r = 0.0
                done = False
                while not done:
                    act = algo.predict(np.expand_dims(obs, axis=0))[0] if hasattr(algo, "predict") else eval_env.action_space.sample()
                    obs, r, term, trunc, _ = eval_env.step(act)
                    ep_r += float(r)
                    done = term or trunc
                eval_rewards.append(ep_r)

            mean_eval_r = float(np.mean(eval_rewards))
            metrics = {
                "eval/reward": mean_eval_r,
                "train/reward": mean_eval_r,
                "step": step,
            }
            logger.log(metrics, step=step, transitions=step)

        eval_env.close()
        status = "completed"
    except BaseException as exc:
        logger.event("error", f"{type(exc).__name__}: {exc}", level="error")
        raise
    finally:
        end_time = time.time()
        # Save checkpoints
        try:
            if hasattr(algo, "save"):
                algo.save(str(ckpt_dir / "best_model.d3"))
                algo.save(str(ckpt_dir / "latest_model.d3"))
        except Exception as e:
            print(f"[Warning] Failed to save d3rlpy checkpoint: {e}")

        logger.save_runtime_metadata(
            config=OmegaConf.to_container(cfg, resolve=True),
            training_time_seconds=end_time - start_time,
        )
        for name in ("best_model.d3", "latest_model.d3"):
            p = ckpt_dir / name
            if p.exists():
                logger.artifact(p, kind="checkpoint")
        logger.finish(status=status)

    print(f"\n[d3rlpy Engine Complete] Total time: {end_time - start_time:.2f}s")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Telemetry logged to: {logger.version_dir}")


if __name__ == "__main__":
    main()
