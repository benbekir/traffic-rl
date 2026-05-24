from __future__ import annotations
from typing import Any
from stable_baselines3.common.base_class import BaseAlgorithm
from envs.factory import DEFAULT_DELTA_TIME, DEFAULT_NUM_SECONDS, make_env

def train_model(
    model_cls: type[BaseAlgorithm],
    *,
    checkpoint_path: str,
    total_training_episodes: int,
    model_kwargs: dict[str, Any],
) -> None:
    env = make_env()

    try:
        steps_per_episode = DEFAULT_NUM_SECONDS // DEFAULT_DELTA_TIME
        total_training_steps = steps_per_episode * env.num_envs * total_training_episodes

        resolved_kwargs = dict(model_kwargs)
        model = model_cls(
            "MlpPolicy",
            env,
            verbose=1,
            **resolved_kwargs,
        )

        model.learn(total_timesteps=total_training_steps, progress_bar=True)
        model.save(checkpoint_path)
    finally:
        env.close()

def run_loaded_model(
    model_cls: type[BaseAlgorithm],
    *,
    checkpoint_path: str,
    recording_name: str,
) -> None:
    env = make_env(recording_name=recording_name)
    env.allow_reset = False

    try:
        model = model_cls.load(checkpoint_path, env=env)
        obs = env.reset()

        while True:
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(actions)

            if any(dones):
                break
    except Exception as exc:
        print(f"An error occurred during evaluation execution: {exc}")
    finally:
        env.close()