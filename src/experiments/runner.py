from __future__ import annotations
from typing import Any
from stable_baselines3.common.base_class import BaseAlgorithm
from envs.factory import DEFAULT_DELTA_TIME, DEFAULT_NUM_SECONDS, make_env

def train_model(
    model_cls: type[BaseAlgorithm],
    *,
    checkpoint_path: str,
    curriculum_stages: list[float] = [0.75, 0.875, 1.0], 
    episodes_per_curriculum: int = 25,
    model_kwargs: dict[str, Any]
) -> None:
    model = None

    for i, scale in enumerate(curriculum_stages):
        print(f"Starting curriculum stage {i+1}/{len(curriculum_stages)}: (Traffic Scale: {scale})")
        env = make_env(scale=scale)
        
        try:
            steps_per_episode = DEFAULT_NUM_SECONDS // DEFAULT_DELTA_TIME
            total_training_steps = steps_per_episode * env.num_envs * episodes_per_curriculum
            
            if model is None:
                resolved_kwargs = dict(model_kwargs)
                model = model_cls(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    **resolved_kwargs,
                )
            else:
                model.set_env(env)
            
            model.learn(total_timesteps=total_training_steps, progress_bar=True, reset_num_timesteps=False)
            stage_checkpoint = f"{checkpoint_path}_stage_{int(scale*100)}"
            model.save(stage_checkpoint)

        finally:
            env.close()

    if model is not None:
        model.save(checkpoint_path)


def run_loaded_model(
    model_cls: type[BaseAlgorithm],
    *,
    checkpoint_path: str,
    recording_name: str,
) -> None:
    env = make_env(scale=1.0, recording_name=recording_name)
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