from gymnasium.envs.registration import register

register(
    id="gym_env/PriorityScheduler-v0",
    entry_point="gym_env.envs:PrioritySchedulerEnv",
    max_episode_steps=300,
)