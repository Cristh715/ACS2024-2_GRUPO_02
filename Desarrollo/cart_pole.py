import gym


def create_cartpole_env():
    return gym.make("CartPole-v1", render_mode="human")


def run_episode(env):
    done = False
    total_reward = 0
    env.reset()

    while not done:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

    return total_reward
