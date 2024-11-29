import gym
import time

env = gym.make("CartPole-v1", render_mode="human")

num_episodes = 5

for episode in range(num_episodes):
    print(f"Inicio del Episodio {episode + 1}")
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = env.action_space.sample()  

        next_state, reward, done, truncated, info = env.step(action)

        total_reward += reward

        time.sleep(0.02)
    
    print(f"Recompensa total en el Episodio {episode + 1}: {total_reward}")

env.close()
