import numpy as np
import random
import time

def q_learning_cartpole(
    env,
    num_episodes=1000,
    gamma=0.99,  # Factor de descuento
    alpha=0.1,   # Factor de aprendizaje
    epsilon=1.0, # Factor de exploración
    epsilon_decay=0.995,
    epsilon_min=0.01
):
    """Implementación de Q-learning para CartPole."""

    # Discretización de los espacios continuos
    state_bins = [np.linspace(-2.4, 2.4, 20),  # Posición del carro
                  np.linspace(-3.0, 3.0, 20),  # Velocidad del carro
                  np.linspace(-0.2, 0.2, 20),  # Ángulo del péndulo
                  np.linspace(-3.0, 3.0, 20)]  # Velocidad angular del péndulo
    
    num_actions = env.action_space.n
    q_table = np.zeros([len(bins) + 1 for bins in state_bins] + [num_actions])

    def discretize_state(state):
        """Mapea un estado continuo a uno discreto."""
        indices = [min(np.digitize(val, state_bins[i]), len(state_bins[i])-1) 
                   for i, val in enumerate(state)]
        return tuple(indices)
    
    rewards = []
    epsilon = 1.0

    for episode in range(num_episodes):
        state, _ = env.reset()  # Obtener el estado inicial
        state = discretize_state(state)
        total_reward = 0

        for _ in range(500):  # Máximo número de pasos por episodio
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Exploración
            else:
                action = np.argmax(q_table[state])  # Explotación

            # Realizar la acción
            next_state_continuous, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state_continuous)

            # Ajuste de la recompensa
            if done:
                reward = -200
            else:
                reward -= abs(next_state_continuous[0]) * 0.1  # Penalización por desviarse del centro
            
            # Actualización de la Q-table (Q-learning)
            td_target = reward + gamma * np.max(q_table[next_state])
            q_table[state][action] += alpha * (td_target - q_table[state][action])
            
            state = next_state
            total_reward += reward

            if done:
                break

            env.render()
            time.sleep(0.01)

        # Decaimiento de epsilon
        # epsilon = max(epsilon * epsilon_decay, epsilon_min)
        epsilon = 0.01
        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    
    return q_table, rewards
