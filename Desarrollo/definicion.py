# definicion.py
import numpy as np
import random
import time

def q_learning_cartpole(
    env,
    num_episodes=1000,
    gamma=0.99,  # Factor de descuento
    alpha_inicial=0.1,   # Tasa de aprendizaje inicial
    alpha_decay=0.999, # Decaimiento de alpha
    alpha_min=0.001, # Valor mínimo de alpha
    epsilon=1.0,  # Factor de exploración
    epsilon_decay=0.995,
    epsilon_min=0.01
):
    """Implementación de Q-learning para CartPole."""

    # Discretización de los espacios continuos (mayor número de bins)
    state_bins = [np.linspace(-4.8, 4.8, 40),  # Posición del carro
                  np.linspace(-4.0, 4.0, 40),  # Velocidad del carro
                  np.linspace(-0.418, 0.418, 40),  # Ángulo del péndulo
                  np.linspace(-4.0, 4.0, 40)]  # Velocidad angular del péndulo

    num_actions = env.action_space.n
    q_table = np.zeros([len(bins) + 1 for bins in state_bins] + [num_actions])

    def discretize_state(state):  # Definir discretize_state dentro de q_learning_cartpole
        """Mapea un estado continuo a uno discreto."""
        indices = [min(np.digitize(val, state_bins[i]), len(state_bins[i])-1) 
                   for i, val in enumerate(state)]
        return tuple(indices)

    rewards = []
    epsilon = 1.0
    alpha = alpha_inicial

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
                reward = 1  # Recompensa por mantenerse en pie
                reward -= abs(next_state_continuous[0]) * 0.1  # Penalización por desviación

            # Actualización de la Q-table (Q-learning)
            td_target = reward + gamma * np.max(q_table[next_state])
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state
            total_reward += reward

            if done:
                break

            # env.render()  # No mostrar la visualización durante el entrenamiento
            # time.sleep(0.01)

        # Decaimiento de alpha
        alpha = max(alpha * alpha_decay, alpha_min)
        # Decaimiento de epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    return q_table, rewards