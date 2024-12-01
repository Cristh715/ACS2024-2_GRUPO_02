from cart_pole import create_cartpole_env
from definicion import q_learning_cartpole
import numpy as np
import time


def main():
    num_episodes = 1000  # Reducido para mejorar tiempos de prueba
    env = create_cartpole_env()
    render_threshold = 400  # Ajustado para visualización más temprana
    reward_history = []
    history_length = 100  # Promedio de recompensas

    print("Iniciando el entrenamiento con Q-learning...")
    q_table, rewards = q_learning_cartpole(env, num_episodes=num_episodes)

    print(f"Entrenamiento completado. El agente ha entrenado durante {num_episodes} episodios.")

    # Evaluar el rendimiento y mostrar la visualización
    for episode in range(num_episodes):
        reward_history.append(rewards[episode])
        if len(reward_history) > history_length:
            reward_history.pop(0)

        avg_reward = np.mean(reward_history)
        print(f"Episodio {episode + 1}: Recompensa total = {rewards[episode]}, Recompensa promedio = {avg_reward}")

        if avg_reward >= render_threshold:
            print("El agente ha aprendido. Mostrando la visualización...")
            state, _ = env.reset()
            done = False
            while not done:
                action = np.argmax(q_table[q_learning_cartpole.discretize_state(state)])
                state, _, done, _, _ = env.step(action)
                env.render()
                time.sleep(0.02)
            break

    env.close()


if __name__ == "__main__":
    main()
