# main.py
from cart_pole import create_cartpole_env
from definicion import q_learning_cartpole
import numpy as np
import time

def main():
    num_episodes = 5000
    env = create_cartpole_env()
    render_threshold = 450  # Umbral de recompensa para activar la visualización
    reward_history = []  # Lista para almacenar las últimas recompensas
    history_length = 100  # Número de recompensas a considerar

    print("Iniciando el entrenamiento con Q-learning...")
    q_table, rewards = q_learning_cartpole(env, num_episodes=num_episodes)

    print(f"Entrenamiento completado. El agente ha entrenado durante {num_episodes} episodios.")

    # Evaluar el rendimiento y mostrar la visualización
    for episode in range(num_episodes):
        reward_history.append(rewards[episode])
        if len(reward_history) > history_length:
            reward_history.pop(0)  # Eliminar la recompensa más antigua

        avg_reward = np.mean(reward_history)
        print(f"Episodio {episode + 1}: Recompensa total = {rewards[episode]}, Recompensa promedio = {avg_reward}")

        if avg_reward >= render_threshold:
            print("El agente ha aprendido. Mostrando la visualización...")
            state, _ = env.reset()
            done = False
            while not done:
                action = np.argmax(q_table[q_learning_cartpole.discretize_state(state)])  # Obtener la mejor acción de la Q-table
                state, _, done, _, _ = env.step(action)
                env.render()
                time.sleep(0.02)  # Ajustar la velocidad de la visualización
            break

    env.close()

if __name__ == "__main__":
    main()