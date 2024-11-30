from cart_pole import create_cartpole_env
from definicion import q_learning_cartpole

def main():
    num_episodes = 1000
    env = create_cartpole_env()

    print("Iniciando el entrenamiento con Q-learning...")
    q_table, rewards = q_learning_cartpole(env, num_episodes=num_episodes)

    print(f"Entrenamiento completado. El agente ha entrenado durante {num_episodes} episodios.")
    
    # Imprimir recompensas de los primeros episodios
    print("Recompensas de los primeros episodios:")
    for episode in range(min(10, num_episodes)):
        print(f"Episodio {episode + 1}: Recompensa total = {rewards[episode]}")

    # Cerrar el entorno
    env.close()

if __name__ == "__main__":
    main()
