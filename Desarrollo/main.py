from cart_pole import create_cartpole_env, run_episode

num_episodes = 5

x = 0.0          # Posición del carrito
v_cart = 0.0     # Velocidad del carrito
ang_pend = 0.0   # Ángulo del péndulo
v_ang = 0.0      # Velocidad angular del péndulo

def main(): #Funcion principal
    num_episodes = 5
    env = create_cartpole_env()

    for episode in range(num_episodes):
        print(f"Inicio del episodio {episode + 1}")
        total_reward = run_episode(env) 
        print(f"Recompensa total en el episodio {episode + 1}: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()