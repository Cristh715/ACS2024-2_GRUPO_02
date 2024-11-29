"""
Acciones posibles:
    0: Mover el carro a la izquierda.
    1: Mover el carro a la derecha.
Espacio de estados:
    Posición del carro: Ubicación del carro en el eje x. (-4.8, 4.8)
    Velocidad del carro: Velocidad del carro en la dirección x. (-Inf, Inf)
    Ángulo del péndulo: Ángulo del péndulo respecto a la vertical. (-0.418 rad (-24°), 0.418 rad (24°))
    Velocidad angular del péndulo: Velocidad angular con la que el péndulo está girando. (-Inf, Inf)
Recompensas:
    El agente recibe +1 por cada paso que mantenga el péndulo de pie.
    El episodio termina cuando el péndulo se cae o el carro se mueve fuera de los límites establecidos.
Episodio:
    El episodio termina si se alcanzan más de 500 pasos
"""

from cart_pole import create_cartpole_env, run_episode

num_episodes = 5

x = 0.0          # Posición del carrito
v_cart = 0.0     # Velocidad del carrito
ang_pend = 0.0   # Ángulo del péndulo
v_ang = 0.0      # Velocidad angular del péndulo

def main():
    """
    Función principal que ejecuta los episodios de CartPole.
    """
    num_episodes = 5
    env = create_cartpole_env()

    for episode in range(num_episodes):
        print(f"Inicio del episodio {episode + 1}")
        total_reward = run_episode(env) 
        print(f"Recompensa total en el episodio {episode + 1}: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()