import gym

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

def create_cartpole_env():
    return gym.make("CartPole-v1", render_mode="human")

def run_episode(env):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
    
    return total_reward