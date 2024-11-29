import numpy as np

def q_learning_cartpole(
    env,
    num_episodes=1000, 
    gamma=0.99, 
    alpha=0.1, 
    epsilon=1.0, 
    epsilon_decay=0.995, 
    epsilon_min=0.01
):
    state_bins = [np.linspace(-2.4, 2.4, 20),  # Posición del carro
                  np.linspace(-3.0, 3.0, 20),  # Velocidad del carro
                  np.linspace(-0.2, 0.2, 20),  # Ángulo del péndulo
                  np.linspace(-3.0, 3.0, 20)]  # Velocidad angular del péndulo
    
    # Inicializando tabla Q
    num_actions = env.action_space.n
    q_table = np.zeros([len(bins) + 1 for bins in state_bins] + [num_actions])
    
    def discretize_state(state):
        """Mapea un estado continuo a uno discreto."""
        indices = []
        for i, val in enumerate(state):
            index = np.digitize(val, state_bins[i])  # Encuentra el índice del bin
            indices.append(min(index, len(state_bins[i])))  # Asegura que no se salga del rango
        return tuple(indices)
    
    # Seguimiento de recompensas por episodio
    rewards = []

    for episode in range(num_episodes):
        state = discretize_state(env.reset())  # Inicializa el estado y discretiza
        total_reward = 0
        
        for _ in range(500):  # Máximo número de pasos por episodio
            # Decisión de acción (exploración vs explotación)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Exploración
            else:
                action = np.argmax(q_table[state])  # Explotación
            
            # Ejecuta la acción y obtiene el siguiente estado, recompensa, y si terminó
            next_state_continuous, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state_continuous)

            # Modifica la recompensa para fomentar estabilidad
            if done:
                reward = -200  # Penalización por caída
            else:
                reward -= abs(next_state_continuous[0]) * 0.1  # Penalización por moverse demasiado
            
            # Actualizandode la tabla Q (Regla Q-Learning) FORMULA
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            q_table[state][action] += alpha * (td_target - q_table[state][action])
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Decaimiento de epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)
    
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    return q_table, rewards
