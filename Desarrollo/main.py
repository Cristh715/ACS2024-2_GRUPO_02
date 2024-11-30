import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cart_pole import create_cartpole_env
from definicion import q_learning_cartpole

class CartPoleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulación CartPole - Q-learning")

        # Entrada para definir el número de episodios
        self.episode_count_var = tk.StringVar(value="1000")
        ttk.Label(root, text="Número de Episodios:").pack(padx=5, pady=5)
        self.episode_entry = ttk.Entry(root, textvariable=self.episode_count_var)
        self.episode_entry.pack(padx=5, pady=5)

        # Botón para ejecutar el entrenamiento
        self.run_button = ttk.Button(root, text="Ejecutar Entrenamiento", command=self.run_training)
        self.run_button.pack(padx=10, pady=10)

        # Tabla de resultados
        self.create_table()

        # Contenedor del gráfico
        self.canvas_frame = ttk.Frame(root)
        self.canvas_frame.pack(padx=10, pady=10)

        # Contenedor de valores en tiempo real
        self.create_real_time_panel()

    def create_table(self):
        self.table_frame = ttk.Frame(self.root)
        self.table_frame.pack(padx=10, pady=10)

        columns = ("Episodio", "Recompensa", "Posición (x)", "Vel. Carro", "Ángulo (θ)", "Vel. Angular")
        self.episode_table = ttk.Treeview(self.table_frame, columns=columns, show="headings", height=10)

        for col in columns:
            self.episode_table.heading(col, text=col)
            self.episode_table.column(col, minwidth=50, width=100)

        self.episode_table.pack(side=tk.LEFT)
        scroll_bar = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.episode_table.yview)
        self.episode_table.configure(yscrollcommand=scroll_bar.set)
        scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_real_time_panel(self):
        self.real_time_frame = ttk.LabelFrame(self.root, text="Valores en Tiempo Real")
        self.real_time_frame.pack(padx=10, pady=5, fill=tk.X)

        self.labels = {}
        for label in ["Posición (x)", "Vel. Carro", "Ángulo (θ)", "Vel. Angular"]:
            frame = ttk.Frame(self.real_time_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            self.labels[label] = ttk.Label(frame, text="0.0")
            self.labels[label].pack(side=tk.RIGHT)

    def update_real_time_values(self, state):
        position, velocity, angle, angular_velocity = state
        self.labels["Posición (x)"].config(text=f"{position:.2f}")
        self.labels["Vel. Carro"].config(text=f"{velocity:.2f}")
        self.labels["Ángulo (θ)"].config(text=f"{np.degrees(angle):.2f}°")
        self.labels["Vel. Angular"].config(text=f"{angular_velocity:.2f}")

    def plot_rewards(self, rewards):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rewards, label="Recompensas")
        ax.set_title("Recompensas por Episodio")
        ax.set_xlabel("Episodio")
        ax.set_ylabel("Recompensa")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def run_training(self):
        self.episode_table.delete(*self.episode_table.get_children())

        num_episodes = int(self.episode_count_var.get())
        env = create_cartpole_env()
        q_table, rewards = q_learning_cartpole(env, num_episodes=num_episodes)

        for i, reward in enumerate(rewards):
            state, _ = env.reset()
            self.episode_table.insert(
                "", "end", values=(i + 1, reward, state[0], state[1], state[2], state[3])
            )
            self.update_real_time_values(state)

        self.plot_rewards(rewards)
        env.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = CartPoleApp(root)
    root.mainloop()
