# **Péndulo Invertido con Q-Learning**

## **Descripción**
Este proyecto implementa un agente basado en **Q-Learning** para estabilizar un péndulo invertido utilizando el entorno `CartPole-v1` de Gym. El agente aprende a equilibrar el péndulo sobre un carrito mientras minimiza el desplazamiento.

### **Características principales**
- Implementación de **Q-Learning** con discretización del espacio de estados.
- Visualización en tiempo real de la simulación después de los primeros 1000 episodios.
- Gráficos para analizar el desempeño del agente (recompensa acumulada y tendencia).

---

## **Requerimientos Técnicos**

### **Versiones necesarias**
- **Python** >= 3.7

### **Librerías necesarias**
Para ejecutar este proyecto, instala las siguientes librerías:

- `numpy`
- `keyboard`
- `gym`
- `scikit-learn`
- `matplotlib`

Puedes instalarlas con el siguiente comando:

```bash
pip install numpy keyboard gym scikit-learn matplotlib

### Si se desea parar el proceso se puede interrumpir el proceso kernel lo cual se mostraría esta imagen
![image](https://github.com/user-attachments/assets/3cc31830-e31f-4010-ad3a-ac3e40ae8860)
### 4 Después de mil episodios empezará a mostrarse la interfaz del pendulo en una ventana .py , esto con el fin de que se pueda entrenar más rápido
### 5 Después de detener el proceso se puede ejecutar las celdas de gráficas.
