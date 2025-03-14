
# **State Evolution in Extended (4+1)D Spacetime**

## **Overview**
This project explores an **extended (4+1)D spacetime model**, incorporating a **state evolution dimension** to analyze quantum and classical system dynamics. The simulation implements **non-linear interference, noise, and chaos terms** to study how quantum states evolve beyond standard (3+1)D spacetime.

Using **heatmaps, 3D surface plots, and Lyapunov exponent analysis**, this framework numerically investigates **state-space curvature, quantum chaos, and emergent phenomena** in an extended dimensional setting.

---

## **Features**
- **State Evolution in (4+1)D Spacetime**: Extends conventional 3+1D physics by introducing an additional **state evolution parameter**.
- **Non-Linear Quantum Dynamics**: Modifies the Schrödinger equation with **non-linearity, probabilistic fluctuations, and chaos terms**.
- **Numerical Simulations**: Uses **finite difference methods** to evolve states and visualize them using **heatmaps & 3D plots**.
- **Chaos Quantification**: Estimates the **Lyapunov exponent** to assess chaotic behavior in quantum state-space.
- **Graphical Interface (GUI)**: Allows interactive parameter tuning and real-time visualization.
- **Data Storage & Analysis**: Saves results in **JSON & CSV formats** for further exploration.

---

## **Mathematical Model**
### **1️⃣ Extended Spacetime and State Evolution**
- Traditional physics models operate in **(3+1)D spacetime**. This project introduces an additional **state evolution dimension** to explore complex dynamics.
- Inspired by **brane-world models** like the **Randall-Sundrum (RS) framework**, where our universe is viewed as a **3D brane in higher-dimensional space**, this approach integrates state-dependent modifications to system evolution.

### **2️⃣ Non-Linear Quantum Dynamics in State-Space**
- The **Schrödinger equation** is extended to include **non-linear terms** and **probabilistic fluctuations**, allowing for state-dependent modifications.
- **State-Space Curvature**: Analogous to spacetime curvature in general relativity, this represents how a quantum state evolves within a higher-dimensional framework.
- **Chaos and Emergent Phenomena**: The introduction of **chaotic perturbations** and non-linearity leads to rich, emergent dynamics.

### **3️⃣ Numerical Simulations and Visualization**
- **Finite Difference Approximation**: The system state evolves using numerical methods that consider **local interactions, interference, and chaos effects**.
- **Heatmaps & 3D Plots**: Visualization of wave function evolution under non-linear dynamics.
- **Lyapunov Exponent Calculation**: A measure of chaotic divergence in the system.

### **4️⃣ Implications for Physics & Applications**
- **Quantum Gravity & Black Holes**: Exploring **quantum gravity, Hawking radiation, and singularities** using extended state-space models.
- **Quantum Computing**: Investigating how **non-linear quantum logic gates** might behave in extended quantum mechanics.
- **Cosmology & Dark Energy**: Studying potential links between **state-space evolution and early universe dynamics**.

---

## **Installation & Usage**
### **Requirements**
Ensure you have the following dependencies installed:
```bash
pip install numpy matplotlib seaborn pandas tkinter
```

### **Running the Simulation**
To launch the GUI for interactive parameter tuning and visualization, run:
```bash
python main.py
```

Alternatively, to execute a simulation directly in Python:
```python
from state_simulator import StateDimensionSimulator
sim = StateDimensionSimulator(timesteps=100, space_points=50, alpha=0.1, beta=0.2, gamma=0.1, delta=0.05)
sim.evolve_state()
sim.plot_heatmap()
sim.plot_3d()
print("Lyapunov Exponent:", sim.compute_lyapunov_exponent())
```

---

## **Project Structure**
```
📂 state_evolution_project
│── main.py               # GUI and main execution script
│── state_simulator.py     # Core simulation class
│── README.md             # Project documentation
│── requirements.txt      # Required dependencies
│── results/              # Folder for saved JSON and CSV results
```

---

## **Future Enhancements**
- **Higher-dimensional simulations** with additional parameters.
- **Integration with quantum computing frameworks**.
- **Parallelized computations for large-scale state evolution studies**.

---

## **Contributing**
Contributions are welcome! Feel free to submit pull requests or open issues for improvements.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for details.

---

## **Acknowledgments**
Special thanks to theoretical physics concepts from **quantum mechanics, chaos theory, and brane-world models**, which inspired this project.

