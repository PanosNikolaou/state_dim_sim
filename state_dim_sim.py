import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
import json
import pandas as pd


class StateDimensionSimulator:
    def __init__(self, timesteps=100, space_points=50, alpha=0.1, beta=0.2, gamma=0.1, delta=0.05):
        self.timesteps = timesteps
        self.space_points = space_points
        self.alpha = alpha  # Interference strength
        self.beta = beta  # Non-linearity strength
        self.gamma = gamma  # Noise strength
        self.delta = delta  # Chaos parameter
        self.states = np.zeros((timesteps, space_points))
        self.states[0, space_points // 2] = 1  # Initial wave disturbance

    def evolve_state(self):
        for t in range(1, self.timesteps):
            for x in range(1, self.space_points - 1):
                prev_state = self.states[t - 1, x]

                # Prevent values from exploding
                prev_state = np.clip(prev_state, -1e3, 1e3)  # Limit range to avoid overflows

                # Compute interference
                interference = np.sin(2 * np.pi * prev_state) + np.cos(2 * np.pi * prev_state)

                # Apply noise with safe handling
                noise = np.random.normal(0, self.gamma)

                # Non-linearity with overflow protection
                non_linear_effect = self.beta * prev_state ** 3 - (self.beta / 2) * prev_state ** 2
                non_linear_effect = np.clip(non_linear_effect, -1e3, 1e3)  # Limit range

                # Chaotic term
                chaotic_term = self.delta * np.sin(prev_state * np.pi)

                # Compute next state safely
                self.states[t, x] = np.nan_to_num(0.5 * (self.states[t - 1, x - 1] + self.states[t - 1, x + 1])
                                                  + self.alpha * interference + noise + non_linear_effect + chaotic_term,
                                                  nan=0.0, posinf=1e3, neginf=-1e3)

        return self.states

    def plot_heatmap(self):
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.states, cmap="viridis", cbar=True)
        plt.xlabel("Spatial Position")
        plt.ylabel("State Snapshot (S)")
        plt.title("Heatmap of State Evolution")
        plt.draw()  # Non-blocking alternative
        plt.pause(0.01)  # Allows Tkinter to update

    def plot_3d(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        X, T = np.meshgrid(np.arange(self.space_points), np.arange(self.timesteps))
        ax.plot_surface(X, T, self.states, cmap="plasma")
        ax.set_xlabel("Spatial Position")
        ax.set_ylabel("State Index (S)")
        ax.set_zlabel("State Value")
        ax.set_title("3D Visualization of State Evolution")
        plt.draw()  # Non-blocking alternative
        plt.pause(0.01)  # Allows Tkinter to update

    def compute_lyapunov_exponent(self):
        """Estimate Lyapunov exponent to measure chaos in the system."""
        diffs = np.abs(np.diff(self.states, axis=0))
        avg_growth = np.mean(np.log(1 + diffs + 1e-10))  # Avoid log(0) issues
        return avg_growth

    def save_results(self, filename="simulation_results.json"):
        """Save simulation results to a JSON file."""
        results = {
            "parameters": {
                "timesteps": self.timesteps,
                "space_points": self.space_points,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "delta": self.delta,
            },
            "states": self.states.tolist()
        }
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {filename}")

    def save_to_csv(self, filename="simulation_results.csv"):
        """Save simulation results to a CSV file."""
        df = pd.DataFrame(self.states)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def load_results(self, filename="simulation_results.json"):
        """Load simulation results from a JSON file."""
        with open(filename, "r") as f:
            results = json.load(f)
        self.timesteps = results["parameters"]["timesteps"]
        self.space_points = results["parameters"]["space_points"]
        self.alpha = results["parameters"]["alpha"]
        self.beta = results["parameters"]["beta"]
        self.gamma = results["parameters"]["gamma"]
        self.delta = results["parameters"]["delta"]
        self.states = np.array(results["states"])
        print(f"Results loaded from {filename}")


# GUI for interactive parameter tuning
def run_gui():
    def run_simulation():
        sim = StateDimensionSimulator(
            timesteps=int(timesteps_var.get()),
            space_points=int(space_points_var.get()),
            alpha=float(alpha_var.get()),
            beta=float(beta_var.get()),
            gamma=float(gamma_var.get()),
            delta=float(delta_var.get())
        )
        sim.evolve_state()
        sim.plot_heatmap()
        sim.plot_3d()
        lyapunov_exp = sim.compute_lyapunov_exponent()
        print(f"Estimated Lyapunov Exponent: {lyapunov_exp}")
        sim.save_results()
        sim.save_to_csv()

    root = tk.Tk()
    root.title("State Dimension Simulator")

    ttk.Label(root, text="Timesteps").grid(row=0, column=0)
    timesteps_var = tk.StringVar(value="100")
    ttk.Entry(root, textvariable=timesteps_var).grid(row=0, column=1)

    ttk.Label(root, text="Space Points").grid(row=1, column=0)
    space_points_var = tk.StringVar(value="50")
    ttk.Entry(root, textvariable=space_points_var).grid(row=1, column=1)

    ttk.Label(root, text="Alpha (Interference Strength)").grid(row=2, column=0)
    alpha_var = tk.StringVar(value="0.1")
    ttk.Entry(root, textvariable=alpha_var).grid(row=2, column=1)

    ttk.Label(root, text="Beta (Non-Linearity Strength)").grid(row=3, column=0)
    beta_var = tk.StringVar(value="0.2")
    ttk.Entry(root, textvariable=beta_var).grid(row=3, column=1)

    ttk.Label(root, text="Gamma (Noise Strength)").grid(row=4, column=0)
    gamma_var = tk.StringVar(value="0.1")
    ttk.Entry(root, textvariable=gamma_var).grid(row=4, column=1)

    ttk.Label(root, text="Delta (Chaos Parameter)").grid(row=5, column=0)
    delta_var = tk.StringVar(value="0.05")
    ttk.Entry(root, textvariable=delta_var).grid(row=5, column=1)

    ttk.Button(root, text="Run Simulation", command=run_simulation).grid(row=6, columnspan=2)

    root.mainloop()


# Example Usage
if __name__ == "__main__":
    run_gui()
