import numpy as np
import ctypes
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initialize protein positions
def initialize_protein(n_beads, dimension=3, fudge = 1e-5):
    """
    Initialize a protein with `n_beads` arranged almost linearly in `dimension`-dimensional space.
    The `fudge` is a factor that, if non-zero, adds a spiral structure to the configuration.
    """
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)  # Fixed bond length of 1 unit
        positions[i, 2] = fudge * np.sin(i*i)  # Fixed bond length of 1 unit                
    return positions

# Lennard-Jones potential function
def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """
    Compute Lennard-Jones potential between two beads.
    """
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# Bond potential function
def bond_potential(r, b=1.0, k_b=100.0):
    """
    Compute harmonic bond potential between two bonded beads.
    """
    return k_b * (r - b)**2

# Total energy function
def total_energy(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    Compute the total energy of the protein conformation.
    """
    positions = positions.reshape((n_beads, -1))
    energy = 0.0

    # Bond energy
    for i in range(n_beads - 1):
        r = np.linalg.norm(positions[i+1] - positions[i])
        energy += bond_potential(r, b, k_b)

    # Lennard-Jones potential for non-bonded interactions
    for i in range(n_beads):
        for j in range(i+1, n_beads):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-2:  # Avoid division by zero
                energy += lennard_jones_potential(r, epsilon, sigma)

    return energy

energy_bfgs = ctypes.CDLL('./energy_bfgs.so')

# Define argument and return types for C functions
energy_bfgs.compute_total_energy.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
]
energy_bfgs.compute_total_energy.restype = ctypes.c_double

energy_bfgs.compute_gradient.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)
]
energy_bfgs.compute_gradient.restype = None

energy_bfgs.bfgs_optimizer.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)
]
energy_bfgs.bfgs_optimizer.restype = None

def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6, kb=1.0, b=1.0, epsilon=1.0, sigma=1.0):
    trajectory = []

    class OptimizeResult:
        def __init__(self, x):
            self.x = x  # Store the optimized positions as the 'x' attribute

    def callback(positions_ptr):
    # Convert ctypes pointer to a NumPy array
        positions = np.ctypeslib.as_array(positions_ptr, shape=(n_beads, 3))
    
    # Append the reshaped array to the trajectory
        if positions.shape != (n_beads, 3):
           raise ValueError(f"Unexpected shape: {positions.shape}, expected ({n_beads}, 3)")

    # Append the reshaped array to the trajectory
       trajectory.append(positions.copy())  
    
       if len(trajectory) % 20 == 0:
          print(f"Step {len(trajectory)}")

    callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_double))
    callback_func = callback_type(callback)

    # Flatten the initial positions for optimization
    positions_flat = positions.flatten()

    # Allocate space for inverse Hessian approximation
    H = np.eye(n_beads * 3)  # Identity matrix as initial Hessian approximation

    # Call custom BFGS optimizer in C
    energy_bfgs.bfgs_optimizer(
        positions_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_beads,
        maxiter,
        tol,
        H.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        callback_func
    )

    optimized_positions = positions_flat.reshape((n_beads, 3))

    result = OptimizeResult(optimized_positions)

    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f'Writing data to file {csv_filepath}')
        np.savetxt(csv_filepath, trajectory[-1], delimiter=",")
    
    return result, trajectory

# 3D visualization function
def plot_protein_3d(positions, title="Protein Conformation", ax=None):
    """
    Plot the 3D positions of the protein.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    positions = positions.reshape((-1, 3))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=6)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# Animation function
# Animation function with autoscaling
def animate_optimization(trajectory, interval=100):
    """
    Animate the protein folding process in 3D with autoscaling.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], '-o', markersize=6)

    def update(frame):
        positions = trajectory[frame]
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])

        # Autoscale the axes
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_zlim(z_min - 1, z_max + 1)

        ax.set_title(f"Step {frame + 1}/{len(trajectory)}")
        return line,

    ani = FuncAnimation(
        fig, update, frames=len(trajectory), interval=interval, blit=False
    )
    plt.show()

# Main function
if __name__ == "__main__":
    n_beads = 10
    dimension = 3
    initial_positions = initialize_protein(n_beads, dimension)

    print("Initial Energy:", total_energy(initial_positions.flatten(), n_beads))
    plot_protein_3d(initial_positions, title="Initial Configuration")

    result, trajectory = optimize_protein(initial_positions, n_beads, write_csv = True)

    optimized_positions = result.x.reshape((n_beads, dimension))
    print("Optimized Energy:", total_energy(optimized_positions.flatten(), n_beads))
    plot_protein_3d(optimized_positions, title="Optimized Configuration")

    # Animate the optimization process
    animate_optimization(trajectory)
