#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define DIM 3  // Number of dimensions for each bead

// Lennard-Jones parameters (can be adjusted in Python)
double kb = 1.0;   // Bond stiffness constant
double b = 1.0;    // Equilibrium bond length
double epsilon = 1.0;  // Depth of Lennard-Jones potential well
double sigma = 1.0;    // Distance at which the Lennard-Jones potential is zero

// Function to calculate Euclidean distance between two points
double distance(double *x, double *y) {
    double sum = 0.0;
    for (int i = 0; i < DIM; i++) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sqrt(sum);
}

// Calculate total energy (bond potential + Lennard-Jones potential)
double compute_total_energy(double *positions, int n_beads) {
    double total_energy = 0.0;
    // Bond potential: sum over adjacent beads
    for (int i = 0; i < n_beads - 1; i++) {
        double dist = distance(&positions[i * DIM], &positions[(i + 1) * DIM]);
        total_energy += kb * pow(dist - b, 2);  // Bond potential
    }
    // Lennard-Jones potential: sum over all unique pairs of beads
    for (int i = 0; i < n_beads; i++) {
        for (int j = i + 1; j < n_beads; j++) {
            double dist = distance(&positions[i * DIM], &positions[j * DIM]);
            double inv_dist = sigma / dist;
            double inv_dist_6 = pow(inv_dist, 6);
            double inv_dist_12 = inv_dist_6 * inv_dist_6;
            total_energy += 4 * epsilon * (inv_dist_12 - inv_dist_6);
        }
    }
    return total_energy;
}

// Calculate gradient of the total energy (derivative with respect to positions)
void compute_gradient(double *positions, int n_beads, double *gradient) {
    for (int i = 0; i < n_beads; i++) {
        for (int j = 0; j < DIM; j++) {
            gradient[i * DIM + j] = 0.0;  // Initialize gradient to 0
        }
    }

    // Bond potential gradients
    for (int i = 0; i < n_beads - 1; i++) {
        double dist = distance(&positions[i * DIM], &positions[(i + 1) * DIM]);
        double force = 2 * kb * (dist - b);  // Force due to bond potential
        for (int j = 0; j < DIM; j++) {
            gradient[i * DIM + j] += force * (positions[(i + 1) * DIM + j] - positions[i * DIM + j]) / dist;
            gradient[(i + 1) * DIM + j] -= force * (positions[(i + 1) * DIM + j] - positions[i * DIM + j]) / dist;
        }
    }

    // Lennard-Jones potential gradients
    for (int i = 0; i < n_beads; i++) {
        for (int j = i + 1; j < n_beads; j++) {
            double dist = distance(&positions[i * DIM], &positions[j * DIM]);
            double inv_dist = sigma / dist;
            double inv_dist_6 = pow(inv_dist, 6);
            double inv_dist_12 = inv_dist_6 * inv_dist_6;
            double force = 24 * epsilon * (2 * inv_dist_12 - inv_dist_6) / dist;
            for (int k = 0; k < DIM; k++) {
                gradient[i * DIM + k] += force * (positions[j * DIM + k] - positions[i * DIM + k]) / dist;
                gradient[j * DIM + k] -= force * (positions[j * DIM + k] - positions[i * DIM + k]) / dist;
            }
        }
    }
}

// Dot product function
double dot_product(double *a, double *b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Matrix-vector multiplication
void multiply(double *H, double *v, double *result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = 0.0;
        for (int j = 0; j < n; j++) {
            result[i] += H[i * n + j] * v[j];
        }
    }
}

// BFGS update step
void bfgs_update(double *H, double *s, double *y, int n) {
    double rho = 1.0 / dot_product(y, s, n);
    double Hy[n];
    multiply(H, y, Hy, n);

    double term1 = 1 + rho * dot_product(s, Hy, n);
    for (int i = 0; i < n; i++) {
        H[i] = Hy[i] * term1;
    }
}

// External callback function
typedef void (*callback_t)(double *positions);

// Main BFGS optimization loop
void bfgs_optimizer(double *positions, int n_beads, int maxiter, double tol, double *H, callback_t callback) {
    int n = n_beads * DIM;
    double gradient[n];
    double delta_x[n];
    double delta_g[n];

    for (int iter = 0; iter < maxiter; iter++) {
        // Compute energy and gradient
        compute_gradient(positions, n_beads, gradient);
        double energy = compute_total_energy(positions, n_beads);
        
        // Convergence check
        double grad_norm = 0.0;
        for (int i = 0; i < n; i++) {
            grad_norm += gradient[i] * gradient[i];
        }
        grad_norm = sqrt(grad_norm);
        if (grad_norm < tol) {
            break;
        }

        // Update step: delta_x = -H * gradient
        multiply(H, gradient, delta_x, n);
        
        // Perform line search (simplified for now)
        double step_size = 0.01;  // Simple fixed step size
        for (int i = 0; i < n; i++) {
            positions[i] -= step_size * delta_x[i];
        }

        // Compute new gradient
        compute_gradient(positions, n_beads, delta_g);
        
        // Compute s and y vectors
        double s[n], y[n];
        for (int i = 0; i < n; i++) {
            s[i] = -step_size * delta_x[i];
            y[i] = delta_g[i] - gradient[i];
        }

        // Update the inverse Hessian approximation (BFGS formula)
        bfgs_update(H, s, y, n);

        // Call the Python callback function with the current positions
        callback(positions);  // This will call the Python callback to track the trajectory
    }
}
