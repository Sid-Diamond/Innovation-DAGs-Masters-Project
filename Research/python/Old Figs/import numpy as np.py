import numpy as np
import matplotlib.pyplot as plt

# Define the Heaviside step function
def H(t, N=0):
    """
    Heaviside step function
    H(t) = 0 for t < N
    H(t) = 1 for t >= N
    """
    return np.where(t >= N, 1, 0)

# Create time array
t = np.linspace(-5, 5, 1000)

# Define N (threshold value)
N = 5  # You can change this value

# Calculate functions
H_t = H(t, N)
H_t_squared = H_t * H_t  # Product of H(t) with itself

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(t, H_t, 'b-', linewidth=2, label=f'H(t), N={N}')
plt.plot(t, H_t_squared, 'r--', linewidth=2, label=f'H(t) × H(t)')

# Add styling
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=N, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f't={N}')
plt.grid(True, alpha=0.3)
plt.xlabel('t', fontsize=12)
plt.ylabel('H(t)', fontsize=12)
plt.title('Heaviside Step Function and Its Product', fontsize=14)
plt.legend(fontsize=10)
plt.ylim(-0.2, 1.3)

plt.tight_layout()
plt.show()