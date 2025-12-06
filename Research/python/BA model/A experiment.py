import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def calculate_A(k, b0, alpha, gamma_param):
    """
    Calculate A(k) = b0 * product * Γ(k+α+γ)/Γ(k+α)
    where product = (α/(α+γ)) * ((α+1)/(α+γ+1)) * ... * ((α+k-1)/(α+γ+k-1))
    """
    # Calculate the product
    product = 1
    for i in range(k):
        product *= (alpha + i) / (alpha + gamma_param + i)
    
    # Calculate gamma function ratio
    gamma_ratio = gamma(k + alpha + gamma_param) / gamma(k + alpha)
    
    # Final result
    return b0 * product * gamma_ratio

# Network parameters (from your model)
m_edges = 30
h = 0.8
f_a = 0.6
f_b = 1 - f_a
mu_a = 2
mu_b = 1

# Compute lambda values
lambda_a = h * f_a + (1 - f_a) * (1 - h)
lambda_b = h * (1 - f_a) + (1 - h) * f_a

# Estimate Z_tilde (you'd normally get this from simulation, but here's an approximation)
# For demonstration, let's use a typical value
# In your actual code, this comes from g_a and g_b fitted from the network
Z_tilde = 1.5  # This should be computed from your network: m / Z_factor

# Calculate b0, alpha, and gamma for type 'b' nodes
b0 = (m_edges * f_b) / (Z_tilde * mu_b + 1)
alpha = mu_b / lambda_b
gamma_param = 1 + 1 / (Z_tilde * lambda_b)

print(f"Network Parameters:")
print(f"  m = {m_edges}")
print(f"  h = {h}")
print(f"  f_a = {f_a}, f_b = {f_b}")
print(f"  mu_a = {mu_a}, mu_b = {mu_b}")
print(f"  λ_a = {lambda_a:.4f}, λ_b = {lambda_b:.4f}")
print(f"  Z̃ = {Z_tilde:.4f}")
print(f"\nType 'b' Distribution Parameters:")
print(f"  b0 = {b0:.6f}")
print(f"  α = {alpha:.6f}")
print(f"  γ = {gamma_param:.6f}")

max_k = 50

# Calculate A(k) for k = 0 to max_k
k_values = np.arange(0, max_k + 1)
A_values = [calculate_A(k, b0, alpha, gamma_param) for k in k_values]

# Print values
print(f"\nA(k) values for type 'b' nodes:")
for k, A in zip(k_values[:20], A_values[:20]):  # Print first 20
    print(f"k={k:2d}: A(k) = {A:.6e}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, A_values, 'bo-', linewidth=2, markersize=6)
plt.xlabel('k', fontsize=12)
plt.ylabel('A(k)', fontsize=12)
plt.title(f"A(k) vs k for type 'b' nodes\n(b0={b0:.4f}, α={alpha:.4f}, γ={gamma_param:.4f})", fontsize=14)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.show()