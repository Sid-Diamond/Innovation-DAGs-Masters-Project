
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# Define legend elements
legend_elements = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='red',
           markersize=10, alpha=0.7, label='Type a (n=15,341)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='blue',
           markersize=10, alpha=0.7, label='Type b (n=12,663)'),
    Line2D([0], [0], color='grey', lw=2, marker='>', markersize=10,
           markerfacecolor='grey', markeredgecolor='grey', label='Edges: 84,012')
]

# Create legend
legend = ax.legend(handles=legend_elements, fontsize=14, loc='center',
                   framealpha=0.95, edgecolor='black', fancybox=False, frameon=True)
legend.get_frame().set_linewidth(1)

# Remove axes
ax.axis('off')

# Save
plt.tight_layout()
plt.savefig('legend.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Legend saved to: legend.png")
plt.show()

