import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import numpy as np

# Create figure and axis with equal aspect ratio and high DPI
fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=300)
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(0.3, 4.7)  # Extended upper limit to accommodate title
ax.set_aspect('equal')  # This ensures circles are perfectly circular
ax.axis('off')

# Define node positions for better layout
positions = {
    1: (0, 2),
    2: (2, 1),
    3: (5, 1),
    4: (5, 3),
    5: (2, 3),
    6: (7, 2)
}

# Node radius
node_radius = 0.4

# Function to calculate point on circle edge
def get_edge_point(center, target, radius):
    """Calculate point on circle edge in direction of target"""
    cx, cy = center
    tx, ty = target
    
    # Calculate angle
    dx = tx - cx
    dy = ty - cy
    distance = np.sqrt(dx**2 + dy**2)
    
    # Unit vector in direction of target
    ux = dx / distance
    uy = dy / distance
    
    # Point on circle edge
    edge_x = cx + ux * radius
    edge_y = cy + uy * radius
    
    return edge_x, edge_y

# Define edges with their properties
# Format: (from, to, color, curvature)
edges = [
    (1, 2, 'black', -0.3),
    (1, 5, 'black', 0.3),      # Red edge from source
    (2, 5, 'black', 0),
    (2, 3, 'black', 0),
    (3, 4, 'black', 0),
    (5, 4, 'black', 0),       # Blue edge
    (4, 6, 'black', 0),       # Blue edge to sink
    (3, 6, 'black', -0.3),
]

# Draw edges with arrows
for start, end, color, curve in edges:
    start_pos = positions[start]
    end_pos = positions[end]
    
    # Calculate edge points
    x1, y1 = get_edge_point(start_pos, end_pos, node_radius)
    x2, y2 = get_edge_point(end_pos, start_pos, node_radius)
    
    if curve != 0:
        # Create curved arrow
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={curve}",
            arrowstyle='->',
            color=color,
            linewidth=2,
            zorder=1,
            mutation_scale=25,
            shrinkA=0,
            shrinkB=0
        )
    else:
        # Create straight arrow
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->',
            color=color,
            linewidth=2,
            zorder=1,
            mutation_scale=25,
            shrinkA=0,
            shrinkB=0
        )
    ax.add_patch(arrow)

# Draw nodes
for node, (x, y) in positions.items():
    circle = Circle((x, y), node_radius, color='white', ec='black', linewidth=1.5, zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, str(node), 
            ha='center', va='center_baseline',
            fontsize=20, fontweight='bold', zorder=3,
            fontfamily='Times New Roman')


# Add title
ax.text(2, 4, 'DAG Citation Network', 
        ha='center', va='bottom',
        fontsize=21, fontweight='normal',
        fontfamily='Times New Roman')

# Remove all padding and margins
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Save with tight bounding box and high resolution
plt.savefig('criticality_graph.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()