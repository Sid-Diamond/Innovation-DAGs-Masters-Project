import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
import random

class PercolationModel:
    def __init__(self, width=100, height=100):
        # Grid dimensions (matching NetLogo's default -5 to 5 = 11x11)
        self.width = width
        self.height = height
        self.num_columns = width
        
        # Parameters
        self.chance_possible_q = 50
        self.search_radius_m = 2
        self.search_effort_e = 1.0
        self.max_ticks = 1000
        
        # State arrays - NetLogo coordinate system (pycor=0 is bottom)
        self.state = np.zeros((height, width), dtype=int)
        self.pathlength = np.full((height, width), -1, dtype=int)
        self.tempdist = np.full((height, width), -1, dtype=int)
        self.diamond_neighbors = [[[] for _ in range(width)] for _ in range(height)]
        self.reachable = np.zeros((height, width), dtype=bool)
        
        # Global variables matching NetLogo
        self.bpf = [None] * self.num_columns
        self.max_q_size = max(1 + (2 * self.search_radius_m * (self.search_radius_m + 1)), 
                             2 * self.num_columns)
        self.patchqueue = [None] * self.max_q_size
        self.state_freq = np.zeros(6, dtype=int)  # 0-5 states
        self.innov_sizes = []
        self.num_changes = 0
        self.deadlocked = False
        self.ticks = 0
        
        # Statistics
        self.num_reachable = 0
        self.num_possible = 0
        self.perc_reachable = 0
        
        # Colors matching NetLogo exactly
        self.state_colors = {
            0: (0, 0, 0),       # black
            1: (1, 1, 1),       # white
            2: (1, 1, 0),       # yellow
            3: (1, 0.75, 0.8),  # pink
            4: (0, 1, 0),       # green
            5: (0.5, 0.5, 0.5)  # grey
        }
    
    def setup(self):
        """NetLogo setup procedure"""
        # Clear all
        self.state = np.zeros((self.height, self.width), dtype=int)
        self.pathlength = np.full((self.height, self.width), -1, dtype=int)
        self.tempdist = np.full((self.height, self.width), -1, dtype=int)
        self.diamond_neighbors = [[[] for _ in range(self.width)] for _ in range(self.height)]
        self.reachable = np.zeros((self.height, self.width), dtype=bool)
        
        self.bpf = [None] * self.num_columns
        self.state_freq = np.zeros(6, dtype=int)
        self.innov_sizes = []
        self.num_changes = 0
        self.deadlocked = False
        self.ticks = 0
        
        # Initialize patches randomly
        for y in range(self.height):
            for x in range(self.width):
                self.tempdist[y, x] = -1
                self.diamond_neighbors[y][x] = []
                if self.chance_possible_q > random.randint(0, 99):
                    self.state[y, x] = 1
                else:
                    self.state[y, x] = 0
        
        # Set baseline (pycor = 0 in NetLogo is bottom row)
        baseline_row = self.height - 1  # Bottom row in numpy array
        for x in range(self.width):
            self.state[baseline_row, x] = 3
        
        # Count possible patches
        self.num_possible = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.state[y, x] > 0 and y != baseline_row:
                    self.num_possible += 1
        
        # Update state frequencies (only for non-baseline patches initially)
        for y in range(self.height):
            if y != baseline_row:
                for x in range(self.width):
                    self.state_freq[self.state[y, x]] += 1
        
        # Calculate initial pathlength and BPF
        self.calc_pathlength_via_possibles()
        self.calc_pathlength()
        
        # Update BPF for all viable patches
        for y in range(self.height):
            for x in range(self.width):
                if self.state[y, x] == 3:
                    self.update_bpf(y, x)
    
    def calc_one_neighborhood(self, py, px):
        """Calculate Manhattan neighborhood for a patch"""
        diamond = []
        q_start = 0
        q_size = 0
        
        # Reset tempdist
        self.tempdist = np.full((self.height, self.width), -1, dtype=int)
        
        # Initialize queue
        self.patchqueue[q_start + q_size] = (py, px)
        q_size += 1
        self.tempdist[py, px] = 0
        
        while q_size > 0:
            q_size -= 1
            cur_y, cur_x = self.patchqueue[q_start]
            q_start += 1
            
            if self.tempdist[cur_y, cur_x] < self.search_radius_m:
                # Check neighbors4 (von Neumann)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cur_y + dy, cur_x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if self.tempdist[ny, nx] == -1:
                            self.tempdist[ny, nx] = self.tempdist[cur_y, cur_x] + 1
                            diamond.append((ny, nx))
                            self.patchqueue[q_start + q_size] = (ny, nx)
                            q_size += 1
        
        # Reset tempdist for diamond patches
        for dy, dx in diamond:
            self.tempdist[dy, dx] = -1
        self.tempdist[py, px] = -1
        
        self.diamond_neighbors[py][px] = diamond
    
    def calc_pathlength_via_possibles(self):
        """Calculate path through patches with state >= 1"""
        self.pathlength = np.full((self.height, self.width), -1, dtype=int)
        q_start = 0
        q_size = 0
        
        # Start from baseline
        baseline_row = self.height - 1
        for x in range(self.width):
            self.patchqueue[(q_start + q_size) % self.max_q_size] = (baseline_row, x)
            q_size = (q_size + 1) % self.max_q_size
            self.pathlength[baseline_row, x] = 0
        
        # BFS
        while q_size > 0:
            if q_size >= self.max_q_size:
                print("Queue is too big!")
                break
            
            q_size -= 1
            cur_y, cur_x = self.patchqueue[q_start]
            q_start = (q_start + 1) % self.max_q_size
            cur_pathlength = self.pathlength[cur_y, cur_x]
            
            # Check neighbors4
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cur_y + dy, cur_x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.state[ny, nx] >= 1 and self.pathlength[ny, nx] == -1:
                        self.pathlength[ny, nx] = cur_pathlength + 1
                        self.patchqueue[(q_start + q_size) % self.max_q_size] = (ny, nx)
                        q_size = (q_size + 1) % self.max_q_size
        
        # Update reachable
        baseline_row = self.height - 1
        for y in range(self.height):
            for x in range(self.width):
                self.reachable[y, x] = (y != baseline_row and 
                                       self.state[y, x] > 0 and 
                                       self.pathlength[y, x] != -1)
        
        self.num_reachable = np.sum(self.reachable)
        self.perc_reachable = self.num_reachable / self.num_possible if self.num_possible > 0 else 0
        
        self.highlight_unreachables()
    
    def highlight_unreachables(self):
        """Mark unreachable patches for visualization"""
        # Handled in visualization by checking reachable status
        pass
    
    def go(self):
        """Main go procedure"""
        if self.ticks >= self.max_ticks or self.deadlocked:
            return False
        
        self.search_from_bpf()
        self.calc_pathlength()
        self.ticks += 1
        return True
    
    def search_from_bpf(self):
        """Search from best practice frontier"""
        for column_index, patch_coord in enumerate(self.bpf):
            if patch_coord is not None:
                py, px = patch_coord
                
                if len(self.diamond_neighbors[py][px]) == 0:
                    self.calc_one_neighborhood(py, px)
                
                num_neighbors = len(self.diamond_neighbors[py][px])
                if num_neighbors > 0:
                    test_chance = self.search_effort_e / num_neighbors
                    
                    for ny, nx in self.diamond_neighbors[py][px]:
                        if self.state[ny, nx] == 1 and self.reachable[ny, nx]:
                            self.search_site(ny, nx, test_chance)
    
    def search_site(self, y, x, test_chance):
        """Try to advance site from state 1 to 2"""
        if test_chance > random.random():
            self.state[y, x] = 2
            self.state_freq[2] += 1
            self.state_freq[1] -= 1
            self.num_changes += 1
    
    def calc_pathlength(self):
        """Calculate path length and update states"""
        self.pathlength = np.full((self.height, self.width), -1, dtype=int)
        q_start = 0
        q_size = 0
        
        # Start from baseline
        baseline_row = self.height - 1
        for x in range(self.width):
            self.patchqueue[(q_start + q_size) % self.max_q_size] = (baseline_row, x)
            q_size = (q_size + 1) % self.max_q_size
            self.pathlength[baseline_row, x] = 0
        
        # BFS
        while q_size > 0:
            if q_size >= self.max_q_size:
                print("Queue is too big!")
                break
            
            q_size -= 1
            cur_y, cur_x = self.patchqueue[q_start]
            q_start = (q_start + 1) % self.max_q_size
            cur_pathlength = self.pathlength[cur_y, cur_x]
            
            # Check neighbors4
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cur_y + dy, cur_x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.pathlength[ny, nx] == -1:
                        if self.state[ny, nx] >= 2:
                            self.pathlength[ny, nx] = cur_pathlength + 1
                            
                            if self.state[ny, nx] == 2:
                                self.state[ny, nx] = 3
                                self.state_freq[3] += 1
                                self.state_freq[2] -= 1
                                self.num_changes += 1
                                self.update_bpf(ny, nx)
                            
                            self.patchqueue[(q_start + q_size) % self.max_q_size] = (ny, nx)
                            q_size = (q_size + 1) % self.max_q_size
    
    def update_bpf(self, y, x):
        """Update best practice frontier"""
        column_index = x
        
        if self.bpf[column_index] is None:
            self.bpf[column_index] = (y, x)
        else:
            old_y, old_x = self.bpf[column_index]
            # In NetLogo, higher pycor means higher up, in numpy lower y means higher
            if y < old_y:  
                self.innov_sizes.insert(0, old_y - y)
                self.bpf[column_index] = (y, x)
    
    def get_display_state(self, y, x):
        """Get state for display including BPF marking"""
        # Check if it's BPF
        if self.bpf[x] == (y, x):
            return 4
        # Check if unreachable
        if self.state[y, x] > 0 and not self.reachable[y, x]:
            return 5
        return self.state[y, x]

# Visualization
def run_simulation():
    model = PercolationModel()
    model.setup()
    
    fig = plt.figure(figsize=(14, 8))
    
    # Create main grid for visualization
    ax_grid = plt.axes([0.05, 0.3, 0.5, 0.6])
    
    # Create initial grid display
    display_grid = np.zeros((model.height, model.width, 3))
    for y in range(model.height):
        for x in range(model.width):
            state = model.get_display_state(y, x)
            display_grid[y, x] = model.state_colors[state]
    
    im = ax_grid.imshow(display_grid, interpolation='nearest', aspect='equal')
    ax_grid.set_title('Percolation Model of Innovation')
    ax_grid.set_xlabel('Columns')
    ax_grid.set_ylabel('Innovation Height')
    ax_grid.set_xticks(range(model.width))
    ax_grid.set_yticks(range(model.height))
    ax_grid.grid(True, alpha=0.3)
    ax_grid.invert_yaxis()  # Match NetLogo coordinates
    
    # Legend in top right
    legend_elements = [
        mpatches.Patch(color=(0,0,0), label='0: Not discovered'),
        mpatches.Patch(color=(1,1,1), label='1: Discovered'),
        mpatches.Patch(color=(1,1,0), label='2: Processing'),
        mpatches.Patch(color=(1,0.75,0.8), label='3: Viable'),
        mpatches.Patch(color=(0,1,0), label='4: BPF'),
        mpatches.Patch(color=(0.5,0.5,0.5), label='5: Unreachable')
    ]
    ax_grid.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Statistics in bottom right
    ax_stats = plt.axes([0.65, 0.05, 0.3, 0.35])
    ax_stats.axis('off')
    stats_text = ax_stats.text(0, 0.9, '', fontsize=10, family='monospace', 
                               verticalalignment='top')
    
    # Controls at bottom
    ax_chance = plt.axes([0.1, 0.2, 0.35, 0.03])
    ax_radius = plt.axes([0.1, 0.15, 0.35, 0.03])
    ax_effort = plt.axes([0.1, 0.1, 0.35, 0.03])
    
    slider_chance = Slider(ax_chance, 'Chance %', 0, 100, valinit=50, valstep=1)
    slider_radius = Slider(ax_radius, 'Search Radius', 1, 5, valinit=2, valstep=1)
    slider_effort = Slider(ax_effort, 'Search Effort', 0, 5, valinit=1.0, valstep=0.1)
    
    # Buttons
    ax_setup = plt.axes([0.1, 0.03, 0.08, 0.04])
    ax_step = plt.axes([0.2, 0.03, 0.08, 0.04])
    ax_run = plt.axes([0.3, 0.03, 0.08, 0.04])
    
    btn_setup = Button(ax_setup, 'Setup')
    btn_step = Button(ax_step, 'Step')
    btn_run = Button(ax_run, 'Run')
    
    running = [False]
    
    def update_display():
        display_grid = np.zeros((model.height, model.width, 3))
        for y in range(model.height):
            for x in range(model.width):
                state = model.get_display_state(y, x)
                display_grid[y, x] = model.state_colors[state]
        
        im.set_data(display_grid)
        
        # Update stats
        stats = f"""Statistics:
Ticks: {model.ticks}
Changes: {model.num_changes}
Reachable: {model.num_reachable}
Possible: {model.num_possible}
% Reachable: {model.perc_reachable:.1%}

State Frequencies:
State 0: {model.state_freq[0]}
State 1: {model.state_freq[1]}
State 2: {model.state_freq[2]}
State 3: {model.state_freq[3]}"""
        
        stats_text.set_text(stats)
        fig.canvas.draw_idle()
    
    def on_setup(event):
        model.chance_possible_q = int(slider_chance.val)
        model.search_radius_m = int(slider_radius.val)
        model.search_effort_e = slider_effort.val
        model.setup()
        update_display()
        running[0] = False
        btn_run.label.set_text('Run')
    
    def on_step(event):
        model.go()
        update_display()
    
    def on_run(event):
        running[0] = not running[0]
        btn_run.label.set_text('Stop' if running[0] else 'Run')
        
        def animate():
            if running[0] and model.go():
                update_display()
                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.01)
                animate()
            else:
                running[0] = False
                btn_run.label.set_text('Run')
        
        if running[0]:
            animate()
    
    btn_setup.on_clicked(on_setup)
    btn_step.on_clicked(on_step)
    btn_run.on_clicked(on_run)
    
    update_display()
    plt.show()

if __name__ == "__main__":
    run_simulation()