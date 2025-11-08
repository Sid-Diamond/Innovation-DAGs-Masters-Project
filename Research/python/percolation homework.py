import random

class Percolation:
    def __init__(self, height, width,grid_elements, probability):
        self.height = height
        self.width = width
        self.grid_elements = grid_elements
        self.probability = probability
        self.grid = self.create_grid()

    def create_grid(self):
        grid =[]
        for i in range (1,self.width+1):
            row = []
            for j in range (1,self.height+1):
                if random.random() <= self.probability:
                    row.append(1)
                else:
                    row.append(0)
            grid.append(row)
        return grid
                                
if __name__ == "__main__":
    percolation = Percolation(5, 5, None, 0.5)
    for row in percolation.grid:
        print(row)