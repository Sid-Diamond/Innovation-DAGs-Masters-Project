import numpy as np
import random

class Percolation:
    def __init__(self, height, width, grid_elements, probability):
        self.height = height
        self.width = width
        self.grid_elements = grid_elements
        self.probability = probability
        self.grid = self.create_grid()

    def create_grid(self):
        list_of_elements =[]
        
        x_values = []
        for i in range (1, self.width+1): #no +1 => if self width =5 then only 4 elements processed
            x_value = []
            x_value.append(i)   
            x_values.append(x_value)

        y_values = []
        for j in range (1, self.height+1):
            column =[]
            column.append(j) in y_values

        for x in x_values:
            for y in y_values:
                list_of_elements.append((x,y))

        for x in x_values:
            for y in y_values:
                 if random.random() <= self.probability:
                    state =[]
                    state.append(1) in state
                 else:
                    state.append(0) in state

        return list_of_elements, state


        

