"""
The GridWorld class is generated using DeepSeek and adapted for our purposes
prompt: I want a python script that generates a 2d grid structure as a numpy array and
        gives options to choose the start and goal grids, random walls in between and a function
        to calculate the shortest path from the start to the goal , don't forget to add a seed and
        implement this as a class.
        create a small Enum called cellType and have members like start, goal, wall, free_cell
        and update the code using that enum
"""
import numpy as np
import heapq
import random
from enum import Enum, auto

from PIL import Image
import matplotlib.pyplot as plt

def draw_image_grid(image_title_pairs, cols=3, figsize=(15, 10)):
    """
    Draws a grid of images with titles.

    :param image_title_pairs: List of tuples, where each tuple contains a PIL Image and a title string.
    :param cols: Number of columns in the grid.
    :param figsize: Size of the matplotlib figure.
    """
    num_images = len(image_title_pairs)
    rows = (num_images + cols - 1) // cols  # Calculate the number of rows needed

    # Create a figure and axis for the grid
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel()  # Flatten the axes array for easy iteration

    # Hide axes for empty subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    # Plot each image with its title
    for idx, (img, title) in enumerate(image_title_pairs):
        axes[idx].imshow(img)
        axes[idx].set_title(title)
        axes[idx].axis('off')  # Hide axes for better visualization

    plt.tight_layout()
    plt.show()
    
class CellType(Enum):
    FREE_CELL = 1  # Represents free space
    WALL = 2       # Represents a wall
    START = 3      # Represents the start position
    GOAL = 4       # Represents the goal position

class GridWorld:
    def __init__(self, rows, cols, seed=None, start_symbol="S", goal_symbol="G", wall_symbol="#", free_symbol="."):
        self.rows = rows
        self.cols = cols
        self.reset()  # Initialize grid with free cells
        self.start = None
        self.goal = None
        self.obstacles = []
        self.seed = seed
        self.start_symbol = start_symbol
        self.goal_symbol = goal_symbol
        self.wall_symbol = wall_symbol
        self.free_symbol = free_symbol
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def set_start(self, row, col):
        if self.grid[row, col] == CellType.FREE_CELL.value:
            self.grid[row, col] = CellType.START.value
            self.start = (row, col)
        else:
            raise ValueError(f"Start position {self.start} must be on a free cell but is {self.grid[row, col]}.")

    def set_goal(self, row, col):
        if self.grid[row, col] == CellType.FREE_CELL.value:
            self.grid[row, col] = CellType.GOAL.value
            self.goal = (row, col)
        else:
            raise ValueError(f"Goal position {self.goal} must be on a free cell but is {self.grid[row, col]}")

    def add_random_walls(self, wall_prob=0.2):
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) != self.start and (row, col) != self.goal:
                    if random.random() < wall_prob:
                        self.grid[row, col] = CellType.WALL.value
                        self.obstacles.append((row, col))

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self):
        if self.start is None or self.goal is None:
            raise ValueError("Start and goal positions must be set.")

        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                path.reverse()

                # Convert path to directions
                directions = []
                for i in range(1, len(path)):
                    dx = path[i][1] - path[i - 1][1]  # Change in column (x-axis)
                    dy = path[i][0] - path[i - 1][0]  # Change in row (y-axis)
                    if dx == -1:
                        directions.append("go left")  # Left
                    elif dx == 1:
                        directions.append("go right")  # Right
                    elif dy == -1:
                        directions.append("go up")  # Up
                    elif dy == 1:
                        directions.append("go down")  # Down
                return tuple(directions)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def get_neighbors(self, pos):
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc] != CellType.WALL.value:
                neighbors.append((nr, nc))
        return neighbors

    def __str__(self):
        grid_str = ""
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) == self.start:
                    grid_str += f"{self.start_symbol} "
                elif (row, col) == self.goal:
                    grid_str += f"{self.goal_symbol} "
                elif self.grid[row, col] == CellType.WALL.value:
                    grid_str += f"{self.wall_symbol} "
                else:
                    grid_str += f"{self.free_symbol} "
            grid_str += "\n"
        return grid_str

    def reset(self):
        self.grid = np.full((self.rows, self.cols), CellType.FREE_CELL.value, dtype=np.int8)
# Example usage:
if __name__ == "__main__":
    grid_world = GridWorld(10, 10, seed=42)
    grid_world.set_start(0, 0)
    grid_world.set_goal(9, 9)
    grid_world.add_random_walls(wall_prob=0.2)

    print("Grid World:")
    print(grid_world)

    path = grid_world.a_star()
    if path:
        print("Shortest Path:")
        for step in path:
            print(step)
    else:
        print("No path found.")