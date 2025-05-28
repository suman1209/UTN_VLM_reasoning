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
from datasets import Dataset, DatasetDict
from copy import deepcopy
import random
from transformers import pipeline
from src_code.data_utils.prompt_utils import prompt_generator
random.seed(42)

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
    def __init__(self, rows, cols, seed=None, start_symbol="S", goal_symbol="G",
                 wall_symbol="#", free_symbol=".", add_surrounding_wall=False):
        self.rows = rows
        self.cols = cols
        assert self.rows == self.cols, "only square grids considered currently!"
        self.reset()  # Initialize grid with free cells
        self.start = None
        self.goal = None
        self.obstacles = []
        self.seed = seed
        self.start_symbol = start_symbol
        self.goal_symbol = goal_symbol
        self.wall_symbol = wall_symbol
        self.free_symbol = free_symbol
        self.add_surrounding_wall = add_surrounding_wall

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

    def add_random_walls(self, wall_prob=0.4, obstacle_count=None):
        """If the obstacle count is given, then place them randomly"""
        if obstacle_count is not None:
            if obstacle_count > self.rows * self.rows:
                raise ValueError("Number of locations cannot exceed grid size squared")
    
            locations = set()
            while len(locations) < obstacle_count:
                r = random.randint(0, self.rows - 1)
                c = random.randint(0, self.cols - 1)
                if (r, c) != self.start and (r, c) != self.goal:
                    locations.add((r, c))
            
            self.obstacles = list(locations)
            for (r, c) in self.obstacles:
                self.grid[r, c] = CellType.WALL.value
        else:
            self.obstacles = []
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

    def str_without_surrounding_walls(self):
        grid_str = ""
        for row in range(self.rows):
            grid_str += " "
            for col in range(self.cols):
                
                if (row, col) == self.start:
                    # we don't want a space after the symbol at the end for tokenization uniformity
                    if col != (self.cols - 1):
                        grid_str += f"{self.start_symbol} "
                    else:
                        grid_str += f"{self.start_symbol}"
                elif (row, col) == self.goal:
                    if col != (self.cols - 1):
                        grid_str += f"{self.goal_symbol} "
                    else:
                        grid_str += f"{self.goal_symbol}"
                elif self.grid[row, col] == CellType.WALL.value:
                    if col != (self.cols - 1):
                        grid_str += f"{self.wall_symbol} "
                    else:
                        grid_str += f"{self.wall_symbol}"
                else:
                    if col != (self.cols - 1):
                        grid_str += f"{self.free_symbol} "
                    else:
                        grid_str += f"{self.free_symbol}"
            grid_str += "\n"
        return grid_str    

    def str_with_surrounding_walls(self):
        grid_str = f" {self.wall_symbol}" * (self.rows+2) + "\n"
        for row in range(self.rows):
            grid_str += f" {self.wall_symbol} "
            for col in range(self.cols):
                
                if (row, col) == self.start:
                    # we don't want a space after the symbol at the end for tokenization uniformity
                    if col != (self.cols - 1):
                        grid_str += f"{self.start_symbol} "
                    else:
                        grid_str += f"{self.start_symbol}"
                elif (row, col) == self.goal:
                    if col != (self.cols - 1):
                        grid_str += f"{self.goal_symbol} "
                    else:
                        grid_str += f"{self.goal_symbol}"
                elif self.grid[row, col] == CellType.WALL.value:
                    if col != (self.cols - 1):
                        grid_str += f"{self.wall_symbol} "
                    else:
                        grid_str += f"{self.wall_symbol}"
                else:
                    if col != (self.cols - 1):
                        grid_str += f"{self.free_symbol} "
                    else:
                        grid_str += f"{self.free_symbol}"
            grid_str += f" {self.wall_symbol}\n"
        grid_str += f" {self.wall_symbol}" * (self.rows+2) + "\n"
        return grid_str

    def random_walk(self, max_steps=25):
        """
        Performs a random walk from start to goal.
        Returns tuple of actions if goal is reached, None otherwise.
        """
        if self.start is None or self.goal is None:
            raise ValueError("Start and goal positions must be set.")
        
        current = self.start
        actions = []
        visited = set()
        steps = 0
        
        while current != self.goal and steps < max_steps:
            # Get valid neighbors (excluding walls)
            neighbors = self.get_neighbors(current)
            
            if not neighbors:
                break  # No possible moves
                
            # Randomly choose next position (might include previously visited cells)
            next_pos = random.choice(neighbors)
            
            # Determine direction
            dx = next_pos[1] - current[1]  # Column change
            dy = next_pos[0] - current[0]  # Row change
            
            # Convert to action string
            if dx == -1:
                actions.append("go left")
            elif dx == 1:
                actions.append("go right")
            elif dy == -1:
                actions.append("go up")
            elif dy == 1:
                actions.append("go down")
                
            current = next_pos
            steps += 1
    
        return tuple(actions)

    def __str__(self):

        if self.add_surrounding_wall:
            grid_str = self.str_with_surrounding_walls()
        else:
            grid_str = self.str_without_surrounding_walls()
        return grid_str

    def reset(self):
        self.grid = np.full((self.rows, self.cols), CellType.FREE_CELL.value, dtype=np.int8)
    




def dataset_generator(dataset, sys_role, train_num=100, val_num=50, test_num=50):
    """this is used for finetuning purposes"""

    train_list = []
    val_list = []
    test_list = []

    message_temp = {
        "messages": [
            {
                "role": "system",
                "content": sys_role
            },
            {
                "role": "user",
                "content": None
            },
            {
                "role": "assistant",
                "content": None
            }
        ]
    }

    for i in range(train_num):
        img_rgb, grid_world = dataset[i]
        user_prompt = prompt_generator(grid_world, pure_language=False, img=None, img_symbol="<image>")
        results = grid_world.a_star()
        message = deepcopy(message_temp)
        message["messages"][1]["content"] = user_prompt
        message["messages"][2]["content"] = str(results)
        train_list.append(message)
    
    for j in range(train_num, train_num + val_num):
        img_rgb, grid_world = dataset[j]
        user_prompt = prompt_generator(grid_world, pure_language=False, img=None, img_symbol="<image>")
        results = grid_world.a_star()
        message = deepcopy(message_temp)
        message["messages"][1]["content"] = user_prompt
        message["messages"][2]["content"] = str(results)
        val_list.append(message)

    for k in range(train_num + val_num, train_num + val_num + test_num):
        img_rgb, grid_world = dataset[k]
        user_prompt = prompt_generator(grid_world, pure_language=False, img=None, img_symbol="<image>")
        results = grid_world.a_star()
        message = deepcopy(message_temp)
        message["messages"][1]["content"] = user_prompt
        message["messages"][2]["content"] = str(results)
        test_list.append(message)

    print(f"{len(train_list)=}")
    print(f"{len(val_list)=}")
    print(f"{len(test_list)=}")
    train_dataset = Dataset.from_list(train_list)
    val_dataset = Dataset.from_list(val_list)
    test_dataset = Dataset.from_list(test_list)
    dataset = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    })

    return dataset
    
# Example usage:
if __name__ == "__main__":
    grid_world = GridWorld(10, 10, seed=42)
    grid_world.set_start(0, 0)
    grid_world.set_goal(9, 9)
    grid_world.add_random_walls(wall_prob=0.4)

    print("Grid World:")
    print(grid_world)

    path = grid_world.a_star()
    if path:
        print("Shortest Path:")
        for step in path:
            print(step)
    else:
        print("No path found.")