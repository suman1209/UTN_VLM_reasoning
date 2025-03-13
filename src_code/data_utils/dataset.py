from PIL import Image, ImageDraw
from torchvision.datasets import VisionDataset
from .dataset_utils import GridWorld, CellType
import random
import numpy as np
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

class GridDataset(VisionDataset):
    def __init__(self , grid_size: int, seed: int = 42, start_symbol="S",
                 goal_symbol="G", wall_symbol="#", free_symbol=".", cell_size:int = 20):
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.seed = seed
        self.grid_world = GridWorld(grid_size, grid_size, seed=grid_size, start_symbol=start_symbol,
                                    goal_symbol=goal_symbol, wall_symbol=wall_symbol, free_symbol=free_symbol)
        super(GridDataset).__init__()

    def __len__(self):
        return -1

    def __getitem__(self, idx):
        random.seed(idx)
        np.random.seed(idx)
        self.grid_world.reset()
        start = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        goal = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        while goal == start:
            goal = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.grid_world.set_start(*start)
        self.grid_world.set_goal(*goal)
        self.grid_world.add_random_walls(wall_prob=0.05)
        path = self.grid_world.a_star()
        img, ascii, path = self.render_img(), self.render_ascii(), path
        
        return img, self.grid_world

    def render_img(self):
        """
        Generates an image of the grid.

        Parameters:
            cell_size (int): Size of each cell in pixels.
        """
        colors = {
            CellType.FREE_CELL.value: (255, 255, 255),  # White
            CellType.WALL.value: (128, 128, 128),  # Grey
            CellType.START.value: (255, 0, 0),  # Red
            CellType.GOAL.value: (0, 255, 0)  # Green
        }
        cell_size = self.cell_size
        # Create a blank image
        img_width = self.grid_size * cell_size + 1
        img_height = self.grid_size * cell_size + 1
        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw each cell
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell_type = self.grid_world.grid[row, col]
                color = colors.get(cell_type, (255, 255, 255))  # Default to white if cell type not in colors
                x0 = col * cell_size
                y0 = row * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))
        return img

    def render_ascii(self):
        return str(self.grid_world)
