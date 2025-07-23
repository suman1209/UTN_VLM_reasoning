from dataset import GridDataset
from dataset_utils import CellType, draw_image_grid, dataset_generator
from prompt_utils import prompt_generator
import json
# only adjust the following variable

DATASET_INFORMATION = {
    "grid_size": 5,
    "num_images": 100,
    "seed": 42,
    "wall_symbol": "#",
    "free_symbol": ".",
    "cell_size": 100,
    "obstacle_count": 6,
    "all_solvable_grids": True
}

dataset = GridDataset(grid_size=DATASET_INFORMATION["grid_size"],
                      seed = DATASET_INFORMATION["seed"],
                      wall_symbol=DATASET_INFORMATION["wall_symbol"],
                      free_symbol=DATASET_INFORMATION["free_symbol"],
                      cell_size=DATASET_INFORMATION["cell_size"],
                      obstacle_count=DATASET_INFORMATION["obstacle_count"],
                      all_solvable_grids=DATASET_INFORMATION["all_solvable_grids"])
# Generating a dataset

gt_dict = {}
dataset_path = "/var/lit2425/jenga/suman/UTN_VLM_reasoning/dataset/2D-GridDataset/5x5"
dataset = GridDataset(grid_size=5, seed = 42, wall_symbol="#", free_symbol=".", cell_size=10, obstacle_count=6, all_solvable_grids=True)
for i in range(DATASET_INFORMATION["num_images"]):
    img_rgb, grid_world = dataset[i]
    img_rgb.save(f"{dataset_path}/images/grid_{i}.jpg", format="JPEG")
    gt = grid_world.a_star()
    gt_dict[i] = gt
    

file_path = f"{dataset_path}/labels/gt_labels.json"

# Write the dictionary to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(gt_dict, json_file, indent=4)

# Write the dictionary to a JSON file
with open(f"{dataset_path}/description_information.txt", 'w') as json_file:
    json.dump(DATASET_INFORMATION, json_file, indent=4)

print(f"Dataset saved to {dataset_path}")
