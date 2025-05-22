import sys
sys.path.insert(0, "../../")
from src_code.data_utils.dataset import GridDataset
from src_code.data_utils.dataset_utils import CellType, draw_image_grid
from src_code.data_utils.prompt_utils import prompt_generator
# Use a pipeline as a high-level helper
from transformers import pipeline

dataset = GridDataset(grid_size=5, seed = 42, wall_symbol="#", free_symbol=".", cell_size=10)
img_rgb1, grid_world1 = dataset[0]



pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")


prompt = prompt_generator(grid_world1, img=None, img_symbol="<image>"))
messages = [
    {"role": "user", "content": prompt},
]
response = pipe(messages).text

print(str(grid_world1.a_start()))
print(f"{response = }")