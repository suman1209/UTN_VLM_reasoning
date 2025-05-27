import sys
sys.path.insert(0, "../../")
from src_code.data_utils.dataset import GridDataset
from src_code.data_utils.dataset_utils import CellType, draw_image_grid
from src_code.data_utils.prompt_utils import prompt_generator
# Use a pipeline as a high-level helper
from transformers import pipeline
import time
import re
NUM_EVAL = 100



# model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" # not possible with a100
# model_name  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
grid_size = 7
out_example = True
pure_language = False

dataset = GridDataset(grid_size=grid_size, seed = 42, wall_symbol="#", free_symbol=".", cell_size=10)
pipe = pipeline("text-generation", model=model_name, max_length=8192)
st_time = time.time()
with open(f"results_{model_name.split('/')[1]}_pure_language_{pure_language}_ouput_{grid_size}x{grid_size}_out_example_{out_example}.txt", "w+") as fo:
    for i in range(NUM_EVAL):
        st_time_temp = time.time()
        img_rgb1, grid_world1 = dataset[i]
        prompt = prompt_generator(grid_world1, pure_language=pure_language, img=None, img_symbol="<image>", out_example=out_example)
        # print(f"{prompt = }")
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = pipe(messages)
        result = response[0]['generated_text'][1]['content']

        gt = str(grid_world1.a_star())

        fo.write(f"### {str(i)} ###" + "\n" + result + "\n" + "time_taken: " + f"{time.time() - st_time_temp}" + "\n")
    et_time = time.time()
    fo.write(f"avg_time_for_inference: {(et_time - st_time)/NUM_EVAL}")
