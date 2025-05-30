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
# model_name = "deepseek-ai/DeepSeek-R1-0528"
model_name  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
grid_sizes = [7, 6, 5, 4, 3]
for grid_size in grid_sizes:
    out_example = True # whether to include the output example or not
    pure_language = False # whether language or ascii

    pure_language_str = "language" if pure_language else "ascii"
    data = {"model_name": model_name,
            "grid_size": grid_size,
            "inp_mode": pure_language_str,
            "out_example": out_example}
    num_obstacles = int(0.25 * (grid_size ** 2))
    dataset = GridDataset(grid_size=grid_size, seed = 42, wall_symbol="#", free_symbol=".", 
                        obstacle_count=num_obstacles, all_solvable_grids=True,
                        add_surrounding_wall=False)


    pipe = pipeline("text-generation", model=model_name, max_length=8192)
    st_time = time.time()
    responses = ""
    for i in range(NUM_EVAL):
        with open(f"results_{model_name.split('/')[1]}_{pure_language_str}_{grid_size}x{grid_size}_out_example_{out_example}.txt", "a+") as fo:
            st_time_temp = time.time()
            img_rgb1, grid_world1 = dataset[i]
            prompt = prompt_generator(grid_world1, pure_language=pure_language, img=None, img_symbol="<image>", out_example=out_example)
            messages = [
                {"role": "user", "content": prompt},
            ]
            response = pipe(messages)
            result = response[0]['generated_text'][1]['content']
            # extract the final answer from the distill model
            try:
                extracted_response = f"### {i} ###" + result.split("</think>")[1]
            except Exception as e:
                extracted_response = f"### {i} ### {e}"
            gt = str(grid_world1.a_star())
            responses += extracted_response + "\n"
            fo.write(f"### {str(i)} ###" + "\n" + result + "\n" + "time_taken: " + f"{time.time() - st_time_temp}" + "\n")
    et_time = time.time()

    # final details 
    with open(f"results_{model_name.split('/')[1]}_{pure_language_str}_{grid_size}x{grid_size}_out_example_{out_example}.txt", "a+") as fo:
        fo.write(f"avg_time_for_inference: {(et_time - st_time)/NUM_EVAL}")
        fo.write(f"\n{str(data)}")
        fo.write(f"Extracted Responses: \n {responses}")
    
