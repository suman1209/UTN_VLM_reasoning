from openai import OpenAI
import sys
sys.path.insert(0, "../../")
from src_code.data_utils.dataset import GridDataset
from src_code.data_utils.dataset_utils import CellType, draw_image_grid
from src_code.data_utils.prompt_utils import prompt_generator
# Use a pipeline as a high-level helper
from transformers import pipeline
import time
import re
NUM_EVAL = 2
 
client = OpenAI(
    base_url="https://api.novita.ai/v3/openai",
    api_key="sk_acOsXuOW_PF0tM-bwFscxN_sTKC8SmVVYuoD_is1l_E",
)

model = "deepseek/deepseek-r1-0528-qwen3-8b"
stream = False # or True
max_tokens = 8192
system_content = """Be a helpful assistant"""
temperature = 1
top_p = 1
min_p = 0
top_k = 50
presence_penalty = 0
frequency_penalty = 0
repetition_penalty = 1
response_format = { "type": "text" }

grid_size = 5
out_example = True # whether to include the output example or not
pure_language = True # whether language or ascii

pure_language_str = "language" if pure_language else "ascii"
data = {"model_name": model,
        "grid_size": grid_size,
        "inp_mode": pure_language_str,
        "out_example": out_example}
num_obstacles = int(0.25 * (grid_size ** 2))
dataset = GridDataset(grid_size=grid_size, seed = 42, wall_symbol="#", free_symbol=".", 
                      obstacle_count=num_obstacles, all_solvable_grids=True,
                      add_surrounding_wall=False)
model_name = model
st_time = time.time()
responses = ""
for i in range(NUM_EVAL):
    print(f"### {i} ###")
    with open(f"results_{model_name.split('/')[1]}_{pure_language_str}_{grid_size}x{grid_size}_out_example_{out_example}.txt", "a+") as fo:
        st_time_temp = time.time()
        img_rgb1, grid_world1 = dataset[i]
        prompt = prompt_generator(grid_world1, pure_language=pure_language, img=None, img_symbol="<image>", out_example=out_example)
        chat_completion_res = client.chat.completions.create(
                                                            model=model,
                                                            messages=[
                                                                {
                                                                    "role": "system",
                                                                    "content": system_content,
                                                                },
                                                                {
                                                                    "role": "user",
                                                                    "content": prompt,
                                                                }
                                                            ],
                                                            stream=stream,
                                                            max_tokens=max_tokens,
                                                            temperature=temperature,
                                                            top_p=top_p,
                                                            presence_penalty=presence_penalty,
                                                            frequency_penalty=frequency_penalty,
                                                            response_format=response_format,
                                                            extra_body={
                                                            "top_k": top_k,
                                                            "repetition_penalty": repetition_penalty,
                                                            "min_p": min_p
                                                            }
                                                        )
    
        result = chat_completion_res.choices[0].message.content
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
    
