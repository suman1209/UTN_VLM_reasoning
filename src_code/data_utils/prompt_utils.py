
from PIL import Image
import requests
import torch
import numpy as np


def create_prompt(dataset, img_symbol="", img=None, query_idx=1):
    """This is for OpenFlamingo"""
    num_shots = 1  # currently only one shot learning
    prompt = prompt_generator(dataset[1][1], pure_language=False, img=img, img_symbol=img_symbol, out_example=False)
    out = []
    
    for i in range(num_shots):
        img_rgb, grid_world = dataset[i]
        ascii_inp, path = str(grid_world), grid_world.a_star()
        context = f"{img_symbol} In this example, the path from the starting cell to the goal cell is {path}<|endofchunk|>"
        out.append(context)
    out.append(prompt)
    out_str = "".join(out)
    return [out_str]

def generate_inputs_for_openflamingo_rgb_and_text(tokenizer, image_processor, dataset, img_symbol="", query_idx=1):

    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
    batch_size x num_media x num_frames x channels x height x width. 
    In this case batch_size = 1, num_media = 3, num_frames = 1,
    channels = 3, height = 224, width = 224.
    """
    num_shots = 1
    # context images
    vision_x = [image_processor(dataset[i][0]).unsqueeze(0) for i in range(num_shots)]
    # query image
    vision_x.append(image_processor(dataset[query_idx][0]).unsqueeze(0))
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    out = create_prompt(dataset, img=dataset[0][0], img_symbol=img_symbol)
    lang_x = tokenizer(out, return_tensors="pt",)
    return vision_x, lang_x, out[0]

def generate_inputs_for_openflamingo_text(tokenizer, image_processor, dataset, query_idx, pure_language):
    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
    batch_size x num_media x num_frames x channels x height x width. 
    In this case batch_size = 1, num_media = 3, num_frames = 1,
    channels = 3, height = 224, width = 224.
    """
    dummy_vision_x = [image_processor(dataset[i][0]).unsqueeze(0) for i in range(1)]
    dummy_vision_x = torch.cat(dummy_vision_x, dim=0)
    dummy_vision_x = dummy_vision_x.unsqueeze(1).unsqueeze(0)
    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    grid_world = dataset[query_idx][1]
    out = [prompt_generator(grid_world, pure_language=pure_language)]
    lang_x = tokenizer(out, return_tensors="pt",)
    return dummy_vision_x, lang_x, out[0]


def prompt_generator(grid_world, pure_language=False, img=None, img_symbol="", out_example=True):
    """
    Generates a prompt for the user to solve.

    Parameters:
        grid_world (GridWorld): The grid world containing the start, goal, and obstacles.

    Returns:
        str: The prompt for the user to solve.
    """
    prompt = ""
    if img is None:
        if not pure_language:
            prompt += (
            f"Here is the grid world:\n{str(grid_world)}\n")
        prompt += (
            f"The {grid_world.start_symbol} cell is the starting cell,\n"
            f"the {grid_world.goal_symbol} cell is the goal cell,\n"
            f"the {grid_world.wall_symbol} cells are obstacles,\n"
            f"and the {grid_world.free_symbol} cells are free cells.\n"
        )
        
    else:
        if not pure_language:
            prompt += (
                f"{img_symbol} is an image of a grid world\n\n"
                "The red cell is the starting cell,\n"
                "the green cell is the goal cell,\n"
                "the gray cells are obstacles,\n"
                "and the white cells are free cells.\n"
            )
        else:
            prompt += (
                "The red cell is the starting cell,\n"
                "the green cell is the goal cell,\n"
                "the gray cells are obstacles,\n"
                "and the white cells are free cells.\n"
            )

    prompt += (
        "\nRules:\n"
        "The path must not pass through the obstacles.\n"
        "You can move up, down, left, or right from one cell to another.\n"
        "You cannot move diagonally.\n"
        "The path must be the shortest path from the starting cell to the goal cell.\n"
        "The output should be a sequence of steps to reach the goal cell.\n"
        "\nActions:\n"
        "Only give me the steps, like 'go up', 'go down', 'go left' or 'go right'\n"
        "go up: move one cell up, in coordinate is x - 1\n"
        "go down: move one cell down, in coordinate is x + 1\n"
        "go left: move one cell left, in coordinate is y - 1\n"
        "go right: move one cell right, in coordinate is y + 1\n"
    )
    if pure_language:
        prompt += ("\nCoordinate system:\n"
        "The top-left cell is (0, 0).\n"
        "The y-coordinate increases to the right.\n"
        "The x-coordinate increases downwards.\n"
        f"The starting cell is at {grid_world.start}.\n"
        f"The goal cell is at {grid_world.goal}.\n"
        f"There are some obstacles at {grid_world.obstacles}.\n")
    if out_example:
        prompt += "\nOutput example:\n"
        "('go up', 'go right', 'go right', 'go down', 'go right')\n"
    prompt +="Can you find the path from the starting cell to the goal cell?\n"
    return prompt
