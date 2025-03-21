

def prompt_generator(grid_world, pure_language=False, img=None, img_symbol=""):
    """
    Generates a prompt for the user to solve.

    Parameters:
        grid_world (GridWorld): The grid world containing the start, goal, and obstacles.

    Returns:
        str: The prompt for the user to solve.
    """
    prompt = "Can you find the path from the starting cell to the goal cell?\n"
    if img is None:
        prompt += (
            f"Here is the grid world:\n{str(grid_world)}\n"
            f"The {grid_world.start_symbol} cell is the starting cell,\n"
            f"the {grid_world.goal_symbol} cell is the goal cell,\n"
            f"the {grid_world.wall_symbol} cells are obstacles,\n"
            f"and the {grid_world.free_symbol} cells are free cells.\n"
        )
    else:
        if not pure_language:
            prompt += (
                f"Here is an image of the grid world:\n{img_symbol}\n"
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
        "\nCoordinate system:\n"
        "The top-left cell is (0, 0).\n"
        "The y-coordinate increases to the right.\n"
        "The x-coordinate increases downwards.\n"
        f"The starting cell is at {grid_world.start}.\n"
        f"The goal cell is at {grid_world.goal}.\n"
        f"There are some obstacles at {grid_world.obstacles}.\n"
        "\nOutput example:\n"
        "('go up', 'go right', 'go right', 'go down', 'go right')"
    )

    return prompt
