from scipy.spatial.distance import cityblock
import json
# Mapping movement commands to coordinate changes
MOVE_MAP = {
    "go up": (-1, 0),
    "go down": (1, 0),
    "go left": (0, -1),
    "go right": (0, 1)
}

def convert_commands_to_path(start, commands):
    """Convert movement commands into a list of coordinates."""
    path = [start]
    x, y = start
    for command in commands:
        if command in MOVE_MAP:
            dx, dy = MOVE_MAP[command]
            path.append((x + dx, y + dy))
            x += dx
            y += dy
    return path

def calculate_score(pred_path, grid_world, debug=False):
    result = {
        "Exact_Match": 0,
        "success": 0,
        "collision": 0,
        "goal_distance": 0,
        "path_length_difference": 0,
        "pass_through_goal": 0
    }

    obstacles = grid_world.obstacles
    start = grid_world.start
    goal = grid_world.goal

    optimal_path = grid_world.a_star()
    if optimal_path is None:
        if pred_path == ("not solvable"):
            result["success"] = 1
            return result
        else:
            result["success"] = 0
            optimal_length = 0
    else:
        # Path length
        optimal_length = len(optimal_path)


    
    path = convert_commands_to_path(start, pred_path)
    path_length = len(pred_path)
    extra_steps = abs(path_length - optimal_length)
    result["path_length_difference"] = extra_steps

    # Obstacle 
    obstacles_hit = 0
    if any(step in obstacles for step in path):
        obstacles_hit += 1
    # Check if the path outside the grid
    if any(step[0] < 0 or step[1] < 0 or step[0] >= grid_world.rows or step[1] >= grid_world.cols for step in path):
        obstacles_hit += 1
    result["collision"] = obstacles_hit

    # Distance to goal (Manhattan distance)
    end_distance = cityblock(path[-1], goal)
    result["goal_distance"] = end_distance

    # Success reward
    if path[-1] == goal and obstacles_hit == 0:
        result["success"] = 1
        if optimal_length == path_length:
            result["Exact_Match"] = 1
    else:
        for step in path:
            if step == goal:
                result["pass_through_goal"] = 1
                break

    

    if debug:
        print(f"result: {result}")
        print(f"Predict path: {pred_path}")
        print(f"Optimal path: {optimal_path}")
        # print(f"Path length: {path_length}")
        # print(f"Optimal path length: {optimal_length}")
        # print(f"path_length_difference: {extra_steps}")
        # print(f"Obstacles: {obstacles}")
        # print(f"Obstacle hit: {obstacles_hit}")
        # print(f"Goal: {goal}") 
        # print(f"End position: {path[-1]}")
        # print(f"End distance: {end_distance}")

    return result


def eval_results(path_results, dataset, debug=False):
    path_results_len = len(path_results)
    em = 0
    success = 0
    collision = 0
    goal_distance = 0
    path_length_difference = 0
    pass_through_goal = 0

    for i in range(path_results_len):
        if debug:
            print(i)
        path_result = path_results[i]
        _, grid_world = dataset[i]
        try:
            result = calculate_score(path_result, grid_world, debug=debug)
            em += result["Exact_Match"]
            success += result["success"]
            collision += result["collision"]
            goal_distance += result["goal_distance"]
            path_length_difference += result["path_length_difference"]
            pass_through_goal += result["pass_through_goal"]
        except Exception as e:
            print(f"Error in evaluating path {i}: {e}")

    return {
        "Exact Match (%)": 100 * (em / path_results_len),
        "success rate (%)": 100 * (success / path_results_len),
        "average collision": collision / path_results_len,
        "average goal_distance": goal_distance / path_results_len,
        "average path_length_difference": path_length_difference / path_results_len,
        "pass_through_goal": pass_through_goal
    }

def eval_result_from_json(json_path: str, dataset, debug=False):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Convert each list to a tuple, keeping the order by sorted numerical keys
    result = []
    for key in sorted(data, key=lambda x: int(x)):
        result.append(tuple(data[key]))
    
    print(eval_results(result, dataset=dataset, debug=debug))

if __name__ == "__main__":
    import sys
    ## Env variables and preparation stuffs
    sys.path.insert(0, "../../")
    from src_code.data_utils.dataset import GridDataset
    from src_code.data_utils.dataset_utils import CellType
    from src_code.eval_utils.eval import calculate_score, eval_results
    grid_size = 3

    num_obstacles = int(0.25 * (grid_size ** 2))
    dataset = GridDataset(grid_size=grid_size, seed = 42, wall_symbol="#", free_symbol=".", 
                          obstacle_count=num_obstacles, all_solvable_grids=True,
                          add_surrounding_wall=False)
    eval_result_from_json("results.json", dataset, True)