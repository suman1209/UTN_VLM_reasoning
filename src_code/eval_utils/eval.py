from scipy.spatial.distance import cityblock

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
        "success": 0,
        "collision": 0,
        "goal_distance": 0,
        "path_length_difference": 0,
    }

    optimal_path = grid_world.a_star()
    obstacles = grid_world.obstacles
    start = grid_world.start
    goal = grid_world.goal
    
    path = convert_commands_to_path(start, pred_path)

    # Path length
    path_length = len(pred_path)
    optimal_length = len(optimal_path)
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
    if path[-1] == goal:
        result["success"] = 1

    if debug:
        print(f"Predict path: {pred_path}")
        print(f"Optimal path: {optimal_path}")
        print(f"Path length: {path_length}")
        print(f"Optimal path length: {optimal_length}")
        print(f"path_length_difference: {extra_steps}")
        print(f"Obstacles: {obstacles}")
        print(f"Obstacle hit: {obstacles_hit}")
        print(f"Goal: {goal}") 
        print(f"End position: {path[-1]}")
        print(f"End distance: {end_distance}")

    return result


def eval_results(path_results, dataset, debug=False):
    path_results_len = len(path_results)
    success = 0
    collision = 0
    goal_distance = 0
    path_length_difference = 0

    for i in range(path_results_len):
        path_result = path_results[i]
        _, grid_world = dataset[i]
        result = calculate_score(path_result, grid_world, debug=debug)
        success += result["success"]
        collision += result["collision"]
        goal_distance += result["goal_distance"]
        path_length_difference += result["path_length_difference"]

    return {
        "success rate (%)": 100 * (success / path_results_len),
        "average collision": collision / path_results_len,
        "average goal_distance": goal_distance / path_results_len,
        "average path_length_difference": path_length_difference / path_results_len
    }