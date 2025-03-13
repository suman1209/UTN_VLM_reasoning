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
    path = []
    x, y = start
    for command in commands:
        if command in MOVE_MAP:
            dx, dy = MOVE_MAP[command]
            path.append((x + dx, y + dy))
            x += dx
            y += dy
    return path

def calculate_score(pred_path, grid_world, step_penalty=5, obstacle_penalty=10, end_penalty=5):
    score = 100
    optimal_path = grid_world.a_star()
    obstacles = grid_world.obstacles
    start = grid_world.start
    goal = grid_world.goal
    print(f"Predict path: {pred_path}")
    print(f"Optimal path: {optimal_path}")
    
    path = convert_commands_to_path(start, pred_path)

    # Path length penalty
    path_length = len(path)
    optimal_length = len(optimal_path)
    print(f"Path length: {path_length}")
    print(f"Optimal path length: {optimal_length}")
    extra_steps = abs(path_length - optimal_length)
    print(f"Step penalty: {extra_steps * step_penalty}")
    score -= extra_steps * step_penalty

    # Obstacle penalty
    print(f"Obstacles: {obstacles}")
    obstacles_hit = 0
    if any(step in obstacles for step in path):
        obstacles_hit += 1
    print(f"Obstacle penalty: {obstacles_hit * obstacle_penalty}")
    score -= obstacles_hit * obstacle_penalty

    # Distance to goal penalty (Manhattan distance)
    end_distance = cityblock(path[-1], goal)
    print(f"Goal: {goal}") 
    print(f"End position: {path[-1]}")
    print(f"End distance: {end_distance}")
    print(f"End distance penalty: {end_distance * end_penalty}")
    score -= max(0, end_distance * end_penalty)

    return int(max(0, score))  # Ensure score doesn't go below 0
