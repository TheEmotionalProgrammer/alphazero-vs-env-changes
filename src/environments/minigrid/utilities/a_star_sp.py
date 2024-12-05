from queue import PriorityQueue
import math

def heuristic(a, b):
    """Heuristic function for A* (Manhattan distance)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_pathfinding(env):
    """Compute the shortest path using A*."""
    # Access the unwrapped environment to get agent_start_pos and grid
    base_env = env.unwrapped
    start = tuple(base_env.agent_start_pos)
    goal = None

    # Extract the goal position from the grid
    for x in range(base_env.grid.width):
        for y in range(base_env.grid.height):
            cell = base_env.grid.get(x, y)
            if cell and cell.type == "goal":
                goal = (x, y)
                break

    if not goal:
        raise ValueError("Goal not found in the grid!")

    # A* algorithm
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        _, current = open_set.get()

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighboring positions
            neighbor = (current[0] + dx, current[1] + dy)

            # Skip invalid neighbors
            if (
                neighbor[0] < 0 or neighbor[0] >= base_env.grid.width or
                neighbor[1] < 0 or neighbor[1] >= base_env.grid.height
            ):
                continue

            # Check for obstacles
            cell = base_env.grid.get(neighbor[0], neighbor[1])
            if cell and cell.type in ["wall", "lava"]:
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                open_set.put((f_score[neighbor], neighbor))

    raise ValueError("No path to the goal!")


def compute_actions_from_path(path, env):
    """Convert a path into actions for the agent."""
    # Access the unwrapped environment for the start position and direction
    base_env = env.unwrapped
    current_pos = tuple(base_env.agent_start_pos)
    current_dir = base_env.agent_start_dir

    actions = []

    for next_pos in path:
        dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
        desired_dir = {
            (-1, 0): 2,  # Down
            (1, 0): 0,   # Up
            (0, -1): 3,  # Left
            (0, 1): 1    # Right
        }[(dx, dy)]

        # Calculate shortest turning direction
        clockwise_turns = (desired_dir - current_dir) % 4
        counter_clockwise_turns = (current_dir - desired_dir) % 4

        if clockwise_turns <= counter_clockwise_turns:
            # Turn right for optimal adjustment
            for _ in range(clockwise_turns):
                actions.append(1)  # Turn right
                current_dir = (current_dir + 1) % 4
        else:
            # Turn left for optimal adjustment
            for _ in range(counter_clockwise_turns):
                actions.append(0)  # Turn left
                current_dir = (current_dir - 1) % 4

        actions.append(2)  # Move forward
        current_pos = next_pos

    return actions