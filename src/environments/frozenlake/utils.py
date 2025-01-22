
from collections import deque
from typing import Dict, List

def fz_perfect_value(distances: Dict[int, int], observation, ncols: int, gamma: float):
     
    current_row = observation // ncols
    current_col = observation % ncols

    distance = distances.get((current_row, current_col), float('inf'))

    return gamma ** distance if distance != float('inf') else 0.0

def fz_compute_distances(lake_map: List[str]) -> Dict[int, int]:
    """
    Computes the distance from each cell to the goal cell, taking holes into account.
    """
    rows = len(lake_map)
    cols = len(lake_map[0])
    goal = (rows - 1, cols - 1)
    distances = {goal: 0}
    visited = set(goal)
    queue = deque([goal])

    while queue:
        row, col = queue.popleft()
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols and (new_row, new_col) not in visited:
                if lake_map[new_row][new_col] == 'H':
                    continue
                visited.add((new_row, new_col))
                distances[new_row, new_col] = distances[row, col] + 1
                queue.append((new_row, new_col))
    
    distances[goal] = 0

    return distances