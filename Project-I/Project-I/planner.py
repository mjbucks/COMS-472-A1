import numpy as np
from typing import List, Tuple, Optional
import scipy

def dfs(grid, start, end):
    """A DFS example"""
    rows, cols = len(grid), len(grid[0])
    stack = [start]
    visited = set()
    parent = {start: None}

    # Consider all 8 possible moves (up, down, left, right, and diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal moves

    while stack:
        x, y = stack.pop()
        if (x, y) == end:
            # Reconstruct the path
            path = []
            while (x, y) is not None:
                path.append((x, y))
                if parent[(x, y)] is None:
                    break  # Stop at the start node
                x, y = parent[(x, y)]
            return path[::-1]  # Return reversed path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and (nx, ny) not in visited:
                stack.append((nx, ny))
                parent[(nx, ny)] = (x, y)

    return None  # Return None if no path is found

def astar(grid, start, end):
    """
    
    Parameters:
    - grid: A 2D list where 0 represents walkable cells and 1 represents obstacles
    - start: A tuple (row, col) representing the starting position
    - end: A tuple (row, col) representing the goal position
    
    Returns:
    - A list of tuples representing the path from start to end, or None if no path exists
    """
    rows, cols = len(grid), len(grid[0])
    
    # Check if start and end are valid positions
    if not (0 <= start[0] < rows and 0 <= start[1] < cols) or not (0 <= end[0] < rows and 0 <= end[1] < cols):
        return None
    
    # Check if start or end positions are obstacles
    if grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1:
        return None
    
    # Already at destination
    if start == end:
        return [start]
    
    # Initialize closed set (visited nodes)
    closed_set = set()
    
    # Initialize open set as a dictionary to track positions and their metadata
    open_set = {start: {'g': 0, 'f': 0, 'parent': None}}
    # List to track order for selection
    open_list = [start]
    
    # Directions: up, down, left, right, and diagonals
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while open_list:
        # Get node with lowest f_score
        current = min(open_list, key=lambda pos: open_set[pos]['f'])
        open_list.remove(current)
        
        # If we reached the destination
        if current == end:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = open_set[current]['parent']
            return path[::-1]  # Return reversed path
        
        # Add to closed set
        closed_set.add(current)
        
        # Check all possible directions
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Skip if neighbor is invalid or an obstacle or already visited
            if (not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols) or 
                grid[neighbor[0]][neighbor[1]] == 1 or 
                neighbor in closed_set):
                continue
                
            # Calculate g score (cost from start to neighbor through current)
            move_cost = 1.414 if dx != 0 and dy != 0 else 1.0  # Diagonal vs cardinal
            tentative_g = open_set[current]['g'] + move_cost
            
            # If this is a new node or we found a better path
            if neighbor not in open_set or tentative_g < open_set[neighbor]['g']:
                # Calculate h score (heuristic - Euclidean distance to goal)
                h = ((neighbor[0] - end[0]) ** 2 + (neighbor[1] - end[1]) ** 2) ** 0.5
                
                # Update or add the neighbor
                open_set[neighbor] = {
                    'g': tentative_g,
                    'f': tentative_g + h,
                    'parent': current
                }
                
                # Add to open list if not already there
                if neighbor not in open_list:
                    open_list.append(neighbor)
    
    # No path found
    return None

def plan_path(world: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], 
             algorithm: str = "astar") -> Optional[np.ndarray]:
    """
    Computes a path from the start position to the end position 
    using a specified planning algorithm.

    Parameters:
    - world (np.ndarray): A 2D numpy array representing the grid environment.
      - 0 represents a walkable cell.
      - 1 represents an obstacle.
    - start (Tuple[int, int]): The (row, column) coordinates of the starting position.
    - end (Tuple[int, int]): The (row, column) coordinates of the goal position.
    - algorithm (str): The algorithm to use for path planning. Options: "dfs", "astar".
      Default is "astar".

    Returns:
    - np.ndarray: A 2D numpy array where each row is a (row, column) coordinate of the path.
      The path starts at 'start' and ends at 'end'. If no path is found, returns None.
    """
    # Ensure start and end positions are tuples of integers
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))

    # Convert the numpy array to a list of lists for compatibility with the pathfinding functions
    world_list: List[List[int]] = world.tolist()

    # Select and execute the appropriate pathfinding algorithm
    if algorithm.lower() == "dfs":
        path = dfs(world_list, start, end)
    elif algorithm.lower() == "astar":
        path = astar(world_list, start, end)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'dfs' or 'astar'")

    return np.array(path) if path else None
