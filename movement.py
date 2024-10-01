import numpy as np
import random

def get_valid_moves(position, grid):
    x, y = position
    moves = [
        (x - 1, y),    # Up
        (x + 1, y),    # Down
        (x, y - 1),    # Left
        (x, y + 1),    # Right
        (x - 1, y - 1), # Up-Left
        (x - 1, y + 1), # Up-Right
        (x + 1, y - 1), # Down-Left
        (x + 1, y + 1)  # Down-Right
    ]
    valid_moves = [(nx, ny) for nx, ny in moves if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] not in [1, 2]]
    return valid_moves

def select_next_position(position, reward_array, grid):
    valid_moves = get_valid_moves(position, grid)
    if not valid_moves:
        return position  # No valid moves, stay in place
    rewards = [reward_array[nx, ny] for nx, ny in valid_moves]
    max_reward = max(rewards)
    best_moves = [move for move, reward in zip(valid_moves, rewards) if reward == max_reward]
    return random.choice(best_moves)

def calculate_angle(point1, point2):
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    angle = np.arctan2(delta_y, delta_x)
    return np.degrees(angle)

def get_surrounding_positions(position):
    x, y = position
    surrounding_positions = [
        (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
        (x, y - 1),               (x, y + 1),
        (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)
    ]
    return surrounding_positions

def select_best_position(surrounding_positions, reward_array, grid):
    valid_moves = [(nx, ny) for nx, ny in surrounding_positions if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] not in [1, 2]]
    if not valid_moves:
        return surrounding_positions[0]  # No valid moves, return the first surrounding position
    rewards = [reward_array[nx, ny] for nx, ny in valid_moves]
    min_reward = min(rewards)
    best_moves = [move for move, reward in zip(valid_moves, rewards) if reward == min_reward]
    return best_moves[0]

def calculate_gradient(point1, point2):
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    if delta_x == 0:
        return np.inf 
    return delta_y / delta_x

def start_position(grid):
    start_position = (random.randint(0, 49), random.randint(0, 49))
    while grid[start_position] in [1, 2]:
        start_position = (random.randint(0, 49), random.randint(0, 49))
    return start_position

def random_step(position, grid):
    potential_cells = get_valid_moves(position, grid)
    length = len(potential_cells)
    index = random.randint(0, length-1)
    return potential_cells[index]

def calculate_next_position(current_position, five_steps_ago_position, prediction):

    # Calculate the current movement vector
    delta_y = current_position[1] - five_steps_ago_position[1]
    delta_x = current_position[0] - five_steps_ago_position[0]
    current_angle = np.degrees(np.arctan2(delta_x, delta_y))
    print(current_angle)
    
    # Calculate the new direction by adding the prediction angle
    new_angle = current_angle + prediction
    
    # Convert the new angle to radians
    new_angle_rad = np.radians(new_angle)
    
    # Determine the movement direction
    delta_row = int(round(np.sin(new_angle_rad)))
    print(f"Delta Row {delta_row}")
    delta_col = int(round(np.cos(new_angle_rad)))
    print(f"Delta Col {delta_col}")
    
    # Calculate the next position
    next_row = current_position[0] + delta_row
    next_col = current_position[1] + delta_col
    
    # Return the next position as a tuple
    return (next_row, next_col)

