import numpy as np
from environment import Maze
from agent import Agent
import matplotlib.pyplot as plt
from map import MapGeneration

def create_reward_array(reward_dict, shape=(50, 50)):
    reward_array = np.zeros(shape)
    for (x, y), reward in reward_dict.items():
        if x < shape[0] and y < shape[1]:
            reward_array[x, y] = reward
    return reward_array

def plot_reward_array(reward_array):
    plt.imshow(reward_array, cmap='magma', interpolation='nearest')
    plt.colorbar()
    plt.title("Reward Distribution")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

def generate_rewards(maze_map):

    maze = Maze(maze_map)
    robot = Agent(maze.maze, alpha=0.1, random_factor=0.25)
    moveHistory = []

    for i in range(3000):
        if i % 1000 == 0:
            print(i)

        maze = Maze(maze_map.copy()) # reinitialize the maze with a new start position
        
        while not maze.is_game_over():
            state, _ = maze.get_state_and_reward() # get the current state
            allowed_moves = maze.allowed_states.get(state, [])
            action = robot.choose_action(state, allowed_moves) # choose an action (explore or exploit)
            if action is None:
                # print(f"No valid moves from state {state}, breaking the loop.")
                break  # No valid moves available, break the loop
            maze.update_maze(action) # update the maze according to the action
            state, reward = maze.get_state_and_reward() # get the new state and reward
            robot.update_state_history(state, reward) # update the robot memory with state and reward

        robot.learn() # robot should learn after every episode
        moveHistory.append(maze.steps) # get a history of number of steps taken to plot later

# plt.semilogy(moveHistory, "b--")
# plt.show()
    reward_dict = robot.G
    reward_array = create_reward_array(reward_dict)
    # plot_reward_array(reward_array)
    return reward_array

def standardise_array(array):
    mean_val = np.mean(array)
    std_val = np.std(array)
    standardised_array = (array - mean_val) / std_val
    return standardised_array

def average_array(grid, number):
    empty_array = np.zeros(grid.shape)
    for i in range(number):
        empty_array +=  generate_rewards(grid)
    return standardise_array(empty_array)

if __name__ == "__main__":
    sample1 = MapGeneration(50, 50)
    map1 = sample1.generate_map_1()
    array = average_array(map1, 15)
    plot_reward_array(array)
    to_plot = array.flatten()

    plt.figure(figsize=(8, 6))
    plt.hist(to_plot, bins=20, edgecolor='k')
    plt.title('Histogram of Standardized Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

