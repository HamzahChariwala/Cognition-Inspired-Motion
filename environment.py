import numpy as np
import random

ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

class Maze(object):
    def __init__(self, maze_map):
        self.maze = maze_map
        self.steps = 0
        self.penalty_boundary = -10
        self.penalty_obstacle = -5
        # self.reward_landmark = -1
        self.robot_position = self.random_start_position()
        self.construct_allowed_states()

    def random_start_position(self):
        empty_spaces = [(y, x) for y in range(self.maze.shape[0]) for x in range(self.maze.shape[1]) if self.maze[y, x] == 0]
        start_position = random.choice(empty_spaces)
        self.maze[start_position[0], start_position[1]] = 4 # Set initial robot position
        return start_position

    def is_allowed_move(self, state, action):
        y, x = state
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        if y < 0 or x < 0 or y >= self.maze.shape[0] or x >= self.maze.shape[1]:
            return False
        return True # allow moving into any cell to handle penalties and rewards

    def construct_allowed_states(self):
        allowed_states = {}
        for y, row in enumerate(self.maze):
            for x, col in enumerate(row):
                allowed_states[(y, x)] = []
                for action in ACTIONS:
                    if self.is_allowed_move((y, x), action):
                        allowed_states[(y, x)].append(action)
        self.allowed_states = allowed_states

    def update_maze(self, action):
        y, x = self.robot_position
        self.maze[y, x] = 0 # reset the current position
        y += ACTIONS[action][0]
        x += ACTIONS[action][1]
        self.robot_position = (y, x)
        self.maze[y, x] = 4 # set the new position
        self.steps += 1

    def is_game_over(self):
        return self.steps >= 3

    def get_state_and_reward(self):
        return self.robot_position, self.give_reward()

    def give_reward(self):
        y, x = self.robot_position
        if self.maze[y, x] == 1:
            return self.penalty_boundary
        elif self.maze[y, x] == 2:
            return self.penalty_obstacle
        # elif self.maze[y, x] == 3:
        #     return self.reward_landmark
        else:
            return -1 # small penalty for each step taken

