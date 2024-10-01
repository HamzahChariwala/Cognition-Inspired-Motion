import numpy as np

ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

class Agent(object):
    def __init__(self, states, alpha=0.10, random_factor=0.2): # 80% explore, 20% exploit
        self.state_history = [((0, 0), 0)] # state, reward
        self.alpha = alpha
        self.random_factor = random_factor
        self.G = {}
        self.init_reward(states)

    def init_reward(self, states):
        for i, row in enumerate(states):
            for j, col in enumerate(row):
                self.G[(j, i)] = np.random.uniform(low=0.1, high=1.0)
    
    def choose_action(self, state, allowedMoves):
        if not allowedMoves:
            # print(f"No allowed moves for state {state}")
            return None  # No valid moves, return None to handle later
        maxG = -10e15
        next_move = None
        randomN = np.random.random()
        if randomN < self.random_factor:
            # if random number below random factor, choose random action
            next_move = np.random.choice(allowedMoves)
        else:
            # if exploiting, gather all possible actions and choose one with the highest G (reward)
            for action in allowedMoves:
                new_state = tuple([sum(x) for x in zip(state, ACTIONS[action])])
                if self.G[new_state] >= maxG:
                    next_move = action
                    maxG = self.G[new_state]

        # print(f"Choosing action {next_move} for state {state} with allowed moves {allowedMoves}")
        return next_move

    def update_state_history(self, state, reward):
        # print(f"Updating state history with state {state} and reward {reward}")
        self.state_history.append((state, reward))

    def learn(self):
        target = 0

        # print(f"Learning with state history: {self.state_history}")
        for prev, reward in reversed(self.state_history):
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])
            target += reward

        self.state_history = []

        self.random_factor -= 10e-5 # decrease random factor each episode of play

