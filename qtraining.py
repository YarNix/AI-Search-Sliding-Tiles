import pickle
import random
from collections import defaultdict
import pandas as pd
import numpy as np

STEPS = ['U', 'D', 'L', 'R']
BLANK = 0
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

type StateData = tuple[int, int, int, int, int, int, int, int, int]

def apply_action(state: StateData, action: str) -> StateData | None:
    idx = state.index(BLANK)
    x, y = divmod(idx, 3)
    dx, dy = 0, 0
    if action == 'U': dx = -1
    elif action == 'D': dx = 1
    elif action == 'L': dy = -1
    elif action == 'R': dy = 1

    nx, ny = x + dx, y + dy
    if not (0 <= nx < 3 and 0 <= ny < 3):
        return None

    new_idx = nx * 3 + ny
    state_list = list(state)
    state_list[idx], state_list[new_idx] = state_list[new_idx], state_list[idx]
    return tuple(state_list)

def step(state: StateData, action: str) -> tuple[StateData, int]:
    next_state = apply_action(state, action)
    if next_state is None:
        return state, -5
    elif next_state == GOAL_STATE:
        return next_state, 100
    else:
        return next_state, -1


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = pd.DataFrame(columns=STEPS, dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def ensure_state(self, state: StateData):
        state_str = str(state)
        if state_str not in self.q_table.index:
            self.q_table.loc[state_str] = [0.0 for _ in STEPS]

    def choose_action(self, state: StateData) -> str:
        self.ensure_state(state)
        if random.random() < self.epsilon:
            return random.choice(STEPS)
        return self.q_table.loc[str(state)].idxmax()

    def update(self, state: StateData, action: str, reward: float, next_state: StateData):
        self.ensure_state(state)
        self.ensure_state(next_state)
        state_str, next_str = str(state), str(next_state)

        old_q = self.q_table.at[state_str, action]
        max_next_q = self.q_table.loc[next_str].max()
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table.at[state_str, action] = new_q

    def save(self, path="qtable.pkl"):
        self.q_table.to_pickle(path)


def scramble_board(state, moves=30):
    for _ in range(moves):
        valid = []
        for action in STEPS:
            new_state = apply_action(state, action)
            if new_state:
                valid.append(new_state)
        state = random.choice(valid)
    return state


def train(agent: QLearningAgent, episodes=2000):
    for episode in range(episodes):
        if episode % 100 == 0:
            print('Running episode', episode)
        state = scramble_board(GOAL_STATE)
        for _ in range(100):  # Max steps per episode
            action = agent.choose_action(state)
            next_state, reward = step(state, action)
            agent.update(state, action, reward, next_state)
            if next_state == GOAL_STATE:
                break
            state = next_state

    agent.save("qtable.pkl")
    print("Training complete. Q-table saved to qtable.pkl.")


if __name__ == "__main__":
    agent = QLearningAgent()
    train(agent)
