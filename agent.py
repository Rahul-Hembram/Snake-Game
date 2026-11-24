
import torch
import random
import numpy as np

from collections import deque
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 0.0005


class Agent:

    def __init__(self):

        self.n_games = 0

        self.epsilon = 80
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        # Online network
        self.model = Linear_QNet(11, 256, 3)

        # Target network
        self.target_model = Linear_QNet(11, 256, 3)

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.target_model.eval()

        # DQN trainer
        self.trainer = QTrainer(
            self.model,
            self.target_model,
            lr=LR,
            gamma=self.gamma
        )

        self.target_update_counter = 0

    def update_target_network(self):

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.target_update_counter = 0

    def remember(self, state, action, reward, next_state, done):

        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def train_short_memory(
        self, state, action, reward, next_state, done
    ):

        self.trainer.train_step(
            state, action, reward, next_state, done
        )

        self.target_update_counter += 1

        if self.target_update_counter >= 100:
            self.update_target_network()

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:

            mini_sample = random.sample(
                self.memory, BATCH_SIZE
            )

        else:

            mini_sample = list(self.memory)

        if not mini_sample:
            return

        states, actions, rewards, next_states, dones = zip(
            *mini_sample
        )

        self.trainer.train_step(
            states,
            actions,
            rewards,
            next_states,
            dones
        )

        self.target_update_counter += 1

        if self.target_update_counter >= 100:
            self.update_target_network()

    def get_action(self, state):

        # Gradually reduce exploration over 1000 games.
        # Keep a small amount of exploration.
        self.epsilon = max(
            5,
            80 - int(self.n_games * 0.075)
        )

        final_move = [0, 0, 0]

        # Exploration
        if random.randint(0, 200) < self.epsilon:

            move = random.randint(0, 2)

        # Exploitation
        else:

            state0 = torch.tensor(
                np.asarray(state),
                dtype=torch.float
            )

            with torch.no_grad():
                prediction = self.model(state0)

            move = torch.argmax(prediction).item()

        final_move[move] = 1

        return final_move
