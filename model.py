
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Deep Q-Network
class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = torch.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def save(self, file_name='model.pth'):

        torch.save(self.state_dict(), file_name)


# DQN Trainer
class QTrainer:

    def __init__(self, model, target_model, lr, gamma):

        self.lr = lr
        self.gamma = gamma

        self.model = model
        self.target_model = target_model

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr
        )

        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):

        state = torch.as_tensor(
            np.asarray(state), dtype=torch.float32
        )

        next_state = torch.as_tensor(
            np.asarray(next_state), dtype=torch.float32
        )

        action = torch.as_tensor(
            np.asarray(action), dtype=torch.float32
        )

        reward = torch.as_tensor(
            np.asarray(reward), dtype=torch.float32
        )

        done = torch.as_tensor(
            np.asarray(done), dtype=torch.float32
        )

        # Handle a single experience
        if state.ndim == 1:

            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)

            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Predicted Q-values for selected actions
        action_indices = action.argmax(dim=1, keepdim=True)

        pred = self.model(state).gather(
            1, action_indices
        ).squeeze(1)

        # Target Q-values
        with torch.no_grad():

            next_q_values = self.target_model(next_state)

            max_next_q = next_q_values.max(dim=1)[0]

            target = (
                reward +
                self.gamma * max_next_q * (1 - done)
            )

        # Calculate loss
        loss = self.criterion(pred, target)

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()
