# 🐍 AI Snake Game – Improved DQN

An AI-powered Snake Game developed using **Python, Pygame, PyTorch, and Deep Reinforcement Learning**. This project explores how an intelligent agent can learn to play the classic Snake Game using a Deep Q-Network (DQN).

The system combines an interactive game environment with reinforcement learning techniques to enable intelligent decision-making through rewards, penalties, and continuous learning.

---

## 🎮 Game Preview

### 🏠 Start Screen
![Snake Game Start Screen](Screenshot%202026-11-24%20220000.png)

### 🎯 Gameplay
![Snake Game Gameplay](Screenshot%202026-11-24%20220015.png)

### 🏆 Game Over Screen
![Snake Game Over](Screenshot%202026-11-24%20220024.png)

---

## 📌 Project Overview

Traditional Snake Games require players to manually control the snake, collect food, and avoid collisions.

This project explores the application of **Deep Reinforcement Learning (DRL)** to develop an intelligent agent capable of learning game strategies through interaction with its environment.

The agent uses a Deep Q-Network to estimate Q-values for possible actions and improve its decision-making over multiple training episodes.

---

## 🚀 Key Features

- 🧠 Deep Q-Network (DQN) implementation.
- 🎯 Epsilon-Greedy action selection strategy.
- 🔁 Experience Replay for efficient learning.
- 🎮 Interactive Snake Game using Pygame.
- ⚡ Target Network for improved training stability.
- 📊 Separate training and evaluation modules.
- 🏆 Score tracking and game performance monitoring.
- 🔄 Play, Pause, Restart, and Exit controls.

---

## 🧠 Deep Reinforcement Learning

The project uses Deep Q-Learning, which combines Q-Learning with neural networks to estimate the expected future rewards of different actions.

### How DQN Works

1. The agent observes the current state of the game.
2. The neural network predicts Q-values for available actions.
3. The agent selects an action using an epsilon-greedy strategy.
4. The environment executes the selected action.
5. The agent receives a reward or penalty.
6. The experience is stored in replay memory.
7. The network learns from previously stored experiences.
8. The process repeats to improve the agent's performance.

### Core Components

| Component | Description |
|---|---|
| Deep Q-Network | Predicts Q-values for possible actions |
| Experience Replay | Stores and samples previous experiences |
| Target Network | Helps stabilize the learning process |
| Epsilon-Greedy | Balances exploration and exploitation |
| Reward Function | Guides the agent's learning process |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Neural networks and deep learning |
| Pygame | Game development and visualization |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |

---

## 📂 Project Structure

```text
Snake-Game/
│
├── models/                 # Trained model files
│
├── results/                # Training and evaluation results
│
├── agent.py                # DQN agent implementation
├── model.py                # Neural network architecture
├── game.py                 # Snake game environment
├── train.py                # Agent training script
├── evaluate.py             # Model evaluation
├── play.py                 # Game execution and interface
│
├── test_agent.py           # Agent testing
├── test_environment.py     # Environment testing
├── test_model.py           # Model testing
├── test_pygame.py          # Pygame testing
│
├── requirements.txt        # Project dependencies
├── .gitignore
└── README.md
