
import os
import csv
import torch
import matplotlib.pyplot as plt

from game import SnakeGameAI
from agent import Agent

# Configuration
TOTAL_GAMES = 100
MODEL_PATH = "models/improved_best_model.pth"

os.makedirs("results", exist_ok=True)

# Load trained model
agent = Agent()

agent.model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)

agent.model.eval()

# Initialize game
game = SnakeGameAI()

scores = []
total_food = 0
best_score = 0

print("====================================")
print("       DQN MODEL EVALUATION")
print("====================================")

# Run 100 games
for episode in range(1, TOTAL_GAMES + 1):

    game.reset()

    while True:

        # Get current state
        state = game.get_state()

        # Greedy action: no random exploration
        state_tensor = torch.tensor(
            state, dtype=torch.float
        )

        with torch.no_grad():
            prediction = agent.model(state_tensor)

        move = torch.argmax(prediction).item()

        action = [0, 0, 0]
        action[move] = 1

        # Perform action
        reward, done, score = game.play_step(action)

        if done:
            scores.append(score)
            total_food += score

            if score > best_score:
                best_score = score

            print(
                f"Game: {episode}/{TOTAL_GAMES} | "
                f"Score: {score} | "
                f"Best: {best_score}"
            )

            break

# Calculate statistics
average_score = sum(scores) / len(scores)
minimum_score = min(scores)

print("\n====================================")
print("          EVALUATION RESULTS")
print("====================================")

print("Total Games:", TOTAL_GAMES)
print("Highest Score:", best_score)
print("Lowest Score:", minimum_score)
print("Average Score:", round(average_score, 2))
print("Total Food Eaten:", total_food)

# Save results to CSV
with open(
    "results/evaluation_results.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(["Game", "Score"])

    for i, score in enumerate(scores, start=1):
        writer.writerow([i, score])

# Plot performance graph
plt.figure(figsize=(10, 5))

plt.plot(scores, label="Score per Game")

plt.axhline(
    average_score,
    linestyle="--",
    label=f"Average Score: {average_score:.2f}"
)

plt.xlabel("Game Number")
plt.ylabel("Score")
plt.title("DQN Snake Game - 100 Game Evaluation")

plt.legend()
plt.grid(True)

plt.savefig("results/evaluation_graph.png")
plt.show()

game.close()

print("\nEvaluation completed successfully!")
print("Results saved in the results folder.")
