
import os
import csv
import matplotlib.pyplot as plt

from game import SnakeGameAI
from agent import Agent


# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# Training configuration
MAX_EPISODES = 1000

agent = Agent()

game = SnakeGameAI()

scores = []
mean_scores = []

record = -1
total_score = 0


print("====================================")
print("    IMPROVED DQN TRAINING STARTED")
print("====================================")


# Training loop
for episode in range(MAX_EPISODES):

    game.reset()

    state_old = game.get_state()

    while True:

        # Select action
        action = agent.get_action(state_old)

        # Execute action
        reward, done, score = game.play_step(action)

        # Get next state
        state_new = game.get_state()

        # Train on current experience
        agent.train_short_memory(
            state_old,
            action,
            reward,
            state_new,
            done
        )

        # Store experience
        agent.remember(
            state_old,
            action,
            reward,
            state_new,
            done
        )

        state_old = state_new

        if done:

            agent.n_games += 1

            # Replay memory training
            agent.train_long_memory()

            # Save best model
            if score > record:

                record = score

                agent.model.save(
                    "models/improved_best_model.pth"
                )

            scores.append(score)

            total_score += score

            mean_score = total_score / agent.n_games

            mean_scores.append(mean_score)

            print(
                f"Episode: {agent.n_games} | "
                f"Score: {score} | "
                f"Record: {record} | "
                f"Average: {mean_score:.2f}"
            )

            # Save checkpoint every 25 episodes
            if agent.n_games % 25 == 0:

                agent.model.save(
                    "models/improved_latest_model.pth"
                )

            break


# Save final model
agent.model.save(
    "models/improved_final_model.pth"
)

game.close()


# Save training results
with open(
    "results/improved_training_results.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Episode",
        "Score",
        "Average Score"
    ])

    for i in range(len(scores)):

        writer.writerow([
            i + 1,
            scores[i],
            mean_scores[i]
        ])


# Calculate rolling average
rolling_average = []

window = 25

for i in range(len(scores)):

    start = max(0, i - window + 1)

    average = sum(scores[start:i + 1]) / (
        i - start + 1
    )

    rolling_average.append(average)


# Plot graph
plt.figure(figsize=(12, 6))

plt.plot(
    scores,
    alpha=0.4,
    label="Score per Episode"
)

plt.plot(
    rolling_average,
    label="25-Episode Rolling Average"
)

plt.plot(
    mean_scores,
    label="Overall Average"
)

plt.xlabel("Episodes")
plt.ylabel("Score")

plt.title("Improved DQN Snake Training Performance")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/improved_training_graph.png"
)

plt.show()


print("====================================")
print("       TRAINING COMPLETED")
print("====================================")

print("Total Episodes:", len(scores))
print("Best Score:", record)
print("Final Average Score:", mean_scores[-1])

print("Model saved successfully!")
