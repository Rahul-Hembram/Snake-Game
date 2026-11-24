
from agent import Agent
from game import SnakeGameAI

agent = Agent()
game = SnakeGameAI()

state = game.get_state()

print("Current State:", state)

action = agent.get_action(state)

print("Selected Action:", action)

reward, done, score = game.play_step(action)

next_state = game.get_state()

print("Reward:", reward)
print("Next State:", next_state)
print("Game Over:", done)
print("Score:", score)

game.close()
