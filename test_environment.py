
from game import SnakeGameAI

game = SnakeGameAI()

state = game.get_state()

print("Initial State:", state)
print("Number of State Features:", len(state))

action = [1, 0, 0]

reward, done, score = game.play_step(action)

print("Reward:", reward)
print("Game Over:", done)
print("Score:", score)

game.close()
