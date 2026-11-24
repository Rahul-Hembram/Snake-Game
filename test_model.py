import torch
from model import Linear_QNet

model = Linear_QNet(11, 256, 3)

state = torch.tensor(
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    dtype=torch.float
)

prediction = model(state)

print("Input State:", state)
print("Predicted Q-values:", prediction)

action = torch.argmax(prediction).item()

print("Selected Action:", action)