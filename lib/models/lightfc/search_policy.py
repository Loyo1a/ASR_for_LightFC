import torch
from torch import nn


class AdaptiveSearchActorCritic(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=32, num_actions=4):
        super().__init__()
        self.policy_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_actions),
        )
        self.value_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        logits = self.policy_model(state)
        value = self.value_model(state).squeeze(-1)
        return logits, value
