import torch.nn as nn


class Feedforward(nn.Module):
    """FFN with expansion
    Args:
        d_model: token embedding size
        mult: expansion factor
    """

    def __init__(
        self, d_model: int, mult: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
