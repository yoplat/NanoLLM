import torch.nn as nn
from attention import MultiHeadSelfAttention
from ffn import Feedforward


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm.
    Args:
        d_model: token embedding size
        n_head: number of attention head
        dropout: dropout probability
    """

    def __init__(
        self, d_model: int, n_head: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.ln1 == nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = Feedforward(d_model, mult=4, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
