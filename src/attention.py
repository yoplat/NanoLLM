import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head attention layer.

    Single-head attention:
        Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_head)) * V

    Args:
        d_model: size of the embeddings of each token
        n_head: number of attention heads
        dropout: dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):  # (B, T, d_model)
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3*d_model)
        qkv = qkv.view(
            B, T, 3, self.n_head, self.d_head
        )  # (B, T, 3, n_heads, d_head)

        q, k, v = qkv.unbind(dim=2)  # (B, T, n_heads, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # (B, n_heads, T, d_head)
        scale = 1.0 / sqrt(self.d_head)

        # Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_head)) * V
        # F.scaled_dot_product_attendion(q, k, v, is_causal=True, dropout_p=)
        # each row (last 2 dim) will be the attention weight for particular query
        attn = (
            torch.matmul(q, k.transpose(-2, -1)) * scale
        )  # (B, n_heads, T, T)
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask, float("-inf"))
        # Softmax of (-inf) is 0
        w = F.softmax(attn, dim=-1)  # per query
        w = self.dropout(w)

        ctx = torch.matmul(w, v)  # (B, n_heads, T, d_head)
        out = ctx.transpose(1, 2).reshape(B, T, C)  # (B, T, d_model)
        out = self.proj(out)
        return out


def causal_mask(T: int, device=None):
    """Returns bool mask for attention mechanism (True if masked).
    Shape: (1, 1, T, T) suitable for beign broadcasted with (B, heads, T, T)
    """

    # upper triangular (zeros on bottom-left including diag)
    m = torch.triu(
        torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1
    )
    return m.view(1, 1, T, T)
