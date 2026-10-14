import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding based on embedding matrix.
    Args:
        max_len: max sequence length
        d_model: token embedding size
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        _, T, _ = x.shape  # (B, T, d_model)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.emb(pos)  # (T, d_model)
        return x + pos_emb.unsqueeze(0)  # (B, T, d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """Precomputed positional embeddings based on sinusoidals.
    Args:
        max_len: max sequence length
        d_model: token embedding size

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    with pos being the token pos while i being the i-th value of the embedding
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        # position:  (max_len, 1)  varies along rows, constant across columns
        # div_term:  (1, n_freqs)  varies along columns, constant across rows
        # product:  (max_len, n_freqs)  entry [pos, i] = pos * div_term[i]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        # non model parameter buffer
        self.register_buffer("pe", pe, persistent=False)  # (max_len, d_model)

    def forward(self, x: torch.Tensor):
        _, T, _ = x.shape
        return x + self.pe[:T].unsqueeze(0)
