"""The compact TabICLv2 neural architecture.

The implementation follows the architecture described in TabICLv2 Section 3
and Appendix A: repeated feature grouping, early target-aware embeddings,
induced column attention, row CLS aggregation with RoPE, and dataset-wise ICL
with QASSMax.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


class MPSafeLinear(nn.Linear):
    """A Linear layer that adds bias separately.

    PyTorch issue #192934 reports incorrect ``F.linear`` results with bias on
    rank-3 tensors on some virtualized Apple Silicon environments. Performing
    the matmul and bias addition separately is equivalent and avoids that path.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        output = F.linear(inputs, self.weight, None)
        return output if self.bias is None else output + self.bias


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [MPSafeLinear(input_dim, hidden_dim), nn.GELU()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(MPSafeLinear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class ClassEmbedding(nn.Embedding):
    """Class lookup initialized like a one-hot vector followed by Linear."""

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, -1 / math.sqrt(self.num_embeddings), 1 / math.sqrt(self.num_embeddings))


class TargetEmbedding(nn.Module):
    def __init__(self, task: str, max_classes: int, output_dim: int) -> None:
        super().__init__()
        self.task = task
        self.embedding: nn.Module
        if task == "classification":
            self.embedding = ClassEmbedding(max_classes, output_dim)
        else:
            self.embedding = MPSafeLinear(1, output_dim)

    def forward(self, y: Tensor) -> Tensor:
        if self.task == "classification":
            return self.embedding(y.long())
        return self.embedding(y.to(torch.float32).unsqueeze(-1))


class RotaryEmbedding(nn.Module):
    """RoPE used by the feature-wise row transformer."""

    def __init__(self, head_dim: int, theta: float = 100_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        half = head_dim // 2
        self.register_buffer("inv_freq", theta ** torch.linspace(0.0, -1.0, half + 1)[:-1], persistent=False)
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        seq_len = inputs.shape[-2]
        if (
            self.cos_cache.numel() == 0
            or self.cos_cache.device != inputs.device
            or self.cos_cache.shape[0] < seq_len
        ):
            positions = torch.arange(seq_len, device=inputs.device, dtype=self.inv_freq.dtype)
            angles = positions[:, None] * self.inv_freq[None, :]
            self.sin_cache, self.cos_cache = angles.sin(), angles.cos()

        sin = self.sin_cache[:seq_len].to(inputs.dtype)
        cos = self.cos_cache[:seq_len].to(inputs.dtype)
        first, second = inputs.chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class QASSMax(nn.Module):
    """Query-aware scalable-softmax query rescaling."""

    def __init__(self, num_heads: int, head_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.base_mlp = make_mlp(1, hidden_dim, num_heads * head_dim)
        self.query_mlp = make_mlp(head_dim, hidden_dim, head_dim)
        final = self.query_mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, queries: Tensor, context_length: int) -> Tensor:
        log_length = queries.new_tensor(math.log(max(context_length, 1))).reshape(1, 1)
        base = self.base_mlp(log_length).reshape(1, self.num_heads, 1, self.head_dim)
        gate = 1 + torch.tanh(self.query_mlp(queries))
        return queries * base * gate


class Attention(nn.Module):
    """Multi-head attention with optional RoPE and QASSMax."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        rope: bool = False,
        qassmax: bool = False,
        qass_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.q_proj = MPSafeLinear(embed_dim, embed_dim)
        self.k_proj = MPSafeLinear(embed_dim, embed_dim)
        self.v_proj = MPSafeLinear(embed_dim, embed_dim)
        self.out_proj = MPSafeLinear(embed_dim, embed_dim)
        self.rope = RotaryEmbedding(self.head_dim) if rope else None
        self.qassmax = QASSMax(num_heads, self.head_dim, qass_hidden_dim) if qassmax else None

    def _split_heads(self, inputs: Tensor) -> Tensor:
        return inputs.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        q = self._split_heads(self.q_proj(queries))
        k = self._split_heads(self.k_proj(context))
        v = self._split_heads(self.v_proj(context))
        if self.qassmax is not None:
            q = self.qassmax(q, k.shape[-2])
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(-3, -2).flatten(-2)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        ff_factor: int,
        dropout: float,
        rope: bool = False,
        qassmax: bool = False,
        qass_hidden_dim: int = 64,
        layernorm_bias: bool = True,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim, bias=layernorm_bias)
        self.ff_norm = nn.LayerNorm(embed_dim, bias=layernorm_bias)
        self.attention = Attention(
            embed_dim,
            num_heads,
            dropout=dropout,
            rope=rope,
            qassmax=qassmax,
            qass_hidden_dim=qass_hidden_dim,
        )
        self.ff = make_mlp(embed_dim, embed_dim * ff_factor, embed_dim, dropout)

    def forward(self, queries: Tensor, context: Tensor | None = None) -> Tensor:
        normalized_queries = self.attn_norm(queries)
        normalized_context = normalized_queries if context is None else self.attn_norm(context)
        hidden = queries + self.attention(normalized_queries, normalized_context)
        return hidden + self.ff(self.ff_norm(hidden))


class InducedTransformerBlock(nn.Module):
    """Set-Transformer ISAB: inputs -> inducing tokens -> inputs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layernorm_bias = config.task == "classification"
        self.inducing = nn.Parameter(torch.randn(1, config.inducing_tokens, config.embed_dim) * 0.02)
        self.read = TransformerBlock(
            config.embed_dim,
            config.col_heads,
            ff_factor=config.ff_factor,
            dropout=config.dropout,
            qassmax=True,
            qass_hidden_dim=config.qass_hidden_dim,
            layernorm_bias=layernorm_bias,
        )
        self.broadcast = TransformerBlock(
            config.embed_dim,
            config.col_heads,
            ff_factor=config.ff_factor,
            dropout=config.dropout,
            layernorm_bias=layernorm_bias,
        )

    def forward(self, inputs: Tensor, train_rows: int) -> Tensor:
        inducing = self.inducing.expand(inputs.shape[0], -1, -1)
        inducing = self.read(inducing, inputs[:, :train_rows])
        return self.broadcast(inputs, inducing)


class MicroTabICLv2(nn.Module):
    """A scalable TabICLv2 model that accepts a batch of in-context tasks.

    Parameters
    ----------
    config:
        Model profile. ``x`` has shape ``(tasks, train+test rows, features)``
        and ``y_train`` has shape ``(tasks, train rows)``.
    """

    FEATURE_SHIFT_BASE: Final[int] = 2

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = (config or ModelConfig()).validate()
        config = self.config
        layernorm_bias = config.task == "classification"

        self.x_embedding = MPSafeLinear(config.feature_group_size, config.embed_dim)
        self.target_embedding_col = TargetEmbedding(config.task, config.max_classes, config.embed_dim)
        self.target_embedding_icl = TargetEmbedding(config.task, config.max_classes, config.icl_dim)
        self.col_blocks = nn.ModuleList(InducedTransformerBlock(config) for _ in range(config.col_blocks))

        self.row_cls = nn.Parameter(torch.randn(1, 1, config.row_cls_tokens, config.embed_dim) * 0.02)
        self.row_blocks = nn.ModuleList(
            TransformerBlock(
                config.embed_dim,
                config.row_heads,
                ff_factor=config.ff_factor,
                dropout=config.dropout,
                rope=True,
                layernorm_bias=layernorm_bias,
            )
            for _ in range(config.row_blocks)
        )
        self.row_norm = nn.LayerNorm(config.embed_dim, bias=layernorm_bias)

        self.icl_blocks = nn.ModuleList(
            TransformerBlock(
                config.icl_dim,
                config.icl_heads,
                ff_factor=config.ff_factor,
                dropout=config.dropout,
                qassmax=True,
                qass_hidden_dim=config.qass_hidden_dim,
                layernorm_bias=layernorm_bias,
            )
            for _ in range(config.icl_blocks)
        )
        self.output_norm = nn.LayerNorm(config.icl_dim, bias=layernorm_bias)
        self.output_head = make_mlp(config.icl_dim, config.icl_dim * 2, config.out_dim)

    def _group_features(self, x: Tensor) -> Tensor:
        n_features = x.shape[-1]
        indices = torch.arange(n_features, device=x.device)
        groups = [
            x[..., (indices + (self.FEATURE_SHIFT_BASE**group - 1)) % n_features]
            for group in range(self.config.feature_group_size)
        ]
        return torch.stack(groups, dim=-1)

    @staticmethod
    def _standardize_x(x: Tensor, train_rows: int) -> Tensor:
        train = x[:, :train_rows]
        mean = train.mean(dim=1, keepdim=True)
        std = train.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def _column_stage(self, embeddings: Tensor, train_rows: int) -> Tensor:
        batch, rows, features, width = embeddings.shape
        columns = embeddings.transpose(1, 2).reshape(batch * features, rows, width)
        for block in self.col_blocks:
            columns = block(columns, train_rows)
        return columns.reshape(batch, features, rows, width).transpose(1, 2)

    def _row_stage(self, embeddings: Tensor) -> Tensor:
        batch, rows, _, width = embeddings.shape
        cls = self.row_cls.expand(batch, rows, -1, -1)
        hidden = torch.cat((cls, embeddings), dim=2).reshape(batch * rows, -1, width)
        for block in self.row_blocks[:-1]:
            hidden = block(hidden)
        hidden = self.row_blocks[-1](hidden[:, : self.config.row_cls_tokens], hidden)
        hidden = self.row_norm(hidden)
        return hidden.reshape(batch, rows, self.config.icl_dim)

    def forward(self, x: Tensor, y_train: Tensor) -> Tensor:
        if x.ndim != 3 or y_train.ndim != 2:
            raise ValueError("Expected x=(batch, rows, features), y_train=(batch, train_rows)")
        if x.shape[0] != y_train.shape[0]:
            raise ValueError("x and y_train batch sizes differ")
        train_rows = y_train.shape[1]
        if train_rows < 1 or train_rows >= x.shape[1]:
            raise ValueError("x must contain at least one training row and one test row")

        x = self._standardize_x(x.to(torch.float32), train_rows)
        embeddings = self.x_embedding(self._group_features(x))
        target_col = self.target_embedding_col(y_train).unsqueeze(2)
        embeddings = torch.cat((embeddings[:, :train_rows] + target_col, embeddings[:, train_rows:]), dim=1)
        embeddings = self._column_stage(embeddings, train_rows)
        rows = self._row_stage(embeddings)

        target_icl = self.target_embedding_icl(y_train)
        rows = torch.cat((rows[:, :train_rows] + target_icl, rows[:, train_rows:]), dim=1)
        for block in self.icl_blocks[:-1]:
            rows = block(rows, rows[:, :train_rows])
        test_rows = self.icl_blocks[-1](rows[:, train_rows:], rows[:, :train_rows])
        return self.output_head(self.output_norm(test_rows))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
