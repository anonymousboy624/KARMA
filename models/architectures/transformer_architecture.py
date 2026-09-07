import math

import torch
import torch.nn as nn

from utils.types.param_types import TransformerModelParam


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # persistent=False: deterministic from d_model/max_len, no need to
        # store it in checkpoints or require it to match on load.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Transformer(nn.Module):
    """Causal Transformer encoder for multivariate time-series multi-step
    regression forecasting.

    Uses a causal (subsequent-position-masked) self-attention encoder so
    each position's representation only depends on itself and earlier
    positions — the same autoregressive constraint LSTM/TCN get for free
    from their sequential/dilated-causal structure.

    Args:
      input_size: number of input features (D)
      d_model: attention embedding dimension
      nhead: number of attention heads
      num_layers: stacked encoder layers
      dim_feedforward: FFN hidden dimension inside each encoder layer
      dropout: dropout throughout
      forecast_steps: number of future timesteps to predict (k)
      output_size: number of output features per timestep
    """

    def __init__(self, model_param: TransformerModelParam):
        super().__init__()
        self.forecast_steps = model_param.forecast_steps
        self.output_size = model_param.output_size
        self.d_model = model_param.d_model

        self.input_proj = nn.Linear(model_param.input_size, model_param.d_model)
        self.pos_encoding = PositionalEncoding(model_param.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_param.d_model,
            nhead=model_param.nhead,
            dim_feedforward=model_param.dim_feedforward,
            dropout=model_param.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=model_param.num_layers
        )

        self.layer_norm = nn.LayerNorm(model_param.d_model)
        self.dropout = nn.Dropout(model_param.dropout)

        # head: map last position's representation -> forecast_steps * output_size
        self.head = nn.Sequential(
            nn.Linear(model_param.d_model, model_param.d_model),
            nn.ReLU(),
            nn.Dropout(model_param.dropout),
            nn.Linear(
                model_param.d_model,
                model_param.forecast_steps * model_param.output_size,
            ),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, D) -> (batch, T, d_model), causal self-attention."""
        T = x.shape[1]
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.encoder(h, mask=mask, is_causal=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T_input, D)
        returns: (batch, forecast_steps, output_size)
        """
        h = self._encode(x)  # (batch, T, d_model)
        last_hidden = h[:, -1, :]  # (batch, d_model)
        out = self.layer_norm(last_hidden)
        out = self.dropout(out)
        out = self.head(out)  # (batch, forecast_steps * output_size)
        out = out.view(x.size(0), self.forecast_steps, self.output_size)
        return out

    def predict(self, x):
        """Returns raw forecasts (regression outputs)."""
        return self.forward(x)
