"""Core PyTorch implementation of the MRPA-STFT + CA-GRU classifier.

This file contains the neural-network architecture used in the manuscript
"Multi-Resolution Phase-Aware Spatio-temporal Network for EEG-Based
Epileptic Seizure Classification".

The model expects pre-extracted features rather than raw EEG recordings.
For the default configuration, the last feature dimension consists of:
    - 192 frequency-domain features (3 modalities x 64 bins), followed by
    - 256 temporal samples/features aligned to each time-frequency frame.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyBranch(nn.Module):
    """Extract local patterns from the fused frequency representation."""

    def __init__(self, in_freq=64, out_freq=64):
        super().__init__()
        self.out_freq = out_freq

        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )
        self.reduce = nn.Conv1d(in_freq, out_freq, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C, F).

        Returns:
            Tensor of shape (B, T, C, out_freq).
        """
        batch_size, time_steps, channels, freq_dim = x.shape

        x = x.reshape(batch_size * time_steps * channels, 1, freq_dim)
        x = self.local_conv(x)
        x = self.reduce(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, time_steps, channels, self.out_freq)
        return x


class TCNBlock(nn.Module):
    """Residual causal temporal-convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.total_padding = dilation * (kernel_size - 1)

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)

        out = F.pad(x, (self.total_padding, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.pad(out, (self.total_padding, 0))
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out + residual)
        return out


class TimeBranch(nn.Module):
    """Temporal branch based on stacked residual TCN blocks."""

    def __init__(self, in_time=256, out_time=64):
        super().__init__()
        self.in_time = in_time
        self.out_time = out_time

        self.tcn = nn.Sequential(
            TCNBlock(in_channels=1, out_channels=32, kernel_size=3, dilation=1),
            TCNBlock(
                in_channels=32,
                out_channels=out_time,
                kernel_size=3,
                dilation=2,
            ),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C, temporal_dim).

        Returns:
            Tensor of shape (B, T, C, out_time).
        """
        batch_size, time_steps, channels, temporal_dim = x.shape

        x = x.reshape(batch_size * time_steps * channels, temporal_dim, 1)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.mean(dim=-1)
        x = x.reshape(batch_size, time_steps, channels, self.out_time)
        return x


class MultiFreqModalFusion(nn.Module):
    """Attention-based fusion of multiple frequency-domain modalities."""

    def __init__(self, num_modalities=3, embed_dim=64, use_attn=True):
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        self.use_attn = use_attn

        if self.use_attn:
            self.attn = nn.MultiheadAttention(
                embed_dim, num_heads=4, batch_first=True
            )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, multi_modal):
        """
        Args:
            multi_modal: Tensor of shape
                (B, T, C, num_modalities * embed_dim).

        Returns:
            fused_avg: Tensor of shape (B, T, C, embed_dim).
            attn_weights: Attention weights when enabled, otherwise None.
        """
        batch_size, time_steps, channels, _ = multi_modal.shape

        x_flat = multi_modal.reshape(batch_size * time_steps * channels, -1)
        x_seq = x_flat.view(
            batch_size * time_steps * channels,
            self.num_modalities,
            self.embed_dim,
        )

        if self.use_attn:
            fused, attn_weights = self.attn(x_seq, x_seq, x_seq)
        else:
            fused = x_seq
            attn_weights = None

        fused_mean = fused.mean(dim=1)
        fused_norm = self.norm(fused_mean)
        fused_avg = fused_norm.reshape(
            batch_size, time_steps, channels, self.embed_dim
        )
        return fused_avg, attn_weights


class ChannelAttGatedGRUCell(nn.Module):
    """GRU-like recurrent cell with channel self-attention inside each gate."""

    def __init__(self, input_dim, hidden_dim, qk_dim, attn_dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.q_proj_r = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.k_proj_r = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.v_proj_r = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.q_proj_z = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.k_proj_z = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.v_proj_z = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.q_proj_n = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.k_proj_n = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.v_proj_n = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [
            self.q_proj_r,
            self.k_proj_r,
            self.q_proj_z,
            self.k_proj_z,
            self.q_proj_n,
            self.k_proj_n,
        ]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.v_proj_r.weight)
        nn.init.orthogonal_(self.v_proj_z.weight)
        nn.init.orthogonal_(self.v_proj_n.weight)

        if self.v_proj_r.bias is not None:
            nn.init.zeros_(self.v_proj_r.bias)
        if self.v_proj_z.bias is not None:
            nn.init.zeros_(self.v_proj_z.bias)
        if self.v_proj_n.bias is not None:
            nn.init.zeros_(self.v_proj_n.bias)

    def self_attention(self, x, q_proj, k_proj, v_proj):
        """Apply self-attention across EEG channels."""
        query = q_proj(x)
        key = k_proj(x)
        value = v_proj(x)

        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(self, x, h):
        """
        Args:
            x: Current input, shape (B, C, input_dim).
            h: Previous hidden state, shape (B, C, hidden_dim).

        Returns:
            Updated hidden state with shape (B, C, hidden_dim).
        """
        combined = torch.cat([x, h], dim=2)

        r, _ = self.self_attention(
            combined, self.q_proj_r, self.k_proj_r, self.v_proj_r
        )
        z, _ = self.self_attention(
            combined, self.q_proj_z, self.k_proj_z, self.v_proj_z
        )

        reset = torch.sigmoid(r)
        update = torch.sigmoid(z)

        combined_n = torch.cat([x, h * reset], dim=2)
        n, _ = self.self_attention(
            combined_n, self.q_proj_n, self.k_proj_n, self.v_proj_n
        )
        candidate = torch.tanh(n)

        new_state = (1 - update) * h + update * candidate
        new_state = self.norm(new_state)
        return new_state


class ChannelAttGatedGRU(nn.Module):
    """Stacked channel-attention recurrent module."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        qk_dim,
        num_layers,
        batch_first=True,
        attn_dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            [
                ChannelAttGatedGRUCell(
                    input_dim,
                    hidden_dim,
                    qk_dim,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, h_prev=None):
        """
        Args:
            x: Tensor of shape (B, T, C, input_dim) when batch_first=True.
            h_prev: Optional hidden state of shape
                (num_layers, B, C, hidden_dim).

        Returns:
            Tensor of shape (B, T, C, hidden_dim) when batch_first=True.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, channels, _ = x.size()

        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers,
                batch_size,
                channels,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )

        layer_output_list = []
        cur_layer_output = x

        for layer in range(self.num_layers):
            h_layer = []
            h = h_prev[layer]
            for t in range(seq_len):
                h = self.cells[layer](cur_layer_output[t], h)
                h_layer.append(h)

            layer_output = torch.stack(h_layer, dim=0)
            cur_layer_output = layer_output
            layer_output_list.append(layer_output)

        output = layer_output_list[-1]
        if self.batch_first:
            output = output.transpose(0, 1)
        return output


class MainModel(nn.Module):
    """MRPA-STFT + temporal branch + CA-GRU seizure classifier."""

    def __init__(self, Freq=192, Time=256, num_classes=2, config=None):
        super().__init__()
        self.config = config or {}
        dropout_p = self.config.get("DROPOUT", 0.5)

        use_standard_gru = bool(
            self.config.get("ABLATION_STANDARD_GRU", False)
        )
        use_modal_attn = not bool(
            self.config.get("ABLATION_NO_MODAL_ATTN", False)
        )

        # Each frequency-domain modality contains an equal number of bins.
        num_modalities = Freq // 64
        if num_modalities < 1:
            num_modalities = 1
        embed_dim = Freq // num_modalities

        self.multi_fusion = MultiFreqModalFusion(
            num_modalities=num_modalities,
            embed_dim=embed_dim,
            use_attn=use_modal_attn,
        )

        self.freq_branch = FrequencyBranch(in_freq=embed_dim, out_freq=64)
        self.time_branch = TimeBranch(in_time=Time, out_time=64)

        if use_standard_gru:
            self.cagru = nn.GRU(
                input_size=128,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
            )
            self._use_standard_gru = True
        else:
            self.cagru = ChannelAttGatedGRU(
                input_dim=128,
                hidden_dim=64,
                qk_dim=64,
                num_layers=1,
                batch_first=True,
                attn_dropout=float(self.config.get("ATTN_DROPOUT", 0.1)),
            )
            self._use_standard_gru = False

        time_attn_hidden = int(self.config.get("TIME_ATTN_HIDDEN", 32))
        self.time_attn = nn.Sequential(
            nn.Linear(64, time_attn_hidden),
            nn.Tanh(),
            nn.Dropout(float(self.config.get("TIME_ATTN_DROPOUT", 0.1))),
            nn.Linear(time_attn_hidden, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, return_features=False, feature_level="final"):
        """
        Args:
            x: Pre-extracted model input of shape (B, T, C, Freq + Time).
                With the default paper configuration, Freq=192 and Time=256.
                The frequency portion consists of three 64-bin modalities:
                long-window magnitude, long-window phase, and short-window
                magnitude.
            return_features: If True, also return an intermediate feature.
            feature_level: One of {'freq', 'time', 'fusion', 'tgcn', 'final'}.

        Returns:
            logits of shape (B, num_classes), optionally accompanied by the
            selected intermediate representation.
        """
        freq_dim = self.config.get("Freq", 192) if self.config else 192

        multi_modal = x[:, :, :, :freq_dim]
        time_raw = x[:, :, :, freq_dim:]

        fused_freq, _ = self.multi_fusion(multi_modal)
        freq_out = self.freq_branch(fused_freq)
        time_out = self.time_branch(time_raw)

        x_feat = torch.cat([freq_out, time_out], dim=3)

        if self._use_standard_gru:
            batch_size, time_steps, channels, feature_dim = x_feat.shape
            x_feat_flat = x_feat.reshape(
                batch_size * time_steps, channels, feature_dim
            )
            x_tgcn_flat, _ = self.cagru(x_feat_flat)
            x_tgcn = x_tgcn_flat.reshape(
                batch_size, time_steps, channels, 64
            )
        else:
            x_tgcn = self.cagru(x_feat)

        time_pool = str(self.config.get("TIME_POOL", "attn")).lower()
        if time_pool == "last":
            x_time_avg = x_tgcn[:, -1, :, :].mean(dim=1)
        elif time_pool == "mean":
            x_time_avg = x_tgcn.mean(dim=1).mean(dim=1)
        else:
            x_tc = x_tgcn.mean(dim=2)
            alpha = torch.softmax(self.time_attn(x_tc), dim=1)
            x_time_avg = (alpha * x_tc).sum(dim=1)

        logits = self.classifier(x_time_avg)

        if not return_features:
            return logits

        if feature_level == "freq":
            return logits, freq_out.mean(dim=1).mean(dim=1)
        if feature_level == "time":
            return logits, time_out.mean(dim=1).mean(dim=1)
        if feature_level == "fusion":
            return logits, x_feat.mean(dim=1).mean(dim=1)
        if feature_level == "tgcn":
            return logits, x_tgcn.mean(dim=1).mean(dim=1)
        if feature_level == "final":
            return logits, x_time_avg

        raise ValueError(
            "feature_level must be one of "
            "['freq', 'time', 'fusion', 'tgcn', 'final']"
        )
