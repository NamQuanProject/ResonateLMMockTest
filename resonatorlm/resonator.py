"""ResonatorLM mixer (report Sec. 2).

Mechanism (report, Sec. 2):
    k_c[tau] = exp(-a_c tau) cos(w_c tau + phi_c)                 (eq. 4)
    y_t      = sum_j k[t-j] u_j          -> causal FFT conv       (eq. 5, 10)
    s_t      = lam s_{t-1} + u_t, lam = exp(-a + i w)             (eq. 6)
    y_t      = Re(exp(i phi) s_t)                                 (eq. 7)
    z_t      = sigmoid(g_t) * coupled(y_t) + v * local_t          (eq. 8, 13)

Adjustments (all switchable through ResonatorConfig, see README):
    modes="channel"  one (a, w, phi) per channel instead of per head
    norm_gain        drive scaled by (1 - e^{-a}) so the kernel has ~unit DC gain
    input_gate       content-dependent write gate on the drive (kernel stays LTI)
    coupling / local can be switched off (report ablation: no measurable benefit)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResonatorConfig:
    n_embd: int = 768
    n_head: int = 12
    modes: str = "channel"          # "head" (report) | "channel"
    norm_gain: bool = True
    input_gate: bool = True
    coupling: bool = False
    local: bool = False
    local_kernel: int = 4
    min_half_life: float = 2.0      # report: 2-2048 tokens
    max_half_life: float = 2048.0
    attn_layers: list[int] = field(default_factory=list)  # layers that keep pretrained attention

    @classmethod
    def variant(cls, name: str, **kw) -> "ResonatorConfig":
        presets = {
            # Mixer as described in the report.
            "report": dict(modes="head", norm_gain=False, input_gate=False, coupling=True, local=True),
            # Report mixer + adjustments motivated by its results and limitations.
            "adjusted": dict(modes="channel", norm_gain=True, input_gate=True, coupling=False, local=False),
            # Adjusted mixer, a few layers keep pretrained attention for content-based retrieval.
            "hybrid": dict(modes="channel", norm_gain=True, input_gate=True, coupling=False, local=False,
                           attn_layers=[5, 10, 15, 20]),
        }
        return cls(**{**presets[name], **kw})


class ResonatorMixer(nn.Module):
    def __init__(self, cfg: ResonatorConfig):
        super().__init__()
        self.cfg = cfg
        D, H = cfg.n_embd, cfg.n_head
        self.H, self.dh = H, D // H
        n_modes = H if cfg.modes == "head" else D

        self._build_projections(D)

        # Half-lives log-spaced over [min, max]; a = ln2 / half_life (eq. 14).
        hl = torch.logspace(math.log10(cfg.min_half_life), math.log10(cfg.max_half_life), n_modes)
        hl = hl[torch.randperm(n_modes)] if n_modes == D else hl
        self.log_a = nn.Parameter(torch.log(math.log(2) / hl))
        # Half the modes start as pure decays (w=0), the rest oscillate with log-uniform periods.
        period = torch.exp(torch.empty(n_modes).uniform_(math.log(4), math.log(2048)))
        w = 2 * math.pi / period
        w[torch.rand(n_modes) < 0.5] = 0.0
        self.w = nn.Parameter(w)
        self.phi = nn.Parameter(torch.zeros(n_modes))

        if cfg.coupling:
            self.C = nn.Parameter(torch.eye(H))
        if cfg.local:
            self.local_conv = nn.Conv1d(D, D, cfg.local_kernel, groups=D, padding=cfg.local_kernel - 1)
            self.v = nn.Parameter(torch.zeros(D))

    def _build_projections(self, D):
        # Drive u and output gate g from one projection; optional input gate as a third slice.
        self.in_proj = nn.Linear(D, 3 * D if self.cfg.input_gate else 2 * D)
        self.out_proj = nn.Linear(D, D)

    # -- per-channel dynamics -------------------------------------------------
    def _params(self):
        a, w, phi = self.log_a.exp(), self.w, self.phi
        if self.cfg.modes == "head":
            a, w, phi = (p.repeat_interleave(self.dh) for p in (a, w, phi))
        b = -torch.expm1(-a) if self.cfg.norm_gain else torch.ones_like(a)
        return a, w, phi, b

    def kernel(self, T: int, device=None) -> torch.Tensor:
        a, w, phi, b = self._params()
        tau = torch.arange(T, device=device or a.device, dtype=a.dtype)
        return b[:, None] * torch.exp(-a[:, None] * tau) * torch.cos(w[:, None] * tau + phi[:, None])  # (D, T)

    def _drive(self, x):
        parts = self.in_proj(x).chunk(3 if self.cfg.input_gate else 2, dim=-1)
        u, g = parts[0], parts[1]
        if self.cfg.input_gate:
            u = u * torch.sigmoid(parts[2])
        return u, g

    def _finish(self, y, g, local):
        if self.cfg.coupling:  # eq. 13: mix heads at the same position
            B, T, D = y.shape
            y = torch.einsum("gh,bthd->btgd", self.C, y.view(B, T, self.H, self.dh)).reshape(B, T, D)
        z = torch.sigmoid(g) * y
        if self.cfg.local:
            z = z + self.v * local
        return self.out_proj(z)

    # -- parallel path (training / prefill) -----------------------------------
    def forward(self, x, return_state: bool = False):
        T = x.shape[1]
        u, g = self._drive(x)
        k = self.kernel(T, x.device)
        n = 2 * T  # zero padding -> linear (causal) rather than circular convolution
        y = torch.fft.irfft(torch.fft.rfft(u.to(k.dtype).transpose(1, 2), n=n) * torch.fft.rfft(k, n=n), n=n)[..., :T]
        y = y.transpose(1, 2).to(x.dtype)
        local = self._local_parallel(x) if self.cfg.local else None
        out = self._finish(y, g, local)
        if not return_state:
            return out
        return out, self._prefill_state(u, x)

    def _local_parallel(self, x):
        T = x.shape[1]
        return self.local_conv(x.transpose(1, 2))[..., :T].transpose(1, 2)

    def _prefill_state(self, u, x):
        """Terminal complex state s_T = sum_j lam^{T-1-j} b u_j, kept as (re, im)."""
        a, w, _, b = self._params()
        T = u.shape[1]
        tau = torch.arange(T - 1, -1, -1, device=u.device, dtype=a.dtype)
        decay = torch.exp(-a[None] * tau[:, None])                     # (T, D)
        re = torch.einsum("btd,td->bd", u.to(a.dtype), decay * torch.cos(w[None] * tau[:, None])) * b
        im = torch.einsum("btd,td->bd", u.to(a.dtype), decay * torch.sin(w[None] * tau[:, None])) * b
        state = {"re": re, "im": im}
        if self.cfg.local:
            K = self.cfg.local_kernel
            state["buf"] = F.pad(x, (0, 0, max(0, K - 1 - x.shape[1]), 0))[:, -(K - 1):]
        return state

    # -- recurrent path (decoding) --------------------------------------------
    def step(self, x, state):
        """x: (B, 1, D). Fixed-size state update, eq. 6-7."""
        a, w, phi, b = self._params()
        u, g = self._drive(x)
        u = u[:, 0].to(a.dtype)
        if state is None:
            state = {"re": torch.zeros_like(u), "im": torch.zeros_like(u)}
            if self.cfg.local:
                state["buf"] = x.new_zeros(x.shape[0], self.cfg.local_kernel - 1, x.shape[2])
        r = torch.exp(-a)
        c, s = torch.cos(w), torch.sin(w)
        re = r * (c * state["re"] - s * state["im"]) + b * u
        im = r * (s * state["re"] + c * state["im"])
        y = (torch.cos(phi) * re - torch.sin(phi) * im)[:, None].to(x.dtype)
        new = {"re": re, "im": im}
        local = None
        if self.cfg.local:
            win = torch.cat([state["buf"], x], dim=1)                   # (B, K, D)
            wk = self.local_conv.weight[:, 0].T                        # (K, D)
            local = ((win * wk).sum(1) + self.local_conv.bias)[:, None]
            new["buf"] = win[:, 1:]
        return self._finish(y, g, local), new

    def state_bytes(self, batch: int = 1) -> int:
        n = 2 * self.cfg.n_embd * 4 * batch
        if self.cfg.local:
            n += (self.cfg.local_kernel - 1) * self.cfg.n_embd * 4 * batch
        return n
