"""Swap ResonatorLM mixers into a pretrained Llama-family model (TinyLlama-1.1B).

Each converted layer's self_attn becomes a SwitchAttention holding both the original
(frozen) attention and a ResonatorAttention, so one forward pass can act as teacher,
student, or layer-wise distillation (teacher output passed on, student loss recorded).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from resonator import ResonatorConfig, ResonatorMixer


class LowRankGate(nn.Module):
    """x -> up(down(x)) + bias; up starts at zero so the gate starts content-independent."""

    def __init__(self, D, rank, bias):
        super().__init__()
        self.down = nn.Linear(D, rank, bias=False)
        self.up = nn.Linear(rank, D)
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.up.bias, bias)

    def forward(self, x):
        return self.up(self.down(x))


class ResonatorAttention(ResonatorMixer):
    """Resonator mixer whose drive/output projections come from a GQA attention layer."""

    def __init__(self, cfg: ResonatorConfig, attn, gate_rank: int = 64):
        self._src = (attn, gate_rank)   # tuple: plain attribute, usable before Module.__init__
        super().__init__(cfg)
        del self._src

    def _build_projections(self, D):
        attn, rank = self._src
        self.n_rep = D // attn.v_proj.out_features
        self.v_proj = nn.Linear(D, attn.v_proj.out_features, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            self.v_proj.weight.copy_(attn.v_proj.weight.float())
            self.out_proj.weight.copy_(attn.o_proj.weight.float())
        self.g_proj = LowRankGate(D, rank, 3.0)
        if self.cfg.input_gate:
            self.i_proj = LowRankGate(D, rank, 3.0)

    def _drive(self, x):
        B, T, _ = x.shape
        v = self.v_proj(x).view(B, T, -1, self.dh)
        u = v.repeat_interleave(self.n_rep, dim=2).flatten(2)   # GQA: kv head j serves heads j*n_rep..
        if self.cfg.input_gate:
            u = u * torch.sigmoid(self.i_proj(x))
        return u, self.g_proj(x)


class SwitchAttention(nn.Module):
    def __init__(self, attn, cfg: ResonatorConfig):
        super().__init__()
        self.teacher = attn
        self.student = ResonatorAttention(cfg, attn)
        self.mode = "teacher"
        self.loss = None
        self.state = None

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kw):
        if self.mode == "teacher":
            return self.teacher(hidden_states, position_embeddings, attention_mask, past_key_values, **kw)
        x = hidden_states.to(self.student.log_a.dtype)
        if self.mode == "distill":
            with torch.no_grad():
                ref = self.teacher(hidden_states, position_embeddings, attention_mask, past_key_values, **kw)[0]
            out = self.student(x.detach())
            r = ref.to(out.dtype)
            self.loss = ((out - r) ** 2).mean() / (r.pow(2).mean() + 1e-6)   # relative MSE
            return ref, None
        # student: parallel path for prompts, recurrent step for single tokens after a prefill
        if x.shape[1] == 1 and self.state is not None:
            out, self.state = self.student.step(x, self.state)
        elif kw.get("use_cache") or past_key_values is not None:
            out, self.state = self.student(x, return_state=True)
        else:
            out, self.state = self.student(x), None
        return out.to(hidden_states.dtype), None


def convert(model, cfg: ResonatorConfig, layers=None):
    """Replace self_attn in `layers` (default: all except cfg.attn_layers) with SwitchAttention."""
    c = model.config
    cfg.n_embd, cfg.n_head = c.hidden_size, c.num_attention_heads
    n = c.num_hidden_layers
    layers = [i for i in range(n) if i not in cfg.attn_layers] if layers is None else layers
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device
    for i in layers:
        layer = model.model.layers[i]
        sw = SwitchAttention(layer.self_attn, cfg)
        sw.student.to(dev, torch.float32)
        layer.self_attn = sw
    return switches(model)


def switches(model):
    return [m for m in model.modules() if isinstance(m, SwitchAttention)]


def set_mode(model, mode):
    for s in switches(model):
        s.mode, s.state, s.loss = mode, None, None


def student_parameters(model):
    return [p for s in switches(model) for p in s.student.parameters()]


def student_state_dict(model):
    return {f"{i}": s.student.state_dict() for i, s in enumerate(switches(model))}


def load_student_state_dict(model, sd):
    for i, s in enumerate(switches(model)):
        s.student.load_state_dict(sd[f"{i}"])
