# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ACEStep
"""Three musical slots encoded on the VAE's 25 Hz timeline."""

import math
import re

import torch
from torch import nn

from sglang_omni.models.mula_edit.protocol import CHORD_QUALITIES, MulaEditRequest

FRAMES_PER_SECOND = 25
PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
BASS_DEGREES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 9: 2, 11: 5, 13: 9}
CHORD_PATTERN = re.compile(r"^([A-G])([#b]*)(?::([^/]+))?(?:/([#b]*)(\d+))?$")
PITCH_HARMONICS = (1, 5, 7, 11, 2, 10, 3, 9, 4, 8, 6)


def parse_chord(label: str) -> tuple[int, int, int] | None:
    """Return root pitch class, quality code, and bass pitch class."""
    match = CHORD_PATTERN.fullmatch(label.strip())
    if match is None:
        return None
    else:
        letter, accidentals, quality, bass_accidentals, degree = match.groups()
    root = (
        PITCH_CLASSES[letter] + accidentals.count("#") - accidentals.count("b")
    ) % 12
    if degree is None:
        bass = root
    elif int(degree) not in BASS_DEGREES:
        return None
    else:
        bass = (
            root
            + BASS_DEGREES[int(degree)]
            + bass_accidentals.count("#")
            - bass_accidentals.count("b")
        ) % 12
    quality = quality or "maj"
    if quality not in CHORD_QUALITIES:
        quality = "other"
    else:
        pass
    return root, CHORD_QUALITIES.index(quality) + 1, bass


def score_events(
    request: MulaEditRequest, frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (frame, slot, pitch, onset, auxiliary) rows and the provided mask."""
    slots = (
        request.symbols.vocal_melody,
        request.symbols.instrumental_melody,
        request.symbols.chord_progression,
    )
    provided = torch.zeros(1, frames, 3, dtype=torch.bool)
    rows: list[list[int]] = []
    for slot, events in enumerate(slots):
        if events is None:
            continue
        else:
            provided[:, :, slot] = True
        for start_seconds, end_seconds, value in events:
            first = math.floor(start_seconds * FRAMES_PER_SECOND)
            last = min(
                frames, max(first + 1, math.ceil(end_seconds * FRAMES_PER_SECOND))
            )
            if slot == 2:
                assert isinstance(value, str)
                chord = parse_chord(value)
                if chord is None:
                    continue
                else:
                    root, quality, bass = chord
                    notes = ((root, quality), (bass, len(CHORD_QUALITIES) + 1))
            else:
                assert isinstance(value, int)
                notes = ((value, 0),)
            for pitch, auxiliary in notes:
                rows.extend(
                    [frame, slot, pitch, int(frame == first), auxiliary]
                    for frame in range(first, last)
                )
    if request.symbolic_scope == "span":
        first = int(request.start_seconds * FRAMES_PER_SECOND)
        last = max(first + 1, int(request.end_seconds * FRAMES_PER_SECOND))
        provided[:, :first] = False
        provided[:, last:] = False
    else:
        pass
    return torch.tensor(rows, dtype=torch.long).reshape(-1, 5), provided


class SymbolicSlotEncoder(nn.Module):
    """Rotary note embeddings pooled independently for each time frame and slot."""

    def __init__(self, slot_dimension: int, epsilon: float) -> None:
        super().__init__()
        self.slot_dimension = slot_dimension
        self.epsilon = epsilon
        self.state_emb = nn.Parameter(torch.empty(3, 2, slot_dimension))
        self.aux_emb = nn.Parameter(torch.empty(18, 2, slot_dimension))
        self.norm_weight = nn.Parameter(torch.empty(3, slot_dimension))
        self.null_emb = nn.Parameter(torch.empty(3, slot_dimension + 1))

    def forward(self, events: torch.Tensor, provided: torch.Tensor) -> torch.Tensor:
        frames = provided.shape[1]
        frame, slot, pitch, onset, auxiliary = events.unbind(dim=1)
        base = self.state_emb[slot, onset].float()
        base = base + self.aux_emb[auxiliary, onset].float() * (
            auxiliary > 0
        ).unsqueeze(-1)
        blocks = self.slot_dimension // 2
        harmonics = torch.tensor(
            [PITCH_HARMONICS[i % len(PITCH_HARMONICS)] for i in range(blocks)],
            device=events.device,
            dtype=torch.float32,
        )
        ratio = torch.arange(blocks, device=events.device, dtype=torch.float32) / max(
            blocks - 1, 1
        )
        octave_frequencies = (
            0.9 * math.pi * ((math.pi / 24) / (0.9 * math.pi)) ** ratio
        ).flip(0)
        angles = (pitch % 12).float()[:, None] * (harmonics * (2 * math.pi / 12)) + (
            pitch // 12
        ).float()[:, None] * octave_frequencies
        pairs = base.unflatten(-1, (-1, 2))
        cosine, sine = angles.cos(), angles.sin()
        embeddings = torch.stack(
            (
                pairs[..., 0] * cosine - pairs[..., 1] * sine,
                pairs[..., 0] * sine + pairs[..., 1] * cosine,
            ),
            dim=-1,
        ).flatten(-2)
        indices = frame * 3 + slot
        grid = (
            torch.zeros(frames * 3, self.slot_dimension, device=events.device)
            .index_add(0, indices, embeddings)
            .view(1, frames, 3, self.slot_dimension)
        )
        counts = torch.zeros(frames * 3, device=events.device).index_add(
            0, indices, torch.ones(len(events), device=events.device)
        )
        grid = (
            grid
            * torch.rsqrt(grid.square().mean(-1, keepdim=True) + self.epsilon)
            * self.norm_weight.float()
        )
        features = torch.cat((grid, counts.log1p().view(1, frames, 3, 1)), dim=-1)
        return (
            torch.where(provided.unsqueeze(-1), features, self.null_emb.float())
            .flatten(2)
            .to(self.state_emb.dtype)
        )
