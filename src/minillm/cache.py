"""A batched KV cache with explicit slot allocation.

The cache in `model.py` lives inside each attention module and holds exactly
one sequence. That is fine for `model.generate`, and useless for serving:
requests arrive and finish at different times, so the engine needs to keep
several sequences resident at once, add and drop them independently, and know
which memory is free.

This is the pre-paging design. Memory is a single dense tensor of shape
(n_layers, max_slots, n_heads, max_seq, d_head) -- one fixed-size row per slot,
reserved for the maximum sequence length whether the request needs it or not.
A 20-token request holds the same memory as a 256-token one.

That waste is the motivation for stage 4. Measure it with `fragmentation()`
before replacing it: the paged version only ever allocates blocks a sequence
actually uses, and the difference between these two numbers is the entire
argument for paging.
"""

from __future__ import annotations

import torch

from .config import ModelConfig


class BatchedKVCache:
    def __init__(
        self,
        cfg: ModelConfig,
        max_slots: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.cfg = cfg
        self.max_slots = max_slots
        self.max_seq = cfg.context_len
        self.device = device

        shape = (cfg.n_layers, max_slots, cfg.n_heads, self.max_seq, cfg.d_head)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

        # lengths[s] = number of tokens currently stored in slot s.
        self.lengths = torch.zeros(max_slots, dtype=torch.long, device=device)
        self._free: list[int] = list(range(max_slots))

    # ------------------------------------------------------------ allocation

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        """Reserve a slot. Raises if the cache is full."""
        if not self._free:
            raise RuntimeError("KV cache is full")
        slot = self._free.pop()
        self.lengths[slot] = 0
        return slot

    def free(self, slot: int) -> None:
        self.lengths[slot] = 0
        self._free.append(slot)

    def bytes_allocated(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def fragmentation(self) -> float:
        """Fraction of reserved cache memory holding no live token.

        Counts both unused slots and the unused tail of every live slot. This
        is the number a paged cache is designed to drive down.
        """
        used = int(self.lengths.sum())
        total = self.max_slots * self.max_seq
        return 1.0 - used / total

    # ----------------------------------------------------------------- write

    def append(
        self,
        layer_idx: int,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write keys/values into `slots` at `positions`, return the full cache rows.

        slots:     (B,)      slot index per sequence in the batch
        keys/vals: (B, H, T, D)
        positions: (B, T)    absolute position of each token

        Returns k, v of shape (B, H, T_k, D) where T_k covers every position
        written so far for these slots.
        """
        B, H, T, D = keys.shape
        # Advanced indexing: slots (B,1) broadcasts against positions (B,T),
        # the slice selects all heads. Target view is (B, T, H, D), so the
        # incoming (B, H, T, D) is transposed to match.
        self.k[layer_idx, slots.view(B, 1), :, positions] = keys.transpose(1, 2)
        self.v[layer_idx, slots.view(B, 1), :, positions] = values.transpose(1, 2)

        if layer_idx == self.cfg.n_layers - 1:
            # Only advance lengths once per token, not once per layer.
            self.lengths[slots] = positions[:, -1] + 1

        t_k = int(positions.max()) + 1
        return (
            self.k[layer_idx, slots, :, :t_k],
            self.v[layer_idx, slots, :, :t_k],
        )
