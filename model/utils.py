from typing import List, Tuple
import torch
from torch import LongTensor


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"



def to_tgt_output(
    tokens: List[List[int]], direction: str, device: torch.device, SOS_IDX: int = 1, EOS_IDX: int = 2, PAD_IDX: int = 0
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
    if direction == "l2r":
        tokens = tokens
        start_w = SOS_IDX
        stop_w = EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = EOS_IDX
        stop_w = SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]
    tgt = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device, SOS_IDX: int = 1, EOS_IDX: int = 2, PAD_IDX: int = 0
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device, SOS_IDX, EOS_IDX, PAD_IDX)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device, SOS_IDX, EOS_IDX, PAD_IDX)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out