import zipfile

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor
import torch.nn.functional as F

from model.bttr import BTTR
from typing import List, Tuple
from einops import rearrange
from data import Batch
from model.utils import to_bi_tgt_out

def ce_loss(
    output_hat: torch.Tensor, output: torch.Tensor, ignore_idx: int
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx)
    return loss



class LitBTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        # training
        learning_rate: float,
        patience: int,
        vocab_size: int,
        SOS_IDX: int = 1,
        EOS_IDX: int = 2,
        PAD_IDX: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bttr = BTTR(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            vocab_size=vocab_size,
            pad_idx=PAD_IDX,
        )

        self.PAD_IDX = PAD_IDX

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.bttr(img, img_mask, tgt)

    def beam_search(
        self,
        img: FloatTensor,
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
    ) -> List[int]:
        """for inference, one image at a time

        Parameters
        ----------
        img : FloatTensor
            [3, h, w]
        beam_size : int, optional
            by default 10
        max_len : int, optional
            by default 200
        alpha : float, optional
            by default 1.0

        Returns
        -------
        """
        assert img.dim() == 3
        img_mask = torch.zeros_like(img, dtype=torch.long)  # squeeze channel
        hyps = self.bttr.beam_search(img.unsqueeze(0), img_mask, beam_size, max_len)
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** alpha))
        return best_hyp.seq

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out, self.PAD_IDX)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=len(batch.indices))

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out, self.PAD_IDX)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch.indices)
        )

        # accuracy out_hat vs out
        mask = out != self.PAD_IDX
        cer = (out_hat.argmax(-1) != out).masked_fill(~mask, 0).reshape(2, len(batch.indices), -1).permute(1, 0, 2).flatten(1,2).sum(-1) / mask.reshape(2, len(batch.indices), -1).permute(1, 0, 2).flatten(1,2).sum(-1)
        seq_acc = (cer == 0.0).float().mean()
        self.log(
            "val_ExpRate",
            seq_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch.indices)
        )


    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
