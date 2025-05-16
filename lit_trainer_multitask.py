# lit_trainer_multitask.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss

from model.bttr_multitask import BTTRMultiTask
from model.utils import to_bi_tgt_out
from data import Batch, SeqVocab


class LitBTTRMultiTask(pl.LightningModule):
    def __init__(
        self,
        vocab: SeqVocab,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        vocab_size: int,
        pad_idx: int,
        # bag‐of‐token head size (same as vocab_size)
        rad_vocab_size: int,
        learning_rate: float,
        patience: int,
        rad_loss_weight: float = 1.0,
    ):
        super().__init__()
        # save_hyperparameters except for 'vocab' object
        self.save_hyperparameters(ignore=['vocab'])
        self.vocab = vocab

        # instantiate the multi‐task BTTR
        self.model = BTTRMultiTask(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            rad_vocab_size=rad_vocab_size,
        )

        self.pad_idx = pad_idx
        # loss fns
        self.seq_loss_fn = lambda logit, tgt: F.cross_entropy(
            logit.view(-1, logit.size(-1)),
            tgt.view(-1),
            ignore_index=self.pad_idx
        )
        self.rad_loss_fn = BCEWithLogitsLoss()
        self.rad_loss_weight = rad_loss_weight

    def build_rad_targets(self, indices):
        """
        Given List[List[int]] of IDS token‐IDs,
        returns a [b, rad_vocab_size] multi-hot tensor.
        """
        b = len(indices)
        R = self.hparams.rad_vocab_size
        rad_t = torch.zeros(b, R, device=self.device)
        for i, seq in enumerate(indices):
            for tok in seq:
                rad_t[i, tok] = 1.0
        return rad_t

    def training_step(self, batch: Batch, _):
        # prepare bidirectional sequence targets
        tgt, out = to_bi_tgt_out(batch.indices, self.device)

        # forward pass
        seq_logits, rad_logits = self.model(batch.imgs, batch.mask, tgt)

        # sequence CE loss
        seq_loss = self.seq_loss_fn(seq_logits, out)
        # radical bag‐of‐tokens BCE loss
        rad_targets = self.build_rad_targets(batch.indices)
        rad_loss = self.rad_loss_fn(rad_logits, rad_targets)

        loss = seq_loss + self.rad_loss_weight * rad_loss

        # log with explicit batch_size
        bs = len(batch.indices)
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=bs)
        self.log('train_seq_loss', seq_loss, on_step=True, on_epoch=True, batch_size=bs)
        self.log('train_rad_loss', rad_loss, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        seq_logits, rad_logits = self.model(batch.imgs, batch.mask, tgt)

        seq_loss = self.seq_loss_fn(seq_logits, out)
        rad_targets = self.build_rad_targets(batch.indices)
        rad_loss = self.rad_loss_fn(rad_logits, rad_targets)
        loss = seq_loss + self.rad_loss_weight * rad_loss

        bs = len(batch.indices)
        self.log('val_loss', loss, prog_bar=True, batch_size=bs)
        self.log('val_seq_loss', seq_loss, prog_bar=False, batch_size=bs)
        self.log('val_rad_loss', rad_loss, prog_bar=False, batch_size=bs)

        # sequence exact‐match accuracy
        mask = out != self.pad_idx
        err = (seq_logits.argmax(-1) != out).masked_fill(~mask, 0)
        cer = (
            err.view(2, bs, -1)
               .permute(1, 0, 2)
               .flatten(1, 2)
               .sum(-1)
            / mask.view(2, bs, -1)
                  .permute(1, 0, 2)
                  .flatten(1, 2)
                  .sum(-1)
        )
        seq_acc = (cer == 0).float().mean()
        self.log('val_seq_ExpRate', seq_acc, prog_bar=True, batch_size=bs)

        # radical‐token exact match accuracy
        rad_pred = (torch.sigmoid(rad_logits) > 0.5).float()
        rad_acc = (rad_pred == rad_targets).all(dim=1).float().mean()
        self.log('val_rad_acc', rad_acc, prog_bar=True, batch_size=bs)

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=self.hparams.patience
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_seq_ExpRate',
                'interval': 'epoch',
                'frequency': 1,
            },
        }
