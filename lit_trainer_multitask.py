import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from typing import List

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
        # đối với head radical: kích thước = số radicals
        rad_vocab_size: int,
        # **danh sách thực tế các radicals** (từ single.txt)
        radicals: List[str],
        learning_rate: float,
        patience: int,
        rad_loss_weight: float = 1.0,
    ):
        super().__init__()
        # lưu hyperparams (trừ list vocab và radicals để avoid pickle errors)
        self.save_hyperparameters(ignore=['vocab','radicals'])
        self.vocab = vocab

        # giữ list radicals & map sang idx
        self.radicals = radicals
        self.rad2idx  = {rad: i for i, rad in enumerate(radicals)}

        # instantiate backbone
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
        self.seq_loss_fn = lambda logit, tgt: F.cross_entropy(
            logit.view(-1, logit.size(-1)),
            tgt.view(-1),
            ignore_index=self.pad_idx
        )
        self.rad_loss_fn = BCEWithLogitsLoss()
        self.rad_loss_weight = rad_loss_weight

    def build_rad_targets(self, indices: List[List[int]]) -> torch.FloatTensor:
        """
        Chuyển List token‐ID -> multi‐hot trên chỉ các radicals.
        """
        b = len(indices)
        R = self.hparams.rad_vocab_size
        rad_t = torch.zeros(b, R, device=self.device)
        for i, seq in enumerate(indices):
            for tok in seq:
                ch = self.vocab.id2char[tok]            # chuyển ID->ký tự
                if ch in self.rad2idx:                 # chỉ những ký tự nằm trong radicals
                    rad_t[i, self.rad2idx[ch]] = 1.0
        return rad_t

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        seq_logits, rad_logits = self.model(batch.imgs, batch.mask, tgt)

        seq_loss = self.seq_loss_fn(seq_logits, out)
        rad_targets = self.build_rad_targets(batch.indices)
        rad_loss    = self.rad_loss_fn(rad_logits, rad_targets)
        loss = seq_loss + self.rad_loss_weight * rad_loss

        bs = len(batch.indices)
        self.log('train_loss',     loss,      on_step=True, on_epoch=True, batch_size=bs)
        self.log('train_seq_loss', seq_loss,  on_step=True, on_epoch=True, batch_size=bs)
        self.log('train_rad_loss', rad_loss,  on_step=True, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        seq_logits, rad_logits = self.model(batch.imgs, batch.mask, tgt)

        seq_loss = self.seq_loss_fn(seq_logits, out)
        rad_targets = self.build_rad_targets(batch.indices)
        rad_loss    = self.rad_loss_fn(rad_logits, rad_targets)
        loss = seq_loss + self.rad_loss_weight * rad_loss

        bs = len(batch.indices)
        self.log('val_loss',     loss,      prog_bar=True, batch_size=bs)
        self.log('val_seq_loss', seq_loss,  prog_bar=False, batch_size=bs)
        self.log('val_rad_loss', rad_loss,  prog_bar=False, batch_size=bs)

        # exact‐match trên sequence
        mask = out != self.pad_idx
        err  = (seq_logits.argmax(-1) != out).masked_fill(~mask, 0)
        cer  = err.view(2,bs,-1).permute(1,0,2).flatten(1,2).sum(-1) / \
               mask.view(2,bs,-1).permute(1,0,2).flatten(1,2).sum(-1)
        seq_acc = (cer == 0).float().mean()
        self.log('val_seq_ExpRate', seq_acc, prog_bar=True, batch_size=bs)

        # exact‐match trên radicals
        rad_pred = (torch.sigmoid(rad_logits) > 0.5).float()
        rad_acc  = (rad_pred == rad_targets).all(dim=1).float().mean()
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
