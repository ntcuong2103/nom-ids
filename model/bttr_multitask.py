import torch
import pytorch_lightning as pl
from torch import FloatTensor, LongTensor
from torch.nn import Linear
from .encoder import Encoder
from .decoder import Decoder

class BTTRMultiTask(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        vocab_size: int,
        pad_idx: int,
        # now we detect the same tokens as we generate
        rad_vocab_size: int,
    ):
        super().__init__()
        self.encoder = Encoder(d_model, growth_rate, num_layers)
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
        )

        # bag-of-tokens head
        self.rad_classifier = Linear(d_model, rad_vocab_size)

    def forward(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        tgt: LongTensor
    ):
        feat, mask = self.encoder(img, img_mask)           # [b, t, d], [b, t]

        # → snowball pooling for bag-of-tokens
        pooled = feat.mean(dim=1)                          # [b, d]
        rad_logits = self.rad_classifier(pooled)           # [b, rad_vocab_size]

        # → bidirectional sequence head
        feat2 = torch.cat((feat, feat), dim=0)             # [2b, t, d]
        mask2 = torch.cat((mask, mask), dim=0)             # [2b, t]
        seq_logits = self.decoder(feat2, mask2, tgt)       # [2b, L, vocab_size]

        return seq_logits, rad_logits

    def beam_search(self, img, img_mask, beam_size, max_len):
        feat, mask = self.encoder(img, img_mask)
        return self.decoder.beam_search(feat, mask, beam_size, max_len)