import wandb
from data import ImageDataModule, SeqVocab
from lit_trainer_multitask import LitBTTRMultiTask
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# 1) Đọc vocab sequence như trước
base_vocab = open('vocab_ids.txt', encoding='utf-8').read().splitlines()
ids_dict  = {c: ids for c, ids in [l.split('\t') for l in open('ids_exp.txt', encoding='utf-8').read().splitlines()]}

# 2) Đọc danh sách radicals (chỉ các bộ) từ single.txt
radicals = [r.strip() for r in open('single.txt', encoding='utf-8') if r.strip()]

# 3) Tạo SeqVocab seq‐generation
vocab = SeqVocab(base_vocab, ids_dict)

# 4) DataModule
dm = ImageDataModule(
    data_dir='datasets/tkh-mth2k2/MTH1000',
    vocab=vocab,
    batch_size=32,
    num_workers=8
)

wandb.init(
    project="nom-ids-train",
    name="BTTR-MultiTask",
)

# 5) Khởi tạo model, truyền cả rad_vocab_size và list radicals
model = LitBTTRMultiTask(
    vocab=vocab,
    d_model=256,
    growth_rate=24,
    num_layers=16,
    nhead=8,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.3,
    vocab_size=len(vocab),
    pad_idx=vocab.PAD_IDX,

    # **Chỉ tính trên các bộ**  
    rad_vocab_size=len(radicals),
    radicals=radicals,

    learning_rate=1.0,
    patience=20,
    rad_loss_weight=0.5,
)

trainer = Trainer(
    callbacks=[
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            filename='{epoch}-{val_seq_ExpRate:.4f}-{val_rad_acc:.4f}',
            save_top_k=5,
            monitor='val_seq_ExpRate',
            mode='max'
        ),
        EarlyStopping(monitor='val_seq_ExpRate', patience=10, mode='max'),
    ],
    check_val_every_n_epoch=1,
    max_epochs=200,
    accelerator='gpu',
    devices=1,
    logger=WandbLogger(),
)

trainer.fit(model, dm)
