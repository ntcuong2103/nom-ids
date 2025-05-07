from data import ImageDataModule, SeqVocab
from lit_trainer import LitBTTR
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import wandb
from pytorch_lightning.loggers import WandbLogger

base_vocab = open('vocab_ids.txt', 'r').read().split('\n')
ids_dict = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in open('ids_exp.txt', 'r').readlines()}

if __name__ == "__main__":
    wandb.init(project="nom-ids-train", name="BTTR", entity="ntcuong2103-vietnamese-german-university")

    dm = ImageDataModule(
        data_dir='datasets/tkh-mth2k2/MTH1000',
        vocab=SeqVocab(base_vocab, ids_dict),  # Replace with your vocabulary
        batch_size=32,
        num_workers=8
    )

    # dm.setup(stage='fit')
    # batch = next(iter(dm.train_dataloader()))
    # exit(0)
    
    model = LitBTTR(d_model=256, growth_rate=24, num_layers=16, nhead=8, num_decoder_layers=3, dim_feedforward=1024, dropout=0.3, beam_size=10, max_len=200, alpha=1.0, learning_rate=1.0, patience=20, vocab_size=len(dm.vocab), SOS_IDX=1, EOS_IDX=2, PAD_IDX=0)

    trainer = Trainer(
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(filename='{epoch}-{step}-{val_ExpRate:.4f}', save_top_k=5, monitor='val_ExpRate', mode='max'),
            EarlyStopping(monitor='val_ExpRate', patience=10, mode='max', verbose=True),
        ], 
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        deterministic=False, 
        max_epochs=200, 
        accelerator='gpu',
        devices=1,
        logger=WandbLogger(),
    )

    trainer.fit(model, dm, ckpt_path=None)
    wandb.finish()