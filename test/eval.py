from nom_ids_ocr.data import ImageDataModule, SeqVocab
from nom_ids_ocr.lit_trainer import LitBTTR

base_vocab = open('vocab_ids.txt', 'r').read().split('\n')
ids_dict = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in open('ids_exp.txt', 'r').readlines()}


dm = ImageDataModule(
    data_dir='../datasets/test_demo',
    vocab=SeqVocab(base_vocab, ids_dict),  # Replace with your vocabulary
    batch_size=1,
    num_workers=1
)

model = LitBTTR.load_from_checkpoint(checkpoint_path='../lightning_logs/h302snoy/checkpoints/epoch=199-step=19048-val_ExpRate=0.9508.ckpt',
    d_model=256, growth_rate=24, num_layers=16, nhead=8, num_decoder_layers=3, dim_feedforward=1024, dropout=0.3, beam_size=10, max_len=200, alpha=1.0, learning_rate=1.0, patience=20, vocab_size=len(dm.vocab), SOS_IDX=1, EOS_IDX=2, PAD_IDX=0)

model.eval()