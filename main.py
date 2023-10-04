from TheSonnetRNN import TheSonnetsRNN
from Trainer import Trainer
from shakespeare_module import ShakespeareDataSet
VOCAB_SIZE = 30_000
SEQUENCE_LEN = 20

if __name__ == '__main__':
    dl_train = ShakespeareDataSet('train', vocab_size=VOCAB_SIZE)
    dl_val = ShakespeareDataSet('val', vocab_size=VOCAB_SIZE)
    model = TheSonnetsRNN(
        n_units_h1=256, n_units_h2=512, 
        vocab_size=VOCAB_SIZE, seq_len=SEQUENCE_LEN
        )
    trainer = Trainer(model, dl_train, dl_val)
    trainer.train(2)