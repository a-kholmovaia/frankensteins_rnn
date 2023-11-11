from FrankensteinsRNN import FrankensteinsRNN
from Trainer import Trainer
from FrankensteinDataset import FrankensteinDataSet
from LSTM import LSTM

if __name__ == '__main__':
    dataset_train = FrankensteinDataSet('train')
    dl_train = dataset_train.get_dataloader()
    dataset_val = FrankensteinDataSet('val')
    dl_val = dataset_val.get_dataloader()

    model_rnn = FrankensteinsRNN(
        n_units_h1=256, n_units_h2=512, 
        vocab_size=dataset_train.vocab_size, seq_len=dataset_train.PAD_SIZE
        )
    #trainer_rnn_custom = Trainer(
    #    model_rnn, dataset_train, dataset_val, 
    #    vocab_size=dataset_train.vocab_size,
    #    file_path='results_custom_rnn.csv')
    #print('Initialised Custom RNN')
    #trainer_rnn_custom.train(15)

    model_lstm = LSTM(
        embedding_dim=256, num_layers=4, 
        n_units=512, vocab_size=dataset_train.vocab_size,
        seq_len=dataset_train.PAD_SIZE).cuda()
    trainer_lstm = Trainer(
        model_lstm, dataset_train, dataset_val, 
        vocab_size= dataset_train.vocab_size,
        file_path='results_lstm_full.csv')
    print('Initialised LSTM')
    trainer_lstm.train(100)