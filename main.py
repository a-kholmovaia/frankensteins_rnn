import requests
from fastai.text.all import *
from TheSonnetRNN import TheSonnetsRNN
from Trainer import Trainer

SEQUENCE_LEN = 16
BATCH_SIZE = 64
VOCAB_SIZE = 250_000

def get_data(link: str):
    f = requests.get(link)
    lines = f.text
    return lines

def group_chunks(ds, bs):
    m = len(ds) // bs
    new_ds = L()
    for i in range(m):
        new_ds += L(ds[i + m*j] for j in range(bs))
        return new_ds

def prepare_data(lines):
    spacy = WordTokenizer()
    tkn = Tokenizer(spacy)
    toks = tkn(lines)
    num = Numericalize(max_vocab=VOCAB_SIZE)
    num.setup(toks)
    nums = num(toks)
    sl = SEQUENCE_LEN
    bs = BATCH_SIZE
    seqs = L([(tensor(nums[i:i+sl]), tensor(nums[i+1:i+sl+1])) for i in range(0,len(nums)-sl-1,sl)])
    cut = int(len(seqs) * 0.8)
    dls = DataLoaders.from_dsets(group_chunks(seqs[:cut], bs),
                                 group_chunks(seqs[cut:], bs), bs=bs, drop_last=True, shuffle=False)
    return dls

    def loss_func(inp, targ):
        return F.cross_entropy(inp.view(-1, VOCAB_SIZE), targ.view(-1))

if __name__ == '__main__':
    link = "https://www.gutenberg.org/files/100/100-0.txt"
    lines = get_data(link).split(
        "VENUS AND ADONIS", 1)[1].split(
        "*** END OF THE PROJECT GUTENBERG", 1)[0][:500_000]
    dls = prepare_data(lines)
    model = TheSonnetRNN(256, 512, VOCAB_SIZE, SEQUENCE_LEN)
    trainer = Trainer(model, train_dl, val_dl)
    trainer.train(50)

