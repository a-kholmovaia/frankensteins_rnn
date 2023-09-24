import requests
from fastai.text.all import *

def get_data(link: str):
    f = requests.get(link)
    lines = f.text
    return lines

def prepare_data(lines):
    spacy = WordTokenizer()
    tkn = Tokenizer(spacy)
    toks = tkn(lines)
    num = Numericalize()
    num.setup(toks)
    print(coll_repr(num.vocab, 20))

if __name__ == '__main__':
    link = "https://www.gutenberg.org/files/100/100-0.txt"
    lines = get_data(link).split(
        "VENUS AND ADONIS", 1)[1].split(
        "*** END OF THE PROJECT GUTENBERG", 1)[0][:500_000]
    prepare_data(lines)
