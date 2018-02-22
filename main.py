"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import random

#local imports
from preprocessing import input_reader
from encdec import RNNEncoder, RNNDecoder
from training import train_setup
from utils import use_cuda


def main():
    print("Use CUDA: {}".format(use_cuda))  #currently always false

    src_lang = 'en'
    tgt_lang = 'de'
    data_prefix = 'data/examples/debug'
    
    src_vocab, tgt_vocab, train_sents = input_reader(data_prefix, src_lang, tgt_lang)

    hidden_size = 64
    input_size  = src_vocab.vocab_size()
    output_size = tgt_vocab.vocab_size()

    #-------------------------------------

    enc = RNNEncoder(input_size, hidden_size)
    dec = RNNDecoder(output_size, hidden_size)
    if use_cuda:
        enc = enc.cuda()
        dec = dec.cuda()

    train_setup(enc, dec, train_sents, num_iters=1000, print_every=25)    

#    enc.save('enc.pkl')
#    dec.save('dec.pkl')

    pass


if __name__ == "__main__":
    main()

