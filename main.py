"""Main file for 11-747 Project. By Alex Coda, Andrew Runge, & Liz Salesky."""
import argparse
import random

# Local imports
from preprocessing import prepareData
from neuralnet import Encoder, Decoder, AttnDecoder
from training import Trainer
from utils import use_cuda
# from scoring import score



def main():
    print("Use CUDA: {}".format(use_cuda))

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))

    hidden_size = 64
    enc = Encoder(input_lang.n_words, hidden_size)
    dec = Decoder(output_lang.n_words, hidden_size)
    if use_cuda:
        enc = enc.cuda()
        dec = dec.cuda()    

    trainer = Trainer(input_lang, output_lang, pairs)
    trainer.trainIters(enc, dec, n_iters=1000, print_every=25)

    enc.save('enc.pkl')
    dec.save('dec.pkl')

    pass


if __name__ == "__main__":
    main()