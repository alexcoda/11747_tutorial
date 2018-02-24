"""training fns"""
import random
import time
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

#local imports
from utils import time_elapsed, save_plot, use_cuda, pair2var
from preprocessing import SOS, EOS


MAX_SENT_LENGTH = 10   #todo: with minibatching, max_length will be set per batch as len(longest sent in batch)

def train(src, tgt, encoder, decoder,
          encoder_optimizer, decoder_optimizer, loss_fn,
          max_length=MAX_SENT_LENGTH):

    encoder.hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0.0

    tgt_length = tgt.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder(src)
    decoder.hidden = encoder.hidden
    decoder_output = decoder(tgt)

    for gen, ref in zip(decoder_output, tgt):
        loss += loss_fn(gen, ref)

    #todo: lecture 2/20 re loss fns. pre-train with teacher forcing, finalize using own predictions

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / tgt_length


#todo: generation
#def generate():


def train_setup(encoder, decoder, sents, num_epochs, learning_rate=0.01,
                print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  #resets every print_every
    plot_loss_total  = 0  #resets every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    train_sents = [ pair2var(s) for s in sents ]
    loss_fn = nn.NLLLoss()

    num_batches = len(sents)  #todo: currently batch_size=1 sentence

    print("Starting training:")
    for ep in range(num_epochs):
        print("Epoch %d:" % ep)
        for iteration in range(len(sents)):
            src_sent = train_sents[iteration][0]
            tgt_sent = train_sents[iteration][1]
    
            loss = train(src_sent, tgt_sent, encoder,
                         decoder, encoder_optimizer, decoder_optimizer,
                         loss_fn)
            print_loss_total += loss
            plot_loss_total  += loss
    
            #todo: evaluate function. every X iterations here calculate dev ppl, bleu every epoch at least
            
            # log
            if iteration % print_every == 0 and iteration > 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('iter %d / %d: %s  %.4f' % (iteration, num_batches, time_elapsed(start, iteration / num_batches), print_loss_avg))
    
            # append losses for plot
            if iteration % plot_every == 0 and iteration > 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    

#    #todo: generate translations for test sentences here
#    sentences = []
#    for sent_id, sent in enumerate(test_sents):
#        translated_sent = generate(sent[0])
#        sentences.append(translated_sent)
#    for sent in sentences:
#        print(sent)
#    with open('output.txt', 'w', encoding='utf-8') as f:
#        f.write('\n'.join(sentences)

    #todo: evaluate test ppl, bleu

    save_plot(plot_losses)
