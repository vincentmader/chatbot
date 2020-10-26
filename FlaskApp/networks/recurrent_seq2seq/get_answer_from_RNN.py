import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
import itertools
import math

from .voc import Voc
from .attn import Attn
from .encoder_rnn import EncoderRNN
from .luong_attn_decoder_rnn import LuongAttnDecoderRNN
from .greedy_search_decoder import GreedySearchDecoder


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

PATH_TO_PROJECT = '/home/vinc/Documents/chatbot/'

# # Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


MAX_LENGTH = 10  # Maximum sentence length to consider

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s).strip()
    return s


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(
                encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (
                x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print('Error: Encountered unknown word.')


def evaluateInputOnce(encoder, decoder, searcher, voc, input_sentence):
    try:
        # Normalize sentence
        input_sentence = normalizeString(input_sentence)
        # Evaluate sentence
        output_words = evaluate(
            encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (
            x == 'EOS' or x == 'PAD')]
        return ' '.join(output_words)

    except KeyError:
        return 'Error: Encountered unknown word.'


def main(q):

    encoder = torch.load(os.path.join(
        PATH_TO_PROJECT, 'network_states/test1_seq2seq_rnn/encoder'))
    decoder = torch.load(os.path.join(
        PATH_TO_PROJECT, 'network_states/test1_seq2seq_rnn/decoder'))
    voc = torch.load(os.path.join(
        PATH_TO_PROJECT, 'network_states/test1_seq2seq_rnn/voc'))
    searcher = torch.load(os.path.join(
        PATH_TO_PROJECT, 'network_states/test1_seq2seq_rnn/searcher'))
    evaluateInput = torch.load(os.path.join(
        PATH_TO_PROJECT, 'network_states/test1_seq2seq_rnn/evaluateInput'))

    return evaluateInputOnce(
        encoder, decoder, searcher, voc, q
    )
