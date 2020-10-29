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
from io import open
import itertools
import math

from .attn import Attn
from .voc import Voc
from .encoder_rnn import EncoderRNN
from .luong_attn_decoder_rnn import LuongAttnDecoderRNN
from .greedy_search_decoder import GreedySearchDecoder


def main():

    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    PATH_TO_PROJECT = '/home/vinc/Documents/chatbot/'

    corpus_name = 'cornell movie-dialogs corpus'
    corpus = os.path.join(PATH_TO_PROJECT, 'training_data', corpus_name)

    # Define path to new file
    datafile = os.path.join(corpus, 'formatted_movie_lines.txt')

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

    # Initialize lines dict, conversations list, and field ids
    lines, conversations = {}, []
    MOVIE_LINES_FIELDS = [
        'lineID', 'characterID', 'movieID', 'character', 'text'
    ]
    MOVIE_CONVERSATIONS_FIELDS = [
        'character1ID', 'character2ID', 'movieID', 'utteranceIDs'
    ]

    def printLines(fileName, n=10):
        filePath = os.path.join(corpus, fileName)
        with open(filePath, 'rb') as datafile:
            lines = datafile.readlines()
        for line in lines[:n]:
            print(line)

    # Splits each line of the file into a dictionary of fields

    def loadLines(fileName, fields):
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(' +++$+++ ')
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
        return lines

    # Groups fields of lines from `loadLines` into
    # conversations based on *movie_conversations.txt*

    def loadConversations(fileName, lines, fields):
        conversations = []
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(' +++$+++ ')
                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]
                # Convert string to list
                # (convObj['utteranceIDs'] == '['L598485', 'L598486', ...]')
                utterance_id_pattern = re.compile('L[0-9]+')
                lineIds = utterance_id_pattern.findall(convObj['utteranceIDs'])
                # Reassemble lines
                convObj['lines'] = []
                for lineId in lineIds:
                    convObj['lines'].append(lines[lineId])
                conversations.append(convObj)
        return conversations

    # Extracts pairs of sentences from conversations

    def extractSentencePairs(conversations):
        qa_pairs = []
        for conversation in conversations:
            # Iterate over all the lines of the conversation
            # We ignore the last line (no answer for it)
            for i in range(len(conversation['lines']) - 1):
                inputLine = conversation['lines'][i]['text'].strip()
                targetLine = conversation['lines'][i+1]['text'].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        return qa_pairs

    # Load lines and process conversations
    print('\nProcessing corpus...')
    lines = loadLines(os.path.join(
        corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
    print('\nLoading conversations...')
    conversations = loadConversations(
        os.path.join(corpus, 'movie_conversations.txt'),
        lines, MOVIE_CONVERSATIONS_FIELDS
    )

    # Write new csv file
    print('\nWriting newly formatted file...')
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter,
                            lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    # print('\nSample lines from file:')
    # printLines(datafile)

    # Default word tokens
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

    # Read query/response pairs and return a voc object

    def readVocs(datafile, corpus_name):
        print('Reading lines...')
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8').\
            read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        voc = Voc(corpus_name)
        return voc, pairs

    # Returns True iff both sentences in a pair 'p' are under
    # the MAX_LENGTH threshold

    def filterPair(p):
        # Input sequences need to preserve the last word for EOS token
        return len(
            p[0].split(' ')
        ) < MAX_LENGTH and len(
            p[1].split(' ')
        ) < MAX_LENGTH

    def filterPairs(pairs):
        # Filter pairs using filterPair condition
        return [pair for pair in pairs if filterPair(pair)]

    # Using the functions defined above,
    # return a populated voc object and pairs list

    def loadPrepareData(corpus, corpus_name, datafile, save_dir):
        print('Start preparing training data ...')
        voc, pairs = readVocs(datafile, corpus_name)
        print('Read {!s} sentence pairs'.format(len(pairs)))
        pairs = filterPairs(pairs)
        print('Trimmed to {!s} sentence pairs'.format(len(pairs)))
        print('Counting words...')
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print('Counted words:', voc.num_words)
        return voc, pairs

    # Load/Assemble voc and pairs
    save_dir = os.path.join(PATH_TO_PROJECT, 'data', 'save')
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    # Print some pairs to validate
    print('\npairs:')
    for pair in pairs[:10]:
        print(pair)

    MIN_COUNT = 3    # Minimum word count threshold for trimming

    def trimRareWords(voc, pairs, MIN_COUNT):
        # Trim words used under the MIN_COUNT from the voc
        voc.trim(MIN_COUNT)
        # Filter out pairs with trimmed words
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True
            # Check input sentence
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    keep_input = False
                    break
            # Check output sentence
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_output = False
                    break

            # Only keep pairs that do not contain trimmed word(s)
            # in their input or output sentence
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print('Trimmed from {} pairs to {}, {:.4f} of total'.format(
            len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
        return keep_pairs

    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, MIN_COUNT)

    def indexesFromSentence(voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

    def zeroPadding(l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(l, value=PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths

    def inputVar(l, voc):
        indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    # Returns padded target sequence tensor, padding mask, and max target length

    def outputVar(l, voc):
        indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = zeroPadding(indexes_batch)
        mask = binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

    # Returns all items for a given batch of pairs

    def batch2TrainData(voc, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = inputVar(input_batch, voc)
        output, mask, max_target_len = outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len

    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs)
                                    for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print('input_variable:', input_variable)
    print('lengths:', lengths)
    print('target_variable:', target_variable)
    print('mask:', mask)
    print('max_target_len:', max_target_len)

    def maskNLLLoss(inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = - \
            torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()

    def train(
        input_variable, lengths, target_variable,
        mask, max_target_len, encoder, decoder,
        embedding, encoder_optimizer, decoder_optimizer,
        batch_size, clip, max_length=MAX_LENGTH
    ):

        # Zero gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor(
            [[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        if random.random() < teacher_forcing_ratio:
            use_teacher_forcing = True
        else:
            use_teacher_forcing = False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(
                    decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor(
                    [[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(
                    decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def trainIters(
        model_name, voc, pairs, encoder, decoder,
        encoder_optimizer, decoder_optimizer, embedding,
        encoder_n_layers, decoder_n_layers, save_dir,
        n_iteration, batch_size, print_every, save_every,
        clip, corpus_name, loadFilename
    ):

        # Load batches for each iteration
        training_batches = [
            batch2TrainData(voc, [random.choice(pairs)
                                  for _ in range(batch_size)])
            for _ in range(n_iteration)
        ]

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if loadFilename:
            start_iteration = checkpoint['iteration'] + 1

        # Training loop
        print('Training...')
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = train(
                input_variable, lengths, target_variable,
                mask, max_target_len, encoder, decoder, embedding,
                encoder_optimizer, decoder_optimizer, batch_size, clip
            )
            print_loss += loss

            # Print progress
            if iteration % print_every == 0:
                print_loss_avg = print_loss / print_every
                print(
                    'Iteration: {}; Percent complete: {:.1f}%; ' +
                    'Average loss: {:.4f}'.format(
                        iteration, iteration / n_iteration * 100, print_loss_avg)
                )
                print_loss = 0

            # Save checkpoint
            if (iteration % save_every == 0):
                directory = os.path.join(
                    save_dir, model_name, corpus_name, '{}-{}_{}'.format(
                        encoder_n_layers, decoder_n_layers, hidden_size
                    )
                )
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': voc.__dict__,
                    'embedding': embedding.state_dict()
                }, os.path.join(
                    directory, '{}_{}.tar'.format(iteration, 'checkpoint'))
                )

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

    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 4000
    # loadFilename = os.path.join(
    #     save_dir, model_name, corpus_name, '{}-{}_{}'.format(
    #         encoder_n_layers, decoder_n_layers, hidden_size
    #       ), '{}_checkpoint.tar'.format(checkpoint_iter)
    # )

    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size,
        voc.num_words, decoder_n_layers, dropout
    )
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 6000
    print_every = 1
    save_every = 1000

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(
        encoder.parameters(), lr=learning_rate
    )
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate * decoder_learning_ratio
    )
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print('Starting Training!')
    trainIters(
        model_name, voc, pairs, encoder, decoder,
        encoder_optimizer, decoder_optimizer, embedding,
        encoder_n_layers, decoder_n_layers, save_dir,
        n_iteration, batch_size, print_every, save_every,
        clip, corpus_name, loadFilename
    )

    # run evaluation

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    torch.save(
        GreedySearchDecoder,
        os.path.join(PATH_TO_PROJECT,
                     'saved_models/test1_seq2seq_rnn/searcher')
    )
    torch.save(
        encoder,
        os.path.join(PATH_TO_PROJECT,
                     'saved_models/test1_seq2seq_rnn/encoder')
    )
    torch.save(
        decoder,
        os.path.join(PATH_TO_PROJECT,
                     'saved_models/test1_seq2seq_rnn/decoder')
    )
    torch.save(
        voc,
        os.path.join(PATH_TO_PROJECT, 'saved_models/test1_seq2seq_rnn/voc')
    )

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, voc)
