import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class CharToWord(nn.Module):
    """
    The character to word-level module.
    """

    def __init__(self, num_chars, char_emb_size, word_hidden_size, context_vector_size,
                 dropout_p=0.0, projection_nonlinearity=nn.Tanh, rnn=nn.GRU):
        super(CharToWord, self).__init__()

        # Character embeddings
        self.char_embeddings = nn.Embedding(num_chars, char_emb_size)

        # Dropout applied to the embeddings
        self.dropout = nn.Dropout(p=dropout_p)

        # Bidirectional RNN
        # Inputs: character embeddings
        # Outputs: word vector (word_hidden_size * 2)
        self.char_to_word = rnn(char_emb_size, word_hidden_size, bidirectional=True,
                                batch_first=True)

        # Learnable context vector
        self.word_context = nn.Parameter(torch.Tensor(context_vector_size, 1).uniform_(-0.1, 0.1))

        # Projects the word vectors to new space to be multiplied by the context vector
        self.word_projection = nn.Linear(word_hidden_size * 2, context_vector_size)

        # The nonlinearity to apply to the projections prior to multiplication
        # by context vector
        self.word_proj_nonlinearity = projection_nonlinearity()

        # Softmax layer to convert attention * projection into weights
        self.softmax = nn.Softmax()


    def _sort_char_tensor(self, padded_tensor, sequence_lens):
        '''
        "Packing" of the character indices prior to embedding requires that they
        be in descending order of length to work.

        Returns the sorted tensor, the sequence lengths, and the indices for
        inverting the order.
        '''
        sequence_lens, order = sequence_lens.sort(0, descending=True)
        padded_tensor = padded_tensor[order]
        return padded_tensor, sequence_lens, order


    def forward(self, padded_char_tensor, sequence_lens):

        char_sorted, sequence_lens, order = self._sort_char_tensor(padded_char_tensor, sequence_lens)

        # embed
        char_embed = self.char_embeddings(char_sorted)

        # apply dropout to the embeddings
        char_embed = self.dropout(char_embed)

        # pack the sequences for efficiency
        packed = pack_padded_sequence(char_embed, sequence_lens.numpy(), batch_first=True)

        # run through the bidirectional GRU
        output, (hidden, cell) = self.char_to_word(packed)

        # unpack the sequence
        output, _ = pad_packed_sequence(output, batch_first=True)

        # revert to the original ordering
        output = output[order, :, :]

        # prepare final word tensor:
        word_tensor = Variable(torch.zeros((output.size(0), output.size(2))))

        # calculate and apply attention
        for word_ind in range(output.size(0)):

            # create the projection of the word representation
            projection = self.word_projection(output[word_ind])
            projection = self.word_proj_nonlinearity(projection)

            # compute "similarity" weighting via the word context vector
            attention = torch.mm(projection, self.word_context)
            attention = self.softmax(attention)

            # multiply the word vectors by their calculated attention weight
            word_tensor[word_ind, :] = output[word_ind].transpose(1, 0).mv(attention.view(-1))

        # return the word vector reps:
        return word_tensor



class WordToMessage(nn.Module):
    """
    The word-to-message module.
    """

    def __init__(self, word_hidden_size, message_hidden_size, context_vector_size,
                 dropout_p=0.0, projection_nonlinearity=nn.Tanh, rnn=nn.GRU):
        super(WordToMessage, self).__init__()

        # Dropout applied to word vectors
        self.dropout = nn.Dropout(p=dropout_p)

        # Message bidirectional RNN
        # Inputs: word vectors
        # Outputs: message vectors
        self.word_to_message = rnn(word_hidden_size, message_hidden_size,
                                   bidirectional=True, batch_first=True)

        # Learned message context layer
        self.message_context = nn.Parameter(torch.FloatTensor(context_vector_size, 1).uniform_(-0.1, 0.1))

        # Message projection transformation
        self.message_projection = nn.Linear(message_hidden_size * 2, context_vector_size)

        # Nonlinearity applied to projection
        self.message_proj_nonlinearity = projection_nonlinearity()

        # Softmax required to turn attention vector to weights
        self.softmax = nn.Softmax()


    def forward(self, word_tensor):
        # currently built for just 1 message; I didn't notice any tangible
        # speedups with batching and the code was a lot less clear and concise...

        # apply dropout
        word_tensor = self.dropout(word_tensor)

        # reshape for RNN
        word_tensor = word_tensor.view(1, word_tensor.size(0), word_tensor.size(1))

        # run through the bidirectional RNN
        output, (hidden, cell) = self.word_to_message(word_tensor)

        # create the projection of the message representation
        projection = self.message_projection(output[0])
        projection = self.message_proj_nonlinearity(projection)

        # compute "similarity" weighting via the context vector
        self.attention = torch.mm(projection, self.message_context)

        # apply softmax to convert to weights
        self.attention = self.softmax(self.attention)

        # apply the weighting and sum words together:
        output = output[0].transpose(1, 0).mv(self.attention.view(-1))

        return output


class HierarchicalAttentionRNN(nn.Module):
    '''
    Designed based on this paper:
    https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf

    Contains the CharToWord and WordToMessage RNNs
    '''

    def __init__(self, num_chars, char_emb_size, word_hidden_size, message_hidden_size,
                 word_context_size, message_context_size, output_size,
                 word_dropout_p=0.0, message_dropout_p=0.0, output_dropout_p=0.0,
                 char_projection_nonlinearity=nn.Tanh, word_projection_nonlinearity=nn.Tanh,
                 char_to_word_rnn=nn.GRU, word_to_message_rnn=nn.GRU,
                 intermediate_output_step=False, intermediate_output_nonlinearity=nn.ELU):
        super(HierarchicalAttentionRNN, self).__init__()

        self.num_chars = num_chars
        self.char_emb_size = char_emb_size
        self.word_hidden_size = word_hidden_size
        self.message_hidden_size = message_hidden_size
        self.word_context_size = word_context_size
        self.messge_context_size = message_context_size
        self.output_size = output_size

        # Output to be performed on the message vectors prior to transformation
        # into likelihoods/probabilities
        self.dropout = nn.Dropout(p=output_dropout_p)

        # character to word module:
        self.char_to_word = CharToWord(num_chars, char_emb_size, word_hidden_size,
                                       word_context_size, dropout_p=word_dropout_p,
                                       projection_nonlinearity=char_projection_nonlinearity,
                                       rnn=char_to_word_rnn)

        # word to message module:
        self.word_to_message = WordToMessage(word_hidden_size * 2, message_hidden_size,
                                             message_context_size, dropout_p=message_dropout_p,
                                             projection_nonlinearity=word_projection_nonlinearity,
                                             rnn=word_to_message_rnn)

        # set up the intermediate output step, if required
        # note that an intermediate nonlinear output step would deviate from the
        # specification in Yang et al.
        self.intermediate = intermediate_output_step
        if self.intermediate:
            self.intermediate_output = nn.Linear(message_hidden_size * 2, message_hidden_size * 2)
            self.intermediate_nonlinearity = intermediate_output_nonlinearity()

        # final transformation to class weightings
        self.likelihoods = nn.Linear(message_hidden_size * 2, output_size)


    def forward(self, padded_char_batches, sequence_len_batches):
        # calculate the number of batches
        batches = len(padded_char_batches)

        # create the placeholder variable for the message encodings:
        self.messages = Variable(torch.zeros((batches, self.message_hidden_size * 2)))

        # iterate through batches
        # perform the char-to-word step, then the word-to-message step
        for i in range(batches):
            word_tensor = self.char_to_word(padded_char_batches[i], sequence_len_batches[i])
            self.messages[i, :] = self.word_to_message(word_tensor)

        # perform any dropout on the message encodings:
        self.messages = self.dropout(self.messages)

        # if an intermediate output step is specified, perform it:
        if self.intermediate:
            self.messages = self.intermediate_output(self.messages)
            self.messages = self.intermediate_nonlinearity(self.messages)

        # get the weights on classes (not actual "likelihoods" since these
        # can be negative, just a convenient name for it)
        # note that softmax is designed to be performed outside of this module
        # with CrossEntropyLoss for example
        outputs = self.likelihoods(self.messages)

        return outputs, self.messages
