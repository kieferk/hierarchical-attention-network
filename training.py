import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import time
import math



def char_to_inds(parsed_str, char_label_encoder, eow_ind=1, eos_ind=2):
    '''
    Takes a parsed string (as in, all characters are accounted for by the encoder)
    and sklearn LabelEncoder fit on the characters.

    Returns a list of lists where the internal lists are character indices for
    each letter in the word.

    End-of-word and end-of-sentence codes are automatically added.
    '''
    words = [x for x in parsed_str.split() if x]
    inds = [list(char_label_encoder.transform(list(w))+3)+[eow_ind] for w in words] + [[eos_ind]]
    return inds


def pad_char_inds(char_inds):
    '''
    To "pack" the character indices in pytorch and make the GRU more efficient,
    the sequences need to be padded with zeros as they are not guaranteed to
    be of the same length.

    Inputs:
        character indices (list of lists)
    Outputs:
        padded tensor (batch first!)
        sequence lengths (torch LongTensor)
    '''
    # Get lengths of each word, in characters:
    sequence_lens = torch.LongTensor(list(map(len, char_inds)))

    # Pad the back of the sequences and make a Variable:
    seq_tensor = Variable(torch.zeros((len(char_inds), sequence_lens.max()))).long()
    for i, (seq, seqlen) in enumerate(zip(char_inds, sequence_lens)):
        seq_tensor[i, :seqlen] = torch.LongTensor(list(map(lambda x: int(x), seq)))

    return seq_tensor, sequence_lens



def inds_to_padded_vars(ind_sets):
    '''
    Pads multiple sets of character indices.
    '''
    return list(zip(*[pad_char_inds(i) for i in ind_sets]))



class Sampler(object):

    def __init__(self, char_indices, targets, with_replacement=True,
                 sample_size=1000, sample_p=None):
        super(Sampler, self).__init__()

        self.N = len(targets)
        self.indices = np.arange(self.N)

        self.data = char_indices
        self.targets = targets

        self.with_replacement = with_replacement
        self.sample_size = sample_size
        self.sample_p = sample_p


    def generate_batch(self):
        return np.random.choice(self.indices, size=self.sample_size,
                                replace=self.with_replacement, p=self.sample_p)

    def next(self):
        inds = self.generate_batch()
        return ([self.data[ind] for ind in inds],
                [self.targets[ind] for ind in inds])



def entropy_loss(outputs, beta):
    '''
    Calculates the average entropy of the outputs. This is then multiplied
    by some scaling factor beta.

    Regularizes the model. Predicted probabilities are penalized when a tight a distribution,
    which is a way to prevent overfitting.

    See: https://arxiv.org/pdf/1701.06548.pdf
    '''
    entropy = torch.sum(F.softmax(outputs) * F.log_softmax(outputs), 1).view(-1) * -1
    return torch.mean(entropy) * beta



class Evaluation(object):

    def __init__(self, data, targets, messages, target_ref_dict, char_encoder,
                 criterion=nn.CrossEntropyLoss):
        super(Evaluation, self).__init__()

        self.criterion = criterion()

        self.data = data
        self.padded, self.seq_lens = inds_to_padded_vars(data)
        self.targets = targets
        self.messages = messages
        self.torch_targets = Variable(torch.from_numpy(np.array(targets)).long(),
                                      volatile=True)
        self.target_ref_dict = target_ref_dict
        self.char_encoder = char_encoder


    def test(self, model):
        model.train(mode=False)

        test_out, _ = model(self.padded, self.seq_lens)

        topv, topi = torch.max(test_out, 1)
        topi = topi.view(-1).data.numpy()

        test_loss = self.criterion(test_out, self.torch_targets).data[0]
        test_acc = accuracy_score(self.targets, topi)

        print_str = 'TESTING:   loss: %.4f   acc: %.4f'
        print(print_str % (test_loss, test_acc))
        print('-----------------------------------------------------')

        model.train(mode=True)

        return test_loss, test_acc


    def example(self, model, examples=1, k=5):
        model.train(mode=False)

        for j in range(examples):
            to_test = np.random.choice(np.arange(len(self.targets)))
            p, sl = self.padded[to_test:to_test+1], self.seq_lens[to_test:to_test+1]
            message = self.messages[to_test]
            target = self.targets[to_test]
            target_label = self.target_ref_dict[target]

            output, _ = model(p, sl)

            topv, topi = output.data.topk(output.view(-1).size(0))
            top = pd.DataFrame({'p':F.softmax(topv).view(-1).data.numpy(),
                                'label':topi.view(-1).numpy()})

            top = top.head(k)
            top['empathy'] = top['label'].map(lambda x: self.target_ref_dict[x])

            tar = pd.DataFrame({'p':F.softmax(output.data.view(-1)).view(-1)[target].data.numpy(),
                                'label':[target],
                                'empathy':[target_label]})

            print(message)
            print(tar[['label','p','empathy']])
            print('Predicted:')
            print(top[['label','p','empathy']])
            print('-----------------------------------------------------')

        model.train(mode=True)

    def eval_message(self, model, message, k=5):
        model.train(mode=False)

        char_input = char_to_inds(message, self.char_encoder)
        padded, sequence_lens = inds_to_padded_vars([char_input])

        output, _ = model(padded, sequence_lens)

        topv, topi = output.data.topk(output.view(-1).size(0))
        top = pd.DataFrame({'p':F.softmax(topv).view(-1).data.numpy(),
                            'label':topi.view(-1).numpy()})

        top = top.head(k)
        top['empathy'] = top['label'].map(lambda x: self.target_ref_dict[x])

        print(message)
        print(top)
        print('-----------------------------------------------------')

        model.train(mode=True)


class Trainer(object):

    def __init__(self, model, data, targets,
                 optimizer=optim.Adamax, criterion=nn.CrossEntropyLoss,
                 learning_rate=0.01, entropy_weight=0.0, weight_entropy_by_accuracy=False,
                 class_weights=None, evaluator=None,
                 with_replacement=True, sample_size=1000, sample_p=None,
                 save_path=None, save_on_best_path=None, verbose=True,
                 early_stopping_test_acc=None):
        super(Trainer, self).__init__()

        self.verbose = verbose
        self.save_path = save_path
        self.save_on_best_path = save_on_best_path

        self.model = model
        self.data = data
        self.targets = targets

        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        self.entropy_weight = entropy_weight
        if weight_entropy_by_accuracy:
            self.entropy_f = self._entropy_from_accuracy
        else:
            self.entropy_f = lambda: self.entropy_weight

        self.sampler = Sampler(self.data, self.targets, with_replacement=with_replacement,
                               sample_size=sample_size, sample_p=sample_p)

        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = torch.from_numpy(class_weights).float()

        self.criterion = criterion(weight=self.class_weights)

        self.validate = False if evaluator is None else True
        self.evaluator = evaluator

        self.training_losses, self.validation_losses = [], []
        self.training_accs, self.validation_accs = [], []

        self.early_stop_thresh = 1.1 if early_stopping_test_acc is None else early_stopping_test_acc


    def _entropy_from_accuracy(self):
        last_acc = self.training_accs[-1]
        safe_acc = 0.99 if (last_acc > 0.99) else last_acc
        return np.arctanh(safe_acc) * self.entropy_weight


    def _as_minutes(self, s):
        """
        Took this from some pytorch documentation/tutorial.
        """
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


    def _time_since(self, since, percent):
        """
        Took this from some pytorch documentation/tutorial.
        """
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self._as_minutes(s), self._as_minutes(rs))


    def train(self, char_batch, target_batch):

        self.optimizer.zero_grad()
        loss = 0.

        padded, sequence_lens = inds_to_padded_vars(char_batch)
        targets = Variable(torch.LongTensor(list(map(lambda x: int(x), target_batch))))

        outputs, _ = self.model(padded, sequence_lens)

        loss += self.criterion(outputs, targets)

        if self.entropy_weight > 0.:
            loss -= entropy_loss(outputs, self.entropy_weight)

        topv, topi = outputs.data.topk(1)
        yhat = topi[:, 0].view(-1)

        loss.backward()
        self.optimizer.step()

        return loss.data[0], yhat


    def test(self):
        test_loss, test_acc = self.evaluator.test(self.model)
        self.evaluator.example(self.model)

        self.validation_losses.append(test_loss)
        self.validation_accs.append(test_acc)

        if (test_acc >= np.max(self.validation_accs)) and self.save_on_best_path:
            to_save = self.model.state_dict()
            torch.save(to_save, self.save_on_best_path)


    def training_loop(self, n_iters=100, test_every=1):

        start = time.time()
        ent_weight = self.entropy_weight

        for i in range(1, n_iters + 1):
            print('Iteration:', i)

            chars, ytrue = self.sampler.next()

            loss, yhat = self.train(chars, ytrue)

            self.training_losses.append(np.sum(loss))
            self.training_accs.append(accuracy_score(ytrue, yhat.view(-1).numpy()))

            ent_weight = self.entropy_f()

            if self.verbose:
                print('New entropy weight: %.4f' % (ent_weight))
                print_str = 'MODEL: %s (%d %d%%)   loss: %.4f   acc: %.4f'
                print(print_str % (self._time_since(start, i / n_iters),
                                   i, i / n_iters * 100,
                                   self.training_losses[-1],
                                   self.training_accs[-1]))
                print('-----------------------------------------------------')

            if self.validate and (i % test_every) == 0:
                self.test()
                if self.validation_accs[-1] >= self.early_stop_thresh:
                    print('--------------------EARLY STOP-----------------------')
                    print('Threshold reached or passed:', self.early_stop_thresh)

        if self.save_path:
            to_save = self.model.state_dict()
            torch.save(to_save, self.save_path)
