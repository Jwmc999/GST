#!usr/bin/bash
# coding=utf-8
import os
import six
import pickle
import copy
import shutil

import torch
import numpy as np
import random

from modeling.model import * 
from utils.util import *
from argparse import ArgumentParser

parser = ArgumentParser()
random_seed = 12345

## Parameters for each dataset
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--reset', action='store_true', 
                    help="Whether to start over entire training from the scratch.")
parser.add_argument('--no-reset', dest='reset', action='store_false')
parser.set_defaults(feature=True)

## Hyperparameter Optimizing 
parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for training.")
parser.add_argument("--beam_size", default=3, type=int, help="Beam size for beam search.")
parser.add_argument(
    "--max_seq_length", default=20, type=int,
    help="The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")
parser.add_argument("--max_predictions_per_seq", default=6, type=int,
                     help="Maximum number of gap SG predictions per sequence. "
                     "Must match data generation.")
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument("--dim", default=128, type=int, help="Dimension of PegiRec model.")
parser.add_argument('--theme', type=int, default=7, help='k-most frequent items in corpus')

## Other parameters
parser.add_argument('--epoch', type=int, default=1, help='number of epochs to train for')

opt = parser.parse_args()
print(opt)

# Todo
unique_id = 0


# Initial checkpoint for evaluation and finetuning 
ckpt_dir = 'models/' + opt.dataset + '/' + str(unique_id) 
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    
# Delete 'models/dataset/unique_id' directory
# Usually when hyperparameter optimizing is needed
if opt.reset:
    shutil.rmtree(ckpt_dir)
    print("models/dataset/unique_id, deleted")


class PegasusConfig(object):
    """Configuration for `PegasusModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_encoder_layers=12,
                 num_decoder_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 label_smoothing=0.1,
                 batch_size=opt.batch_size,
                 lr=opt.lr):
        """Constructs PegasusConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `PegasusModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_encoder_layers: Number of hidden layers in the Transformer encoder.
            num_decoder_layers: Number of hidden layers in the Transformer decoder.
                If None, use `num_encoder_layers`
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder and decoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder and decoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder, decoder, and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, decoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `PegasusModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers # encoder_layers
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else num_encoder_layers # decoder_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size
        self.lr = lr

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `PegasusConfig` from a Python dictionary of parameters."""
        config = PegasusConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output



# libraries
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch.nn

matplotlib.rcParams['figure.figsize'] = [18, 12]

# code from this library - import the lines module
import loss_landscapes
import loss_landscapes.metrics

# contour plot resolution
STEPS = 40


def main():
    train_data = pickle.load(open('data_dir/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('data_dir/' + opt.dataset + '/test.txt', 'rb'))
    
    # Randomness (Non-deterministic) Control
    # torch.manual_seed(random_seed)
    rng = random.Random(random_seed)    
    
    param = {
        "vocab_size": None, "hidden_size": opt.dim,
        "num_encoder_layers": 2, "num_decoder_layers": 2,
        "num_attention_heads": 2, "hidden_act": "relu",
        "intermediate_size": 256, "hidden_dropout_prob": 0.5,
        "attention_probs_dropout_prob": 0.5, 
        "type_vocab_size": 2, "initializer_range": 0.02,
        "label_smoothing": 0.0, #"beta": opt.beta, "layers": opt.layers
        }
        
    train_data, test_data, vocab_size = reorder(train_data, test_data, rank=opt.theme)
    _train_data = Load(train_data, rng, shuffle=True, 
                      max_seq_len=opt.max_seq_length,
                      n_pred=opt.max_predictions_per_seq)
    train_loader = torch.utils.data.DataLoader(_train_data, 
                        batch_size=opt.batch_size, 
                        shuffle=True)

    train_data = Data(train_data, rng, shuffle=True, 
                      max_seq_len=opt.max_seq_length,
                      n_pred=opt.max_predictions_per_seq)
    test_data = Data(test_data, rng, shuffle=False, 
                     max_seq_len=opt.max_seq_length,
                     n_pred=opt.max_predictions_per_seq)

    param['vocab_size'] = vocab_size
    config = PegasusConfig.from_dict(param)
    model = trans_to_cuda(GST(dataset=opt.dataset,
                                  max_seq_len=opt.max_seq_length,
                                  config=config))

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    criterion = torch.nn.CrossEntropyLoss()
    train(model, optimizer, criterion, train_loader, opt.epoch)


    model_final = copy.deepcopy(model)
    x, y = iter(train_loader).__next__()
    metric = loss_landscapes.metrics.Loss(criterion, x, x)
    loss_data_fin = loss_landscapes.random_plane(model_final, metric, 10, STEPS,
                                                 normalization='model', deepcopy_model=True)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')
    plt.savefig('theme' + str(opt.theme) + 'fig.png', dpi=300)

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        # Todo
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data,
                                         max_dec_len=opt.max_predictions_per_seq, 
                                        beam_size=opt.beam_size)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
    
        # Save model checkpoint
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'loss': total_loss,
                    'metrics': metrics}, os.path.join(ckpt_dir,"model.pt"))

        result = {  'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'loss': total_loss,
                    'metrics': metrics}

        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tP@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
    



if __name__ == '__main__':
    main()
    
