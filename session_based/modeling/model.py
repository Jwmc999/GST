import datetime
import numpy as np
import collections
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import torch.sparse

import modeling.attention as attention
import modeling.transformer_block as transformer_block
import modeling.decoding as decoding
import utils.timing as timing

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
    

class Session(Module):
    """Session Embedding layer supporting shared input/output weights."""
    def __init__(self, vocab_size, hidden_size):
        super(Session, self).__init__()
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._dtype = torch.float32

    def _ids_to_weights(self, ids_BxI):
        """Maps IDs to embedding weights."""
        weights_BxIxD = trans_to_cuda(F.embedding(ids_BxI.cuda(), self.weights_VxD.cuda()))
        weights_BxIxD *= self._hidden_size**0.5
        return weights_BxIxD

    def _weights_to_logits(self, states_BxIxD):
        shapes = list(states_BxIxD.size())
        B, I, D = shapes[0], shapes[1], shapes[2]
        states_BIxD = states_BxIxD.view(-1, D)
        states_BIxV = trans_to_cuda(torch.matmul(states_BIxD, self.weights_VxD.t().cuda()))
        states_BxIxV = states_BIxV.view(B, I, self._vocab_size)
        return states_BxIxV

    @property
    def weights_VxD(self):
        """Gets embedding weights."""
        # Initialization is important here, and a normal distribution with stdev
        # equal to rsqrt hidden_size is significantly better than the default
        # initialization used for other layers (fan in / out avg).        
        stddev=self._hidden_size**-0.5
        embeddings_VxD = torch.autograd.Variable(
            torch.randn(size=(self._vocab_size, self._hidden_size), 
                        dtype=self._dtype, 
                        requires_grad=False)*stddev)
        return embeddings_VxD

    def forward(self, tensor, is_input_layer):
        if is_input_layer:
            return self._ids_to_weights(tensor)
        else:
            return self._weights_to_logits(tensor)


class GST(Module):
    def __init__(self, dataset, max_seq_len, config):
        super(GST, self).__init__()
        self.dataset = dataset
        self._dtype = torch.float32
        self._embedding_layer = trans_to_cuda(Session(config.vocab_size,
                                        config.hidden_size))
        self._decoder_layers = trans_to_cuda(nn.ModuleList(
            [transformer_block.TransformerBlock(
            hidden_size=config.hidden_size, 
            hidden_act=config.hidden_act, 
            intermediate_size=config.intermediate_size,
            num_heads=config.num_attention_heads, 
            dropout=config.hidden_dropout_prob) for _ in range(config.num_decoder_layers)]))
        dropout = trans_to_cuda(nn.Dropout(p=config.hidden_dropout_prob))
        self._dropout_fn = lambda x, training: dropout(x) if training else x
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.max_seq_len = max_seq_len
        self.num_attention_heads = config.num_attention_heads
        self.label_smoothing = config.label_smoothing
        self.lr = config.lr
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, 
                                           weight_decay=0.01, eps=1e-08,
                                           betas=(0.9, 0.999))

    def decoder(self, inputs, training):
        """Create pretraining model. 

        Args:
            inputs: dictionary of tensors including "sess" for session embedding
            and "hgcn" for HGCN embedding

        Returns:
            Tuple of (loss, outputs): Loss is a scalar. Output is a dictionary of
             tensors, containing model's output logits.
        """
        # input for session
        output_label = inputs
        # output_ids, output_label, pad_ids = inputs["sess"]
        targets_BxT = output_label 
        bias_1xTxT = attention.upper_triangle_bias(
            targets_BxT.shape[1], self._dtype).cuda()
        
        # Pegasus-based session embedding
        states_BxTxD = self._embedding_layer(targets_BxT, True)
        states_BxTxD = F.pad(states_BxTxD, [0, 0, 1, 0])[:, :-1, :]
        states_BxTxD = timing.add_time_signal(states_BxTxD)
        states_BxTxD = self._dropout_fn(states_BxTxD, training)
        states_BxTxD = transformer_block.stack(self._decoder_layers, training,
                                                    states_BxTxD, bias_1xTxT,
                                                    None, None)
        states_BxTxD = transformer_block.layer_norm(states_BxTxD)
        # embeding weights to logits
        logits_BxTxV = self._embedding_layer(states_BxTxD, False)
        targets_mask_BxT = torch.gt(targets_BxT, 0).type(self._dtype)
        # Loss funciton: Softmax cross entropy
        # weight=targets_mask_BxT[-1, :].clone(),
        #                                 ignore_index=pad_ids.item() 
        criterion = nn.CrossEntropyLoss()
        criterion.cuda()
        loss = criterion(states_BxTxD[:, :, -1].clone() + 1e-8, 
                         torch.max(targets_BxT.clone().cuda(), 1)[1])
        return loss, states_BxTxD[:, :, -1].clone() + 1e-8 #{"logits": logits_BxTxV}
        
    def predict(self, labels, max_decode_len, beam_size, **beam_kwargs):
        """Predict. Fine tuning model."""
        cache = collections.defaultdict(list)
        # Initialize cache for decoding
        B, D = self.batch_size, self.hidden_size
        T, V, H = max_decode_len, self.vocab_size, self.num_attention_heads
        bias_1xTxT = attention.upper_triangle_bias(T, self._dtype).cuda()
        for i in range(len(self._decoder_layers)):
            cache[str(i)] = {
                "k": torch.zeros([B, H, T, D // H], dtype=self._dtype),
                "v": torch.zeros([B, H, T, D // H], dtype=self._dtype)
            }
            
        def symbols_to_logits_fn(dec_BxT, context, i):
            """Decode loop."""
            dec_shape = list(dec_BxT.size())
            zero = 0
            max = torch.maximum(torch.tensor((zero)).clone().type(torch.int32), 
                                torch.tensor((i - 1)).clone())
            dec_Bx1 = dec_BxT[0:dec_shape[0], max:max+1]
            bias_1x1xT = bias_1xTxT[0:1, i:i+1, 0:T].clone()
            dec_Bx1xD = self._embedding_layer(dec_Bx1, True)
            dec_Bx1xD *= torch.gt(torch.tensor(i).cuda(), 0).type(self._dtype)  
            dec_Bx1xD = timing.add_time_signal(dec_Bx1xD, start_index=i) 
            dec_Bx1xD = transformer_block.stack(self._decoder_layers, False,
                                                    dec_Bx1xD, bias_1x1xT,
                                                    None,
                                                    None, context, 
                                                    torch.tensor(i).cuda())
            dec_Bx1xD = transformer_block.layer_norm(dec_Bx1xD)
            # embeding weights to logits
            logits_Bx1xV = self._embedding_layer(dec_Bx1xD, False)  
            return logits_Bx1xV

        decodes_BxT, scores = decoding.left2right_decode(symbols_to_logits_fn, cache, B, T,
                                                V, beam_size, **beam_kwargs)
        if beam_size > 1:
            decodes_BxT = torch.flatten(decodes_BxT)
            scores = torch.flatten(scores)
        return {"outputs": decodes_BxT}, {"gap_sg_log_probs": scores}
        
    def forward(self, inputs, training=True, max_dec_len=None, 
                beam_size=None, **beam_kwargs):
        ## Beam-search Decoder
        if training:
            predictions = self.decoder(inputs, training)    
        else:
            predictions = self.predict(inputs, max_dec_len, 
                                       beam_size,**beam_kwargs)
        return predictions
    
    
def forward(model, i, data, training, max_dec_len=None,
                beam_size=None, **beam_kwargs):
    output_ids, output_label, pad_ids = data.get_batch(i)
    # Session masking Layer input dataset
    output_ids = trans_to_cuda(output_ids)
    output_label = trans_to_cuda(output_label)
    pad_ids = trans_to_cuda(torch.Tensor([pad_ids]).long())
    # inputs = {"sess": [output_ids, output_label, pad_ids]}
    results = model(output_label, training, max_dec_len, beam_size, **beam_kwargs)
    return output_label, results

def train_test(model, train_data, test_data, max_dec_len, beam_size, **beam_kwargs):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i in slices:
        model.zero_grad()
        _, results = forward(model, i, train_data, True)
        loss, logits = results
        loss.backward()
        model.optimizer.step()
        total_loss += loss

    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    output = []
    for i in slices:
        tar, results = forward(model, i, test_data, False,
                               max_dec_len, beam_size, **beam_kwargs)
        outputs, log_probs = results
        output.append(outputs["outputs"])
        tar = trans_to_cpu(tar).detach().numpy()

        for K in top_K:
            sub_scores = log_probs["gap_sg_log_probs"].topk(K)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, tar):
                metrics['hit%d' %K].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    metrics['mrr%d' %K].append(0)
                else:
                    metrics['mrr%d' %K].append(1 / (np.where(score == target - 1)[0][0] + 1))

    print('predicting finished: ', datetime.datetime.now())
    return metrics, total_loss


from tqdm import tqdm

def train(model, optimizer, criterion, train_loader, epochs):
    """ Trains the given model with the given optimizer, loss function, etc. """
    model.train()
    # train model
    for _ in tqdm(range(epochs), 'Training'):
        for count, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            x, y = batch
            x = trans_to_cuda(x)
            y = trans_to_cuda(y.long())

            _, pred = model(x)
            loss = criterion(pred, torch.max(x.clone(), 1)[1])
            loss.backward()
            optimizer.step()

    model.eval()