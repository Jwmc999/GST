# GST
The Pytorch implementation of "Session-based recommendation" part of "Autoregressive Decoder with Extracted Gap Sessions for Sequential/Session-based Recommendation", submitted for AAAI 23' Main Track. 

# Requirements
- Pytorch-cuda 1.9.0
- Python>=3.6.13

# Run
```
python run.py --dataset yoochoose --batch_size BATCH_SIZE --beam_size BEAM_SIZE --max_seq_length MAX_SEQ_LENGTH --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ --dim DIM --theme THEME --epoch EPOCH 
```

