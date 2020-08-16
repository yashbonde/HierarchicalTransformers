# vaayuvidha
Destination Brazil!


## Installation

There will be versioning issues so better install it in venv using command `python3 -m venv .` and to activate `source bin/activate`. You'll need to run [`pytorch-geometric`](https://github.com/rusty1s/pytorch_geometric) so install it this way, where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation.
```
pip3 install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html
pip3 install torch-geometric
```

## Network

Since we are using `pytorch-geometric` things can become super easy, when making the graph neural networks. In order to test out lstm-rnn run the file [rnn_test.py](tests/rnn_text.py). LSTMs have a hidden state which means it can carry previous information in a certain `state`, which can be reasoned as, modeling the data. This makes it perfect for problems that require knowledge over very long time steps, in our case this can be 100s of step before. All these make LSTMs the obvious choice.

Only thing is that they are super difficult to train!
