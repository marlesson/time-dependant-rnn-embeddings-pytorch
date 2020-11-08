# Time-Dependent Embedding Layer

This is a Pytorch implementation of the best performing approach to embedding time into RNNs. This is usefull for cases there your stream of RNNs steps contain timestep information, and those timesteps are non-equally spaced. This approach performs better than just adding timestep as a single feature.

This repository was based on https://github.com/crazyleg/time-dependant-rnn-embeddings-keras

References:

* Original Paper - https://arxiv.org/abs/1708.00065
* How-to encode time property in recurrent neural networks. - https://fridayexperiment.com/how-to-encode-time-property-in-recurrent-neutral-networks/

## Usage

