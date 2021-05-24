# Variance model based speech enhancement

The underlying variance model framework is proposed in

+ [*S. Leglaive, L. Girin, R Horaud (2019)*](https://arxiv.org/abs/1902.01605)

See below for abstract.

The variational autoencoder used here as a supervised non-linear modeling of the speech variances is a slightly modified version that is in 

+ [*BLyndon/bayesian_methods/notebooks/*](https://github.com/BLyndon/bayesian_methods/tree/master/notebooks)

applied as a generative model for images.

The supervised speech model is trained on

+ [The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://github.com/philipperemy/timit)

## Abstract

>In this paper we address the problem of enhancing speech signals in noisy mixtures using a source separation approach. We explore the use of neural networks as an alternative to a popular speech variance model based on supervised non-negative matrix factorization (NMF). More precisely, we use a variational autoencoder as a speaker-independent supervised generative speech model, highlighting the conceptual similarities that this approach shares with its NMF-based counterpart. In order to be free of generalization issues regarding the noisy recording environments, we follow the approach of having a supervised model only for the target speech signal, the noise model being based on unsupervised NMF. We develop a Monte Carlo expectation-maximization algorithm for inferring the latent variables in the variational autoencoder and estimating the unsupervised model parameters. Experiments show that the proposed method outperforms a semi-supervised NMF baseline and a state-of- the-art fully supervised deep learning approach.