# RNNs & Audioset's VGGish Inference Model

- [RNNs & Audioset's VGGish Inference Model](#rnns--audiosets-vggish-inference-model)

  - [About the VGGish Model](#about-the-vggish-model)
  - [What is Loss?](#what-is-loss)
  - [Which Youtube-8m Model do I choose?](#which-youtube-8m-model-do-i-choose)
    - [LSTM Model](#lstm-model)
    - [DBof Model](#dbof-model)
    - [Frame Level Logistic Model](#frame-level-logistic-model)
  - [How should I format my audio for building a sound effect classifier that uses Audioset?](#how-should-i-format-my-audio-for-building-a-sound-effect-classifier-that-uses-audioset)
  - [Audioset Model Comparison](#audioset-model-comparison)

- [Audioset Model Comparison](#audioset-model-comparison)

## About the VGGish Model

The initial AudioSet release included 128-dimensional embeddings of each AudioSet segment produced from a VGG-like audio classification model that was trained on a large YouTube dataset (a preliminary version of what later became YouTube-8M).

They provide a TensorFlow definition of this model, which is called VGGish, as well as supporting code to extract input features for the model from audio waveforms and to post-process the model embedding output into the same format as the released embedding features.

- For example, when running the Eval.py console out gives us:

* Whats a **HIT** - Performs a local (numpy) calculation of the hit at one. float: The average hit at one across the entire batch. hits = actuals[numpy.arange(actuals.shape[0]), top_prediction]

* Whats a **GAP** - this Performs a local (numpy) calculation of the global average precision. GAP Calculates or keeps track of the interpolated average precision.It provides an interface for calculating interpolated average precision for an entire list or the top-n ranked items. For the definition of the (non-)interpolated average precision: http://trec.nist.gov/pubs/trec15/appendices/CE.MEASURES06.pdf Use it as a static function call to directly calculate average precision for a short ranked list in the memory.

* What is **PERR** - this is the Precision at Equal Recall Rate. Afloat: The average precision at equal recall rate across the entire batch. In pattern recognition, information retrieval and binary classification, precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, while recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. Both precision and recall are therefore based on an understanding and measure of relevance.

- How many **EPOCHs** - An epoch is one forward pass and one backward pass of all the training examples. the epochs you define during initializiation of training your model will be stored in your checkpoint. For LSTMs

> **Note**: Eval.py binary runs "forever" (i.e. keeps watching for updated model checkpoint and re-runs evals). To run once, pass flag `--run_once`

## What is Loss?

- What is **loss** - Loss is what you want to minimize by updating the weights in the model during training

- After each epoch the loss will be calculated on the models predictions

Common Loss Functions:

- Mean Squared Error (MSE)
- Sparse Categorical Cross Entropy

**An in depth look**: Loss function is an important part in artificial neural networks, which is used to measure the inconsistency between predicted value (^yy^) and actual label (yy). It is a non-negative value, where the robustness of model increases along with the decrease of the value of loss function. Loss function is the hard core of empirical risk function as well as a significant component of structural risk function. Generally, the structural risk function of a model is consist of empirical risk term and regularization term, which can be represented as
θ∗=argminθL(θ)+λ⋅Φ(θ)
=argminθ1nn∑i=1L(y(i),^y(i))+λ⋅Φ(θ)
=argminθ1nn∑i=1L(y(i),f(x(i),θ))+λ⋅Φ(θ) where Φ(θ)Φ(θ) is the regularization term or penalty term, θθ is the parameters of model to be learned, f(⋅)f(⋅) represents the activation function and x(i)={x(i)1,x(i)2,…,x(i)m}∈Rmx(i)={x1(i),x2(i),…,xm(i)}∈Rm denotes the a training sample.

> Resources: https://isaacchanghau.github.io/post/loss_functions/

- Loss: What kind of loss to apply to the model. It should inherit from BaseLoss.
  This is a tensor of loss for the examples in the mini-batch.
  Tf.losses = Loss operations for use in neural networks.

> Resources: https://www.tensorflow.org/api_docs/python/tf/losses

## Which Youtube-8m Model do I choose?

Youtube-8m provides the following sample models to use for dataset in the Youtube-8m Frame Level format.

- LSTM
- DBofModel
- FrameLevelLogisticModel

INFO:tensorflow:training step 3320 | Loss: 0.02 Examples/sec: 2684.25 | Hit@1: 1.00 PERR: 1.00 GAP: 1.00

### LSTM Model

- Recurrent Neural Networks - a class of artificial neural network where connections between nodes form a directed graph along a sequence. This allows it to exhibit temporal dynamic behavior for a time sequence. Unlike feedforward neural networks, RNNs can use their internal state (memory) to process sequences of inputs.

Long Short Term Memory is a Recurrent Neural Network Model in Youtube -8m sample models.

```
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
Args:
model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
input features.
vocab_size: The number of classes in the dataset.
num_frames: A vector of length 'batch' which indicates the number of
frames for each video (before padding).
Returns:
A dictionary with a tensor containing the probability predictions of the
model in the 'predictions' key. The dimensions of the tensor are
'batch_size' x 'num_classes'.
```

> Resources: https://github.com/google/youtube-8m/blob/2c94ed449737c886175a5fff1bfba7eadc4de5ac/frame_level_models.py

### DBof Model

**DbofModel** or Deep Bag of Frame Model is the "bag-of-frames" approach (BOF), which encodes audio signals as the long-term statistical distribution of short-term spectral features, is commonly regarded as an effective and sufficient way to represent environmental sound recordings (soundscapes) since its introduction in an influential 2007 article. The present paper describes a concep-tual replication of this seminal article using several new soundscape datasets, with results strongly questioning the adequacy of the BOF approach for the task. We show that the good accuracy originally re-ported with BOF likely result from a particularly thankful dataset with low within-class variability, and that for more realistic datasets, BOF in fact does not perform significantly better than a mere one-point av-erage of the signal's features. Soundscape modeling, therefore, may not be the closed case it was once thought to be. Progress, we ar-gue, could lie in reconsidering the problem of considering individual acoustical events within each soundscape.

- The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.
- The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

```
Args:
model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
input features.
vocab_size: The number of classes in the dataset.
num_frames: A vector of length 'batch' which indicates the number of
frames for each video (before padding).
Returns:
A dictionary with a tensor containing the probability predictions of the
model in the 'predictions' key. The dimensions of the tensor are
'batch_size' x 'num_classes'.
```

> Resources: https://arxiv.org/pdf/1707.03296.pdf
> Bag of Words Model
> https://machinelearningmastery.com/gentle-introduction-bag-words-model/

### Frame Level Logistic Model

```
def create_model(self, model_input, vocab_size, num_frames, **unused_params):
"""Creates a model which uses a logistic classifier over the average of the
frame-level features.
This class is intended to be an example for implementors of frame level
models. If you want to train a model over averaged features it is more
efficient to average them beforehand rather than on the fly.
Args:
model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
input features.
vocab_size: The number of classes in the dataset.
num_frames: A vector of length 'batch' which indicates the number of
frames for each video (before padding).
Returns:
A dictionary with a tensor containing the probability predictions of the
model in the 'predictions' key. The dimensions of the tensor are
'batch_size' x 'num_classes'.
```

> Resources: https://arxiv.org/pdf/1706.08217.pdf
 > https://groups.google.com/forum/#!topic/youtube8m-users/0VWJPPXdjCU

## How should I format my audio for building a sound effect classifier that uses Audioset?

Both Video-Level models and Frame-Level models use a similar format when leveraging Audioset to train a model. Video-Level models is a utility of Frame-Level models that does an aggregated projection of the output features with a logarithmic function that calculates the probabilities.

As mentioned previously, the models provided by Youtube-m8 are RNNs.RNNs can use their internal state (memory) to process sequences of inputs. The RNN treats the vector of VGGish audio embeddings as a vector sequence for the model. For example, if the input sequence for an audio wave corresponding to a particular sound, the final target output at the end of the sequence may be a label classifying the sound. The network needs to understand memory and the progression of sound.

The Audioset dataset is manually annotated audio events that was collected from human labelers to probe the presences of specific audio classes in 10 second segments of YouTube videos. For formatting, the Audioset VGGish model operates on 0.96s log Mel spectogram patches extracted from 16kHz audio, and outputs a 128-dimensional embedding vector to be inputted into your RNN Model. Commonly, white noise is used as an audio augmenter to increase the size of an audio dataset. This is unnecessary as the random signal frequencies would skew the end vector value label results an RNN would need to use for classification.

## Audioset Model Comparison

For more on Audioset Model Comparison view the validation documentation.
