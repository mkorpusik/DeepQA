# Copyright 2015 Conchylicultor. All Rights Reserved.
# Modifications copyright (C) 2016 Carlos Segura
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf

from chatbot.textdata import Batch
from chatbot.decoders import *


class ProjectionOp:
    """ Single layer perceptron
    Project input tensor on the output dimension
    """
    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[1],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class Model:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, args, textData):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        print("Model creation...")

        self.textData = textData  # Keep a reference on the dataset
        self.args = args  # Keep track of the parameters of the model
        self.dtype = tf.float32

        # Placeholders
        self.encoderInputs  = None
        self.decoderInputs  = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size
        self.decoderContext = None

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        # Construct the graphs
        self.buildNetwork()

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Parameters of sampled softmax (needed for attention mechanism and a large vocabulary size)
        outputProjection = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.args.softmaxSamples < self.textData.getVocabularySize():
            outputProjection = ProjectionOp(
                (self.args.hiddenSize, self.textData.getVocabularySize()),
                scope='softmax_projection',
                dtype=self.dtype
            )

            def sampledSoftmax(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                localWt     = tf.cast(tf.transpose(outputProjection.W), tf.float32)
                localB      = tf.cast(outputProjection.b,               tf.float32)
                localInputs = tf.cast(inputs,                           tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=localWt,  # Should have shape [num_classes, dim]
                        biases=localB,
                        inputs=localInputs,
                        labels=labels,
                        num_sampled=self.args.softmaxSamples,  # The number of classes to randomly sample per batch
                        num_classes=self.textData.getVocabularySize()),  # The number of classes
                    self.dtype)

        # Creation of the rnn cell
        encoDecoCell = tf.contrib.rnn.BasicLSTMCell(self.args.hiddenSize, state_is_tuple=bool(not self.args.beam_search))  # Or GRUCell, LSTMCell(args.hiddenSize)
        #encoDecoCell = tf.contrib.rnn.DropoutWrapper(encoDecoCell, input_keep_prob=1.0, output_keep_prob=1.0)  # TODO: Custom values (WARNING: No dropout when testing !!!)
        encoDecoCell = tf.contrib.rnn.MultiRNNCell([encoDecoCell] * self.args.numLayers, state_is_tuple=bool(not self.args.beam_search))

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.maxLengthDeco)]

            if self.args.corpus == 'healthy-comments':
                self.decoderContext = [tf.placeholder(tf.float32, [None, 64,], name='context') for _ in range(self.args.maxLengthDeco)]

        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation
        if self.args.attention:
            rnn_model = embedding_attention_seq2seq
            #rnn_model = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq
        elif self.args.food_context:
            rnn_model = embedding_attention_context_seq2seq
        else:
            rnn_model = embedding_rnn_seq2seq
            #rnn_model = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq


        if self.args.food_context:
            decoderOutputs, states, beamPath, beamSymbols, beamProbs = rnn_model(
                self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
                self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
                self.decoderContext,
                encoDecoCell,
                self.textData.getVocabularySize(),
                self.textData.getVocabularySize(),  # Both encoder and decoder have the same number of class
                embedding_size=self.args.embeddingSize,  # Dimension of each word
                output_projection=outputProjection.getWeights() if outputProjection else None,
                feed_previous=bool(self.args.test),  # When we test (self.args.test), we use previous output as next input (feed_previous)
                first_step=self.args.first_step,
                beam_search=bool(self.args.beam_search),
                beam_size=self.args.beam_size
            )
        else:
            decoderOutputs, states, beamPath, beamSymbols, beamProbs = rnn_model(
                self.encoderInputs,
                self.decoderInputs,
                encoDecoCell,
                self.textData.getVocabularySize(),
                self.textData.getVocabularySize(),
                embedding_size=self.args.embeddingSize,
                output_projection=outputProjection.getWeights() if outputProjection else None,
                feed_previous=bool(self.args.test),
                beam_search=bool(self.args.beam_search),
                beam_size=self.args.beam_size
            )
            print(len(decoderOutputs))

        # For testing only
        if self.args.test:
            self.outputs = decoderOutputs
            if self.args.beam_search:
                self.outputs.append(beamPath)
                self.outputs.append(beamSymbols)
                self.outputs.append(beamProbs)
            elif self.args.attention or self.args.food_context:
                self.outputs = [outputProjection(out) for out in decoderOutputs]
            
            # TODO: Attach a summary to visualize the output

        # For training only
        else:
            # Finally, we define the loss function
            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.textData.getVocabularySize(),
                softmax_loss_function= sampledSoftmax if outputProjection else None  # If None, use default SoftMax
            )
            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)

    def step(self, batch, match_encoder_decoder_input=False):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}
        ops = None

        if not self.args.test:  # Training
            if not self.args.finetune:
                for i in range(self.args.maxLengthEnco):
                    feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]
                if self.args.corpus == 'healthy-comments':
                    feedDict[self.decoderContext[i]] = batch.contextSeqs[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            if match_encoder_decoder_input:
                # use encoder input as decoder input
                for i in range(self.args.maxLengthDeco):
                    feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i]
            else:
                feedDict[self.decoderInputs[0]]  = [self.textData.goToken]
                #print('decoder input size', len(batch.decoderSeqs[i]), batch.decoderSeqs[i], self.textData.goToken)
                if self.args.corpus == 'healthy-comments':
                    for i in range(self.args.maxLengthDeco):
                        #print('context size', len(batch.contextSeqs[i]), batch.contextSeqs[i])
                        #print(i, batch.contextSeqs[i])
                        feedDict[self.decoderContext[i]] = batch.contextSeqs[i]

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
