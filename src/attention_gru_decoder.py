# coding=utf-8
import keras
from keras.layers import *
import keras.backend as K


class AttentionGRUCell(GRUCell):
    def __init__(self, attention_units,
                 prev_units,
                 encoder_outputs,
                 style='additive',
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 reset_after=False,
                 **kwargs):
        """
        GRU decoder cell that implements an Attention mechanism for one step.

        `hidden_unit` for the decoder cell must be equal to encoder's `prev_units`, thus
        only  `prev_units` need to be assigned.

        # Usage:
            cell = AttentionGRUCell(...)

            decoder = RNN(cell, return_sequences=True, name='attention_gru')

            x = decoder(x, initial_state=[...])

        """
        super(AttentionGRUCell, self).__init__(prev_units,
                                               activation=activation,
                                               recurrent_activation=recurrent_activation,
                                               use_bias=use_bias,
                                               kernel_initializer=kernel_initializer,
                                               recurrent_initializer=recurrent_initializer,
                                               bias_initializer=bias_initializer,
                                               kernel_regularizer=kernel_regularizer,
                                               recurrent_regularizer=recurrent_regularizer,
                                               bias_regularizer=bias_regularizer,
                                               kernel_constraint=kernel_constraint,
                                               recurrent_constraint=recurrent_constraint,
                                               bias_constraint=bias_constraint,
                                               dropout=dropout,
                                               recurrent_dropout=recurrent_dropout,
                                               implementation=implementation,
                                               reset_after=reset_after,
                                               **kwargs)

        self.W_initializer = initializers.get(kernel_initializer)
        self.b_initializer = initializers.get(bias_initializer)

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)

        self.W_constraint = constraints.get(kernel_constraint)
        self.b_constraint = constraints.get(bias_constraint)

        self.use_bias = use_bias
        self.attention_units = attention_units
        self.prev_units = prev_units
        self.style = style
        self.encoder_outputs = encoder_outputs

    def build(self, input_shape):
        """
        input_shape == (batch_size, vec_dim).
        Input to cell is `K.concatenate([x, context_vector])`
        Thus input_shape should be (batch_size, vec_dim + hidden_units)
        """
        input_shape = (input_shape[0], input_shape[1]+self.prev_units)

        # Weights for attention calculation.
        # -------------------------------------------------
        if self.style == 'additive':
            self.V = self.add_weight((self.attention_units, 1),
                                      initializer=self.W_initializer,
                                      name='{}_V'.format(self.name),
                                      regularizer=self.W_regularizer,
                                      constraint=self.W_constraint)
        elif self.style == 'dot':
            pass
        else:
            raise ValueError('Style {} not supported.'.format(self.style))

        self.W1 = self.add_weight((self.prev_units, self.attention_units),
                                 initializer=self.W_initializer,
                                 name='{}_W1'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.W2 = self.add_weight((self.prev_units, self.attention_units),
                                  initializer=self.W_initializer,
                                  name='{}_W2'.format(self.name),
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)
        if self.use_bias:
            self.b1 = self.add_weight((self.attention_units,),
                                     initializer=self.b_initializer,
                                     name='{}_b1'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b2 = self.add_weight((self.attention_units,),
                                      initializer=self.b_initializer,
                                      name='{}_b2'.format(self.name),
                                      regularizer=self.b_regularizer,
                                      constraint=self.b_constraint)
            if self.style == 'additive':
                self.bv = self.add_weight((1,),
                                          initializer=self.b_initializer,
                                          name='{}_b3'.format(self.name),
                                          regularizer=self.b_regularizer,
                                          constraint=self.b_constraint)
        else:
            self.b1, self.b2, self.bv = None, None, None
        # --------------------------------------------------------
        super(AttentionGRUCell, self).build(input_shape)

    def call(self, inputs, states, training=None):
        """
        :param inputs: x
        :param states: [prev_hidden_state]
        :param training:
        :return: cell_output, [cell_output]
        """
        x = inputs
        # hidden shape == (batch_size, hidden_size)
        prev_hidden_state = states[0]
        hidden_with_time = K.expand_dims(prev_hidden_state, axis=1)
        # encoder_outputs shape == (batch_size, time_step, hidden_size)
        encoder_outputs = self.encoder_outputs

        # Calculate context vector.
        # ------------------------------------------
        if self.style == 'additive':
            # score shape == (batch_size, time_step, 1)
            score = K.bias_add(K.dot(encoder_outputs, self.W1), self.b1, 'channels_last') \
                    + K.bias_add(K.dot(hidden_with_time, self.W2), self.b2, 'channels_last')
            score = K.bias_add(K.dot(K.tanh(score), self.V), self.bv, 'channels_last')
        elif self.style == 'dot':
            # score shape == (batch_size, time_step, 1)
            # TODO: Not sure if this is implemented right...
            x1 = K.bias_add(K.dot(encoder_outputs, self.W1), self.b1, 'channels_last')
            x2 = K.bias_add(K.dot(hidden_with_time, self.W2), self.b2, 'channels_last')
            x2 = K.permute_dimensions(x2, (0, 2, 1))
            score = K.batch_dot(x1, x2)
        else:
            raise ValueError('Style {} not supported.'.format(self.style))

        # attention_weights shape == (batch_size, time_step, 1)
        attention_weights = K.softmax(score, axis=1)

        context_vector = attention_weights*encoder_outputs
        # Sum up along time axis
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = K.sum(context_vector, axis=1, keepdims=False)
        # ----------------------------------------------

        cell_input = K.concatenate([x, context_vector])
        cell_output, _ = super(AttentionGRUCell, self).call(cell_input, states, training)

        return cell_output, _

    def compute_output_shape(self, input_shape):
        return super(AttentionGRUCell, self).compute_output_shape(input_shape)


class AttentionGRUDecoder(object):
    def __init__(self, attention_units,
                 prev_units,
                 encoder_outputs,
                 return_sequences=False,
                 return_state=False,
                 bidirectional=False,
                 merge_mode='concat',
                 stateful=False):
        self.supports_masking = True
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.cell = AttentionGRUCell(attention_units=attention_units,
                                     prev_units=prev_units,
                                     encoder_outputs=encoder_outputs)

        self.rnn_forward = RNN(self.cell,
                               return_sequences=return_sequences,
                               return_state=return_state,
                               stateful=stateful,
                               name='foward_rnn')
        if bidirectional:
            self.rnn_backward = RNN(self.cell,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    go_backwards=True,
                                    stateful=stateful,
                                    name='backward_rnn')
        self.bidirectional = bidirectional

        self.supported_merge_layers = {'sum': Add, 'mul': Multiply, 'concat': Concatenate, 'ave': Average, None: None}

        self.supported_merge_mode = self.supported_merge_layers.keys()

        if merge_mode not in self.supported_merge_mode:
            raise ValueError('Merge mode {} not supported.'.format(merge_mode))
        else:
            self.merge_layer = self.supported_merge_layers[merge_mode]()

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        forward_rnn_outputs = self.rnn_forward(inputs, initial_state, constants, **kwargs)
        if self.return_state:
            forward_outputs, forward_final_state = forward_rnn_outputs

        if self.bidirectional:
            backward_rnn_outputs = self.rnn_backward(inputs, initial_state, constants, **kwargs)
            if self.return_state:
                backward_outputs, backward_final_state = backward_rnn_outputs
                merged_outputs = self.merge_layer([forward_outputs, backward_outputs])
                return merged_outputs, forward_final_state, backward_final_state
            else:
                merged_outputs = self.merge_layer([forward_rnn_outputs, backward_rnn_outputs])
                return merged_outputs

        elif self.return_state:
            return forward_outputs, forward_final_state
        else:
            return forward_rnn_outputs


if __name__ == '__main__':
    time_step = 32
    main_input = Input((time_step, 300))
    x = Masking()(main_input)
    encoder_outputs, final_state = GRU(64, return_sequences=True, return_state=True)(x)

    decoder = AttentionGRUDecoder(attention_units=16, prev_units=64, encoder_outputs=encoder_outputs,
                           return_sequences=True)
    outputs = decoder(x, initial_state=[final_state])
    outputs = TimeDistributed(Dense(40000, activation='softmax'))(outputs)
    model = keras.Model(inputs=main_input, outputs=outputs)
    model.compile('adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())






















