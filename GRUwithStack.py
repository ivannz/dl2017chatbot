import numpy as np
import theano
import theano.tensor as tt
from lasagne import init

from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers import helper


class GRUStackLayer(MergeLayer):
    def __init__(self, incoming,
                 stack, num_units,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        assert all(dim is not None for dim in stack.output_shape[1:])
        assert len(stack.output_shape) == 3

        incomings = [incoming, stack]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(GRUStackLayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        stack_shape = self.input_shapes[1]

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])
        stack_width = np.prod(stack_shape[2:])
        self.W_h = self.add_param(
            init.Normal(0.1), (num_units, 3 * num_units + stack_width + 3),
            name="W_hid", trainable=True, regularizable=True)
        self.W_x = self.add_param(
            init.Normal(0.1), (num_inputs, 3 * num_units),
            name="W_in", trainable=True, regularizable=True)
        self.W_s = self.add_param(
            init.Normal(0.1), (stack_width, 3 * num_units),
            name="W_top", trainable=True, regularizable=True)
        self.b = self.add_param(
            init.Constant(0.0), (3 * num_units,),
            name="b", trainable=True, regularizable=False)
        self.b_sh = self.add_param(
            init.Constant(0.0), (stack_width + 3,),
            name="b_sh", trainable=True, regularizable=False)

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]

        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input, stack = inputs[0], inputs[1]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = tt.flatten(input, 3)
        if stack.ndim > 3:
            stack = tt.flatten(stack, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.swapaxes(1, 0)
        seq_len, num_batch, _ = input.shape

        W_x, b = self.W_x, self.b
        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = tt.dot(input, W_x) + b

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def gru_stack_step(x_t, h_tm1, s_tm1, W_s, b_sh, W_h, *args):
            def s_(x, i, n=self.num_units):
                s = x[..., slice(i * n, (i + 1) * n)]
                return s if n > 1 else tt.addbroadcast(s, -1)

            # 3 * num_units + stack_width + 3)
            hid_input = tt.dot(h_tm1, W_h)
            if self.grad_clipping:
                x_t = theano.gradient.grad_clip(
                    x_t, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                x_t = tt.dot(x_t, W_x) + b

            # Stack update: get actions and data. hid_stack is `Bx(W+3)`
            hid_stack = hid_input[..., 3 * self.num_units:] + b_sh

            # a_t is `Bx3`, d_t is `BxW`
            a_t = tt.nnet.softmax(hid_stack[..., -3:]).dimshuffle(0, 1, "x", "x")
            d_t = tt.tanh(hid_stack[..., :-3]).dimshuffle(0, "x", 1)

            # Differential stack update, s_tm1 is `BxDxW`
            wilderness = tt.zeros_like(s_tm1[:, 0]).dimshuffle(0, "x", 1)
            s_pop = tt.concatenate([s_tm1[:, 1:], wilderness], axis=1)
            s_push = tt.concatenate([d_t, s_tm1[:, :-1]], axis=1)
            s_t = a_t[:, 0] * s_tm1 + a_t[:, 1] * s_push + a_t[:, 2] * s_pop
            stack_input = tt.dot(s_t[:, 0], W_s)

            # the regular gru step. hid_input is `Bx(3*N)`, s_t is `BxDxW`
            hid_input = hid_input[..., :3 * self.num_units]

            r_t = tt.nnet.sigmoid(s_(x_t, 0) + s_(stack_input, 0) + s_(hid_input, 0))
            z_t = tt.nnet.sigmoid(s_(x_t, 1) + s_(stack_input, 1) + s_(hid_input, 1))
            hat_t = s_(x_t, 2) + s_(stack_input, 2) + s_(hid_input, 2) * r_t
            if self.grad_clipping:
                hat_t = theano.gradient.grad_clip(
                    hat_t, -self.grad_clipping, self.grad_clipping)

            h_t = (1 - z_t) * h_tm1 + z_t * tt.tanh(hat_t)
            return h_t, s_t

        def gru_mask_stack_step(x_t, m_t, h_tm1, s_tm1, *args):
            h_t, s_t = gru_stack_step(x_t, h_tm1, s_tm1, *args)
            h_t = tt.switch(m_t, h_t, h_tm1)
            # m_t has dimensions Bx1
            s_t = tt.switch(m_t.dimshuffle(0, 1, 'x'), s_t, s_tm1)
            return h_t, s_t

        sequences, fn_step = [input], gru_stack_step
        if mask is not None:
            # mask has (original) dimensions BxS
            sequences.append(mask.dimshuffle(1, 0, 'x'))
            fn_step = gru_mask_stack_step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = tt.dot(tt.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [self.W_s, self.b_sh, self.W_h]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [self.W_x, self.b]

        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        result, updates = theano.scan(fn=fn_step, sequences=sequences,
                                      non_sequences=non_seqs, return_list=True,
                                      outputs_info=[hid_init, stack],
                                      strict=True, go_backwards=self.backwards,
                                      truncate_gradient=self.gradient_steps,
                                      name="{}_scan".format(self.name))
        hid_out, stack_out = result

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
            stack_out = stack_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.swapaxes(1, 0)
            stack_out = stack_out.swapaxes(1, 0)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]
                stack_out = stack_out[:, ::-1]

        self.symbolic_stack_out = stack_out
        return hid_out


class GRUStackReadoutLayer(Layer):
    def __init__(self, incoming, **kwargs):
        assert isinstance(incoming, GRUStackLayer)
        super(GRUStackReadoutLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        """
        Un-broadcasts the broadcast layer (see class description)
        :param input: input tensor
        :param kwargs: no effect
        :return: un-broadcasted tensor
        """
        return self.input_layer.symbolic_stack_out

    def get_output_shape_for(self, input_shape, **kwargs):
        stack_shape = self.input_layer.input_shapes[1]

        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        shape = [input_shape[0]]
        if not self.input_layer.only_return_final:
            shape.append(input_shape[1])

        shape.extend([stack_shape[1], stack_shape[2]])

        return tuple(shape)
