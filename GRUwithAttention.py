"""GrU layer with input attention."""
import numpy as np
import theano

import theano.tensor as tt
from lasagne import init

from lasagne.layers.base import MergeLayer, Layer


class GRUAttentionLayer(MergeLayer):
    def __init__(self, incoming, context,
                 num_units, num_attn_units,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 precompute_input=True,
                 mask_input=None,
                 context_mask_input=None,
                 only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming, context]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if context_mask_input is not None:
            incomings.append(context_mask_input)
            self.context_mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(GRUAttentionLayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.num_attn_units = num_attn_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        ctx_shape = self.input_shapes[1]

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])
        ctx_width = np.prod(ctx_shape[2:])  # ctx is BxSxW

        self.W_x = self.add_param(
            init.Normal(0.1), (num_inputs, 3 * num_units),
            name="W_in", trainable=True, regularizable=True)
        self.W_c = self.add_param(
            init.Normal(0.001), (ctx_width, 3 * num_units + num_attn_units),
            name="W_ctx", trainable=True, regularizable=True)
        self.W_h = self.add_param(
            init.Normal(0.1), (num_units, 3 * num_units + num_attn_units),
            name="W_hid", trainable=True, regularizable=True)
        self.b = self.add_param(
            init.Constant(0.0), (3 * num_units,),
            name="b", trainable=True, regularizable=False)
        self.v_attn = self.add_param(
            init.Constant(0.0), (num_attn_units,),
            name="v_attn", trainable=True, regularizable=False)
        self.b_attn = self.add_param(
            init.Constant(0.0), (num_attn_units,),
            name="b_attn", trainable=True, regularizable=False)

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
        input, context = inputs[0], inputs[1]
        # Retrieve the mask when it is supplied
        mask, context_mask = None, None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.context_mask_incoming_index > 0:
            context_mask = inputs[self.context_mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = tt.flatten(input, 3)

        if context.ndim > 3:
            context = tt.flatten(context, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.swapaxes(1, 0)
        seq_len, num_batch, _ = input.shape

        W_x, b = self.W_x, self.b
        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = tt.dot(input, W_x) + b

        # the context product can be precomputed
        context = tt.dot(context, self.W_c)  # BxSx(3 * N + A)
        if self.grad_clipping:
            context = theano.gradient.grad_clip(
                context, -self.grad_clipping, self.grad_clipping)

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def gru_attn_step(x_t, h_tm1, ctx, ctx_mask, v_a, b_a, W_h, *args):
            n_units = self.num_units

            def s_(x, i, n=n_units):
                s = x[..., slice(i * n, (i + 1) * n)]
                return s if n > 1 else tt.addbroadcast(s, -1)

            # 3*N + A
            hid_input = tt.dot(h_tm1, W_h)

            if self.grad_clipping:
                x_t = theano.gradient.grad_clip(
                    x_t, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                x_t = tt.dot(x_t, W_x) + b

            # # Attention mechanism
            # first term is `Bx1xA` and the second -- `BxSxA`
            sl_ = np.s_[3 * n_units: 3 * n_units + self.num_attn_units]
            attn = hid_input[..., sl_].dimshuffle(0, "x", 1) + ctx[..., sl_]

            # the attention model's output is `BxS`, ctx_mask is `BxS`
            alpha = tt.nnet.softmax(tt.dot(tt.tanh(attn + b_a), v_a)) * ctx_mask
            alpha /= alpha.sum(axis=1, keepdims=True)

            # ctx_input becomes `BxSx(3*N)`
            ctx = ctx[..., :3 * n_units] * alpha.dimshuffle(0, 1, "x")

            # sum-reduce along the sequence dimension: `Bx(3*N)`
            ctx = ctx.sum(axis=1)

            # # the regular gru step. hid_input is `Bx(3*N)`, ctx is `BxDxW`
            hid_input = hid_input[..., :3 * n_units]

            r_t = tt.nnet.sigmoid(s_(x_t, 0) + s_(ctx, 0) + s_(hid_input, 0))
            z_t = tt.nnet.sigmoid(s_(x_t, 1) + s_(ctx, 1) + s_(hid_input, 1))
            hat_t = s_(x_t, 2) + s_(ctx, 2) + s_(hid_input, 2) * r_t
            if self.grad_clipping:
                hat_t = theano.gradient.grad_clip(
                    hat_t, -self.grad_clipping, self.grad_clipping)

            h_t = (1 - z_t) * h_tm1 + z_t * tt.tanh(hat_t)
            return h_t

        def gru_mask_attn_step(x_t, m_t, h_tm1, *args):
            h_t = gru_attn_step(x_t, h_tm1, *args)

            h_t = tt.switch(m_t, h_t, h_tm1)
            return h_t

        sequences, fn_step = [input], gru_attn_step
        if mask is not None:
            # mask has (original) dimensions BxS
            sequences.append(mask.dimshuffle(1, 0, 'x'))
            fn_step = gru_mask_attn_step

        if context_mask is None:
            context_mask = tt.ones((input.shape[0], input.shape[1]), "bool")

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = tt.dot(tt.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [context, context_mask, self.v_attn, self.b_attn, self.W_h]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [self.W_x, self.b]

        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        result, updates = theano.scan(fn=fn_step, sequences=sequences,
                                      non_sequences=non_seqs, return_list=True,
                                      outputs_info=[hid_init],
                                      strict=True, go_backwards=self.backwards,
                                      truncate_gradient=self.gradient_steps,
                                      name="{}_scan".format(self.name))
        hid_out = result[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.swapaxes(1, 0)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out
