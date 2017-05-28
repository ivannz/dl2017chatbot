import logging
import os
import pickle
import signal

import numpy as np

import theano
theano.config.exception_verbosity = 'high'

import theano.tensor as tt

import lasagne
from lasagne.layers import DenseLayer, EmbeddingLayer
from lasagne.layers import GRULayer, InputLayer
from lasagne.layers import SliceLayer, ConcatLayer
from lasagne.utils import floatX

from lasagne.layers.base import Layer

from telegram.ext import CommandHandler, Filters, Job
from telegram.ext import MessageHandler, Updater


# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)


# Uninterruptible section
class DelayedKeyboardInterrupt(object):
    """Create an atomic section with respect to the Keyboard Interrupt."""
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        """Delayed KeyboardInterrupt handler."""
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


###############################################################################
#                           UNSERIALIZE THE NETWORK                           #
###############################################################################
# Load the vocbulary and weights
model_file = None
with open(model_file, "rb") as fin:
    ver, *rest = pickle.load(fin)

assert ver == "DIALOGUE"
hyper, vocab, weights = rest


###############################################################################
#                         VOCABULARY AND ARCHITECTURE                         #
###############################################################################
# Define the network architecture
n_embed_char = hyper["n_embed_char"]              # 32
n_hidden_encoder = hyper["n_hidden_encoder"]      # 256
n_hidden_decoder = hyper["n_hidden_decoder"]      # 512
n_recurrent_layers = hyper["n_recurrent_layers"]  # 2
b_xfeed = hyper["b_xfeed"]                        # False
b_project = hyper["b_project"]                    # True

# Translate the characters into vocabulary IDs
token_to_index = {w: i for i, w in enumerate(vocab)}


def as_tensor3(dialogues, max_dialogue_len=None, max_seq_len=None):
    if isinstance(dialogues, list) and isinstance(dialogues[0], str):
        dialogues = [dialogues]

    max_dialogue_len = max_dialogue_len or max(map(len, dialogues))
    max_seq_len = max_seq_len or max(max(map(len, lines)) for lines in dialogues)

    tensor = np.full((len(dialogues), max_dialogue_len, max_seq_len), -1, dtype='int32')
    for i, dialogue in enumerate(dialogues):
        for j, line in enumerate(dialogue):
            row_ix = [token_to_index.get(c, -1) for c in line[:max_seq_len]]
            tensor[i, j, :len(row_ix)] = row_ix

    return tensor


###############################################################################
#                              BASE COLUMN BLOCK                              #
###############################################################################
# A helper to create stacked RNN
def gru_column(input, num_units, hidden, **kwargs):
    kwargs.pop("only_return_final", None)
    assert isinstance(hidden, (list, tuple))

    name = kwargs.pop("name", "default")
    column = [input]
    for i, l_hidden in enumerate(hidden):
        kwargs_ = kwargs.copy()
        if isinstance(l_hidden, Layer):
            kwargs_.pop("learn_init", None)
            kwargs_["hid_init"] = l_hidden

        layer = GRULayer(column[-1], num_units,
                         name=os.path.join(name, "gru_%02d" % i),
                         **kwargs_)
        column.append(layer)
    return column[1:]


# Create readouts of the last hidden state
def gru_hidden_readout(column, indices):
    hidden = []
    for layer in column:
        name = os.path.join(layer.name, "slice")
        slice_ = SliceLayer(layer, indices, axis=1, name=name)
        hidden.append(slice_)
    return hidden


###############################################################################
#                              SENTENCE ENCODER                               #
###############################################################################
# Encoder's Recurrent subnetwork
l_encoder_mask = InputLayer((None, None, None), name="encoder/mask")
l_encoder_embed = InputLayer((None, None, None, n_embed_char), name="encoder/input")

# Flatten the first two axes of the embedding output
l_encoder_mask_reshaped = BroadcastLayer(l_encoder_mask, broadcasted_axes=(0, 1),
                                         name="encoder/mask/broadcast")
l_encoder_embed_reshaped = BroadcastLayer(l_encoder_embed, broadcasted_axes=(0, 1),
                                          name="encoder/embed/broadcast")

hidden = n_recurrent_layers * [None]
enc_rnn_layers = gru_column(l_encoder_embed_reshaped, n_hidden_encoder, hidden,
                            mask_input=l_encoder_mask_reshaped, learn_init=True,
                            backwards=False, name="encoder")

enc_rnn_layers_reshaped_sliced = gru_hidden_readout(enc_rnn_layers, -1)

# Unbroadcast the slices
enc_rnn_layers_sliced = []
for layer in enc_rnn_layers_reshaped_sliced:
    enc_rnn_layers_sliced.append(
        UnbroadcastLayer(layer, l_encoder_mask_reshaped,
                         name=os.path.join(layer.name, "unbroadcast")))


###############################################################################
#                              DIALOGUE ENCODER                               #
###############################################################################
# This subnetwork runs a recurrent stack augmented GRU along the
#  dialog dimension.
l_stack_aug_mask = InputLayer((None, None), name="dialogues/mask")
l_stack_aug_cat = ConcatLayer(enc_rnn_layers_sliced, axis=-1, name="dialogues/cat")

# Add two layers of GRU-s
hidden = n_stack_layers * [None]
dialogue_rnn_layers = gru_column(l_stack_aug_cat, n_hidden_stack, hidden,
                                 mask_input=l_stack_aug_mask, learn_init=True,
                                 backwards=False, name="dialogues")

dialogue_rnn_layers_sliced = gru_hidden_readout(dialogue_rnn_layers, -1)


###############################################################################
#                                   DECODER                                   #
###############################################################################
# Tap into the common embedding layer but with decoder's own input.
l_decoder_mask = InputLayer((None, None), name="decoder/mask")
l_decoder_embed = InputLayer((None, None, n_embed_char), name="decoder/input")

# Project the hidden state of the encoder
dec_hid_inputs = []
for layer in dialogue_rnn_layers_sliced:
    l_project = DenseLayer(layer, n_hidden_decoder, nonlinearity=None,
                           name=os.path.join(layer.name, "proj"))
    dec_hid_inputs.append(l_project)

# Construct layers of GRU-s which recieve the final state of the encoder's network.
dec_rnn_layers = gru_column(l_decoder_embed, n_hidden_decoder, dec_hid_inputs,
                            mask_input=l_decoder_mask, learn_init=True,
                            backwards=False, name="decoder")

dec_rnn_layers_sliced = gru_hidden_readout(dec_rnn_layers, -1)

l_decoder_reembedder = DenseLayer(dec_layers[-1], num_units=len(vocab),
                                  nonlinearity=None, num_leading_axes=2,
                                  name="decoder/project")

lasagne.layers.set_all_param_values(l_decoder_reembedder,
                                    weights["l_decoder_reembedder"])


###############################################################################
#                              COMMON EMBEDDING                               #
###############################################################################
# Common embedding subnetwork
l_input_char = InputLayer((None, None), name="char/input")
l_embed_char = EmbeddingLayer(l_input_char, len(vocab),
                              n_embed_char, name="char/embed")

lasagne.layers.set_all_param_values(l_embed_char, weights["l_embed_char"])


###############################################################################
#                                  GENERATOR                                  #
###############################################################################
# Helper functions to freeze the GRULayer's hidden input's initialization, if one is a parameter.
def GRULayer_freeze(layer, input):
    assert isinstance(layer, GRULayer)
    if isinstance(layer.hid_init, Layer):
        return layer

    assert not (layer.hid_init_incoming_index > 0)
    assert isinstance(layer.hid_init, theano.compile.SharedVariable)

    # Broadcast the fixed /learnt hidden init over the batch dimension
    hid_init = tt.dot(tt.ones((input.shape[0], 1)), layer.hid_init)

    # Create a fake Input Layer, which receives it as input
    layer._old_hid_init = layer.hid_init
    layer.hid_init = InputLayer((None, None), input_var=hid_init,
                                name=os.path.join(layer.name,
                                                  "hid_init_fix"))
    
    # Cache former values
    layer._old_input_layers = layer.input_layers
    layer._old_input_shapes = layer.input_shapes
    layer._old_hid_init_incoming_index = layer.hid_init_incoming_index
    
    # Emulate hidden layer input (is in GRULayer/MergeLayer.__init__())
    layer.input_layers.append(layer.hid_init)
    layer.input_shapes.append(layer.hid_init.output_shape)
    layer.hid_init_incoming_index = len(layer.input_layers) - 1

    layer._layer_frozen = True
    return layer


# A handy slicer (copied and modified)
def slice_(x, i, n):
    s = x[..., slice(i, i + n)]
    return s if n > 1 else tt.addbroadcast(s, -1)


# Generator's one step update function
def generator_step_sm(x_tm1, h_tm1, m_tm1, tau, eps):
    """One step of the generative decoder version."""
    # x_tm1 is `BxT` one-hot, h_tm1 is `batch x ...`
    # m_tm1 is `batch`, tau, eps are scalars

    # collect the inputs
    inputs = {l_decoder_embed: x_tm1.dimshuffle(0, "x", 1),
              l_decoder_mask: m_tm1.dimshuffle(0, "x")}

    # Connect the prev variables to the the hidden and stack state feeds
    j = 0
    for layer in dec_rnn_layers:
        inputs[layer.hid_init] = slice_(h_tm1, j, layer.num_units)
        j += layer.num_units

    # Get the outputs
    outputs = [l_decoder_reembedder] + dec_rnn_layers_sliced

    # propagate through the decoder column
    logit_t, *h_t_list = lasagne.layers.get_output(outputs, inputs,
                                                   deterministic=True)

    # Pack the hidden states
    h_t = tt.concatenate(h_t_list, axis=-1)
    
    # Generate the next symbol: logit_t is `Bx1xV`
    logit_t = logit_t[:, 0]
    prob_t = tt.nnet.softmax(logit_t)

    # Gumbel-softmax sampling: Gumbel (e^{-e^{-x}}) distributed random noise
    gumbel = -tt.log(-tt.log(theano_random_state.uniform(size=logit_t.shape) + eps) + eps)
#     logit_t = theano.ifelse.ifelse(tt.gt(tau, 0), gumbel + logit_t, logit_t)
#     inv_temp = theano.ifelse.ifelse(tt.gt(tau, 0), 1.0 / tau, tt.constant(1.0))
    logit_t = tt.switch(tt.gt(tau, 0), gumbel + logit_t, logit_t)
    inv_temp = tt.switch(tt.gt(tau, 0), 1.0 / tau, tt.constant(1.0))

    # Get the softmax: x_t is `BxV`
    x_t = tt.nnet.softmax(logit_t * inv_temp)

    # Get the best symbol
    c_t = tt.cast(tt.argmax(x_t, axis=-1), "int8")

    # Get the estimated probability of the picked symbol.
    p_t = prob_t[tt.arange(c_t.shape[0]), c_t]

    # Compute the mask and inhibit the propagation on a stop symbol.
    # Recurrent layers return the previous state if m_tm1 is Fasle
    m_t = m_tm1 & tt.gt(c_t, vocab.index("\x03"))
    c_t = tt.switch(m_t, c_t, vocab.index("\x03"))

    # There is no need to freeze the states as they will be frozen by
    # the RNN passthrough according to the mask `m_t`.

    # Embed the current character.
    x_t = tt.dot(x_t, l_embed_char.W)

    return x_t, h_t, m_t, p_t, c_t

# Create scalar inputs to the scan loop. Also initialize the random stream.
theano_random_state = tt.shared_randomstreams.RandomStreams(seed=0xDEADC0DE)

eps = tt.fscalar("generator/epsilon")
n_steps = tt.iscalar("generator/n_steps")
tau = tt.fscalar("generator/gumbel/tau")

# Generator's input variables for the Encoder
v_gen_input = tt.imatrix(name="generator/input")

# Generator's embedding subnetwork readout for the Encoder
v_gen_embed = lasagne.layers.get_output(l_embed_char, v_gen_input)

# Freeze the hidden inputs of the decoder layers, which do not tap into the encoder.
for layer in dec_rnn_layers:
    GRULayer_freeze(layer, v_gen_input)

# Readout the last state from the encoder.
inputs = {l_encoder_embed: v_gen_embed,
          l_encoder_mask: tt.ge(v_gen_input, 0)}
inputs[l_stack_aug_mask] = tt.gt(tt.sum(inputs[l_encoder_mask], axis=-1), 0)

outputs = [l.hid_init for l in dec_rnn_layers]

dec_hid_inits = lasagne.layers.get_output(outputs, inputs,
                                          deterministic=True)

# Prepare the initial values fed into the scan loop of the Generator
h_0 = tt.concatenate(dec_hid_inits, axis=-1)

x_0 = tt.fill(tt.zeros((v_gen_input.shape[0],), dtype="int32"),
              vocab.index("\x02"))
x_0 = lasagne.layers.get_output(l_embed_char, x_0)

m_0 = tt.ones((v_gen_input.shape[0],), 'bool')

# Compile the Generator's scan op
result, updates = theano.scan(generator_step_sm, sequences=None, n_steps=n_steps,
                              outputs_info=[x_0, h_0, m_0, None, None],
                              strict=False, return_list=True,
                              non_sequences=[tau, eps], go_backwards=False,
                              name="generator/scan")
x_t, h_t, m_t, p_t, c_t = [r.swapaxes(0, 1) for r in result]

# Use `FAST_RUN` optimizations with C Virtual Machine.
compile_mode = theano.Mode(optimizer="fast_run", linker="cvm")
op_generate = theano.function([v_gen_input, n_steps, tau],
                              [c_t, h_t, m_t, p_t],
                              updates=updates, givens={eps: floatX(1e-20)},
                              mode=compile_mode)

# A generator procedure, which automatically select the best replies (lowest perplexity).
def generate(question, n_steps, n_samples=10, tau=0, seed=None):
    """Generate the best reply from the Generator."""
    question = np.repeat(question, n_samples, axis=0)
    if seed is not None:
        theano_random_state.seed(seed)

    x_t, h_t, m_t, p_t = op_generate(question, n_steps, tau)

    # may produce NaN, but they are shifted in the back by arsort
    perplexity, n_chars = (- np.log2(p_t) * m_t).sum(axis=-1), m_t.sum(axis=-1)
    perplexity /= n_chars

    result = []
    for i in perplexity.argsort():
        reply = "".join(map(vocab.__getitem__, x_t[i, :n_chars[i]]))
        result.append((reply, perplexity[i]))
            
    return result


###############################################################################
#                         MULTIDIALOG CONTEXT MANAGER                         #
###############################################################################
context_db = {}
MAX_USERS = 15

def alloc_context(chat_id):
    global context_db

    if (chat_id not in context_db) and (len(context_db) >= MAX_USERS):
        raise RuntimeError("""Too many connections(%d). Service refused.""" %(len(context_db)))

    return context_db.setdefault(chat_id, [])


def free_context(chat_id):
    global context_db

    context_db.pop(chat_id, None)


###############################################################################
#                             TELEGRAM BOT HOOKS                              #
###############################################################################
# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def error(bot, update, error):
    logger.warn('Update "%s" caused error "%s"' % (update, error))


def start(bot, update):
    name = update.message.from_user.first_name
    update.message.reply_text("Hello {}!".format(name))


def info(bot, update, args):
    update.message.reply_text("%r" % context_db)


def reset(bot, update, args):
    if update.message.chat_id != 191745228:
        update.message.reply_texte("you are not the Creator.")
        return

    theano_random_state.seed(seed=0xDEADC0DE)
    # update.message.reply_text("%r" % args)


def reply(bot, update):
    chat_id = update.message.chat_id
    try:
        context = alloc_context(chat_id)
        context.append("\x02" + update.message.text[:512] + "\x03")

        query = as_tensor3(context, max_dialogue_len=None, max_len=None)

        results = generate(query, 200, tau=2**-5, n_samples=25)
        reply, perplexity = results[0]

        context.append("\x02" + reply + "\x03")
        context_db[chat_id] = context[-9:]

        text = "[%.3g] %s" % (perplexity, reply)

    except Exception as e:
        text = "%s %s" % (e.__class__.__name__, str(e))

    update.message.reply_text(text)


def main():
    # Create the EventHandler and pass it your bot's token.
    updater = Updater("<<TOKEN>>")

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("reset", reset, pass_args=True))
    dp.add_handler(CommandHandler("info", info, pass_args=True))

    # on noncommand i.e message - reply to the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, reply))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.

    updater.idle()


if __name__ == '__main__':
    logger.info("Ready!")
    main()
