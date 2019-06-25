import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, cuda, max_seq_len=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        positional_encodings = torch.zeros(max_seq_len, embedding_dim).to(cuda)

        for seq_position in range(max_seq_len):
            for embedding_position in range(0, embedding_dim, 2):
                positional_encodings[seq_position, embedding_position] = math.sin(
                    seq_position / (10000 ** ((2 * embedding_position) / embedding_dim)))
                positional_encodings[seq_position, embedding_position + 1] = math.cos(
                    seq_position / (10000 ** ((2 * (embedding_position + 1) / embedding_dim))))

        self.positional_encodings = positional_encodings

    def forward(self, x):
        # Input is embedded input
        seq_len = x.size(1)

        # Scale embeddings, preserve original meaning
        x = x * self.embedding_dim ** 0.5
        x = x + self.positional_encodings.unsqueeze(0)[:, :seq_len]

        return x


class SequenceEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        embedding = self.embedding_layer(x)

        return embedding


class NormLayer(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.alpha = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True) /
                             (x.std(dim=-1, keepdim=True) + self.eps)) + self.bias

        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        # QKV Dimension = Embedding Dimension / Number Heads
        self.qkv_dim = embedding_dim // num_heads

        # Batch Size x Sequence Length x Embedding Dimension
        self.query_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key_linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value_linear = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.output_linear = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        sequence_size = q.size(1)
        self.qkv_dim = self.embedding_dim // self.num_heads

        # Batch Size x Sequence Length x Heads x QKV Dimension
        q = self.query_linear(q).view(
            batch_size, sequence_size, self.num_heads, self.qkv_dim)
        k = self.key_linear(k).view(
            batch_size, sequence_size, self.num_heads, self.qkv_dim)
        v = self.key_linear(v).view(
            batch_size, sequence_size, self.num_heads, self.qkv_dim)

        # Batch Size x Heads x Sequence Length x QKV Dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        multi_head_attention = self.attention(q, k, v, mask)
        multi_head_attention = multi_head_attention.transpose(
            1, 2).contiguous()

        concat_head_attention = multi_head_attention.view(
            batch_size, sequence_size, self.embedding_dim)

        mh_output = self.output_linear(concat_head_attention)

        return mh_output

    def attention(self, Q, K, V, mask_padding):
        qkv_dim = self.qkv_dim

        a = torch.matmul(Q, K.transpose(-2, -1)) / qkv_dim ** 0.5

        if mask_padding is not None:
            a = a.masked_fill(mask_padding == 0, -1e9)

            a_softmax = F.softmax(a, dim=-1)
            a_softmax = a_softmax.masked_fill(mask_padding == 0, 0)

        else:
            a_softmax = F.softmax(a, dim=-1)

        return torch.matmul(a_softmax, V)


class PositionWiseFFN(nn.Module):
    def __init__(self, embedding_dim, ff_dim):
        super().__init__()
        self.ff_dim = ff_dim

        self.ff1 = nn.Linear(embedding_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, ff_dim)
        self.ff3 = nn.Linear(ff_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.ff1(x))
        x = F.relu(self.ff2(x))
        x = F.relu(self.ff3(x))

        return x


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim):
        super().__init__()
        self.mh_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ffc = PositionWiseFFN(embedding_dim, ff_dim)
        self.norm1 = NormLayer(embedding_dim)
        self.norm2 = NormLayer(embedding_dim)

    def forward(self, x):
        x, mask = x[0], x[1]

        x = self.mh_attention(x, x, x, mask=mask)
        x = x + self.norm1(x)
        ff_x = self.ffc(x)
        x = x + self.norm2(ff_x)

        return {0: x, 1: mask}


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim):
        super().__init__()

        self.masked_mh_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.mh_attention = MultiHeadAttention(embedding_dim, num_heads)

        self.ffc = PositionWiseFFN(embedding_dim, ff_dim)
        self.norm1 = NormLayer(embedding_dim)
        self.norm2 = NormLayer(embedding_dim)
        self.norm3 = NormLayer(embedding_dim)

    def forward(self, x):
        x, encoder_output, encoder_mask, decoder_mask = x[0], x[1], x[2], x[3]

        # Self-attention for output sequence
        masked_mh_attention = self.masked_mh_attention(
            x, x, x, mask=decoder_mask)
        masked_mh_attention = masked_mh_attention + self.norm1(x)

        # Attention between encoder output and decoder output
        mh_attention = self.mh_attention(
            x, encoder_output, encoder_output, mask=encoder_mask)
        mh_attention = mh_attention + self.norm2(masked_mh_attention)

        # FFC
        ff_x = self.ffc(mh_attention)
        block_output = mh_attention + self.norm3(ff_x)

        return {0: block_output, 1: encoder_output, 2: encoder_mask, 3: decoder_mask}


class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoders=4, ff_dim=256):
        super().__init__()

        encoder_modules = []

        for _ in range(num_encoders):
            encoder_modules.append(EncoderBlock(
                embedding_dim, num_heads, ff_dim=ff_dim))

        self.encoder = nn.Sequential(*encoder_modules)

    def forward(self, x, mask):
        x = {0: x, 1: mask}

        encoder_output = self.encoder(x)[0]

        return encoder_output


class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_decoders=4, ff_dim=256):
        super().__init__()

        decoder_modules = []

        for _ in range(num_decoders):
            decoder_modules.append(DecoderBlock(
                embedding_dim, num_heads, ff_dim=ff_dim))

        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, output_sequence, encoder_output, encoder_mask, decoder_mask):
        x = {0: output_sequence, 1: encoder_output,
             2: encoder_mask, 3: decoder_mask}

        decoder_output = self.decoder(x)[0]

        return decoder_output


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, num_heads=1, num_encoders=1, ff_dim=256, cuda='cpu'):
        super().__init__()
        self.cuda = cuda

        self.input_embedding_layer = SequenceEmbedder(
            input_vocab_size, embedding_dim)

        self.output_embedding_layer = SequenceEmbedder(
            output_vocab_size, embedding_dim
        )

        self.positional_encodings = PositionalEncoding(embedding_dim, cuda)

        self.encoder = Encoder(embedding_dim,
                               num_heads,
                               num_encoders,
                               ff_dim)

        self.decoder = Decoder(embedding_dim,
                               num_heads,
                               num_encoders,
                               ff_dim)

        self.linear = nn.Linear(embedding_dim, output_vocab_size)

        self.bleu1 = -1

    def forward(self, input_sequence, output_sequence=None):
        # x: Tokened Sequence

        # Mask padding
        encoder_mask = self.mask(input_sequence)

        # Embed sequence
        x = self.input_embedding_layer(input_sequence)

        # Add positional encodings
        x = self.positional_encodings(x)

        # Pass through encoder
        encoder_output = self.encoder(x, mask=encoder_mask)

        if output_sequence is not None:
            decoder_mask = self.mask(
                output_sequence, size=output_sequence.size(1))

            y = self.input_embedding_layer(output_sequence)
            y = self.positional_encodings(y)

            decoder_output = self.decoder(
                y, encoder_output, encoder_mask, decoder_mask)

        output = self.linear(decoder_output)

        return output

    def predict(self, input_sequence, output_sequence):

        i = 0
        stop = False

        while i < output_sequence.size(1) - 1 and stop != True:

            output = self(input_sequence, output_sequence)
            pred = torch.argmax(output, dim=2)

            output_sequence[:, i + 1] = pred[:, i]

            if pred[:, i].item() == 1:
                stop = True

            i += 1

        return output_sequence

    def masked_padding(self, sequence):
        padding_idx = 1

        padding_matrix = torch.zeros_like(sequence) + padding_idx

        mask = (sequence != padding_matrix).unsqueeze(1)
        mask_square = mask.transpose(1, 2) * mask
        mask_square = mask_square.unsqueeze(1)

        mask_square = mask_square.to(self.cuda)

        return mask_square

    def mask_subsequent(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

        subsequent_mask = torch.from_numpy(subsequent_mask) == 0
        subsequent_mask = subsequent_mask.to(self.cuda)

        return subsequent_mask

    def mask(self, sequence, size=None):
        mask_padding = self.masked_padding(sequence)

        if size is None:
            return mask_padding

        mask_subsequent = self.mask_subsequent(size)

        return mask_padding & mask_subsequent
