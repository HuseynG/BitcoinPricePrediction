import torch
import torch.nn as nn
import math

# positional encoding
class PE(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super(PE, self).__init__() # calling nn.Module constructor

        # creating a dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # defining matrix of 0s for pe
        pe = torch.zeros(max_len, d_model)

        # a tensor representing the positions in the sequence
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # calculating divisor (based on the original formula) for positional encoding
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # applying sine to even indices (2i) and cosine to odd indices (2i+1)
        pe[:, 0::2] = torch.sin(pos * divisor)
        pe[:, 1::2] = torch.cos(pos * divisor)

        # adding an extra dimension to the array and transposing it, that is, dimensions 0 and 1
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register pe as a buffer that should not be considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class VanillaTimeSeriesTransformer_EncoderOnly(nn.Module):
    def __init__(self, num_features, num_layers, d_model,
                 num_heads, dff, input_seq_len, output_seq_len,
                 mlp_size, dropout_rate, mlp_dropout_rate):
        super(VanillaTimeSeriesTransformer_EncoderOnly, self).__init__()

        self.d_model = d_model
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.embedding = nn.Linear(num_features, d_model)
        self.pos_encoding = PE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_size),
            nn.ReLU(),
            nn.Dropout(mlp_dropout_rate),
            nn.Linear(mlp_size, 1),
        )

    def forward(self, x):
        # print("Input Shape:", x.shape)

        x = self.embedding(x) * math.sqrt(self.d_model)
        # print("After Embedding:", x.shape)

        x = self.pos_encoding(x)
        # print("After Positional Encoding:", x)

        x = x.transpose(0, 1)  # transformer expect input as (seq_len, batch_size, num_features) by default
        # print("After Transposing for Transformer:", x.shape)

        x = self.transformer_encoder(x)
        # print("After Transformer Encoder:", x.shape)

        # taking the final output for each sequence
        x = x[-1, :, :]
        # print("Final output for each sequence:", x.shape)

        x = self.mlp(x)
        # print("After MLP:", x.shape)

        return x