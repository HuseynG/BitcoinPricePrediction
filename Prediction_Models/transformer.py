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
                 num_heads, dff, mlp_size, dropout_rate, mlp_dropout_rate):
        super(VanillaTimeSeriesTransformer_EncoderOnly, self).__init__()

        self.d_model = d_model
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
        x_last = x[-1, :, :]
        x_mean = torch.mean(x, dim=0)
        # print("Final output for each sequence:", x.shape)
        
        x_res = x_mean + x_last

        x = self.mlp(x_res)
        # print("After MLP:", x.shape)

        return x
    
class VanillaTimeSeriesTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(VanillaTimeSeriesTransformer, self).__init__()


        self.num_features = kwargs.get('num_features')
        # self.input_seq_len = kwargs.get('input_seq_len')
        # self.output_seq_len = kwargs.get('output_seq_len')

        self.d_model = kwargs.get('d_model', 64)
        self.teacher_forcing_ratio = kwargs.get('teacher_forcing_ratio', 0.5)
        self.num_layers = kwargs.get('num_layers', 6)
        self.num_heads = kwargs.get('num_heads', 8)
        self.dff = kwargs.get('dff', 1024)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.mlp_size = kwargs.get('mlp_size', 64)
        self.mlp_dropout_rate = kwargs.get('mlp_dropout_rate', 0.2)

        self.embedding = nn.Linear(self.num_features, self.d_model)
        self.pos_encoding = PE(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.num_heads, self.dff, self.dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        decoder_layer = nn.TransformerDecoderLayer(self.d_model, self.num_heads, self.dff, self.dropout_rate)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.mlp_size),
            nn.ReLU(),
            nn.Dropout(self.mlp_dropout_rate),
            nn.Linear(self.mlp_size, 1),
        )

    def forward(self, x, **kwargs):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # transformer expect input as (seq_len, batch_size, num_features) by default

        encoder_output = self.transformer_encoder(x) # encoded input

        # preparing target variable for decoder which is intilised with zeros. size is 1 (sequence length) x batch size x model dimension
        target_variable = torch.zeros(1, x.size(1), self.d_model, device=x.device)

        decoder_input = target_variable

        # checking if the model is in training mode and true ys are present in kwargs
        if self.training and 'y' in kwargs and torch.rand(1).item() < self.teacher_forcing_ratio:
            # then Teacher Forcing: feeding the target as the next input
            decoder_input = kwargs['y'].transpose(0, 1)

        # passing input for decoder (which is either true ys or 0s) and encoders encoding memory
        output = self.transformer_decoder(decoder_input, encoder_output)

        # taking the final output for each sequence
        x_last = output[-1, :, :]
        # taking mean output for each seqeuence
        x_mean = torch.mean(output, dim=0)

        # res connection
        x_res = x_mean + x_last
        x = self.mlp(x_res)

        return x