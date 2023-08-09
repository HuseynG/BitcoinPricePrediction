import torch
import torch.nn as nn
import math

# positional encoding
class PE(nn.Module):
    """
    Positional Encoding (PE) module for adding positional information to embeddings in vanilla Transformer models.

    Attributes:
    - dropout (nn.Module): Dropout layer to be applied on the output.
    - pe (Tensor): Positional encoding matrix.

    Parameters:
    - d_model (int): The dimension of the embeddings.
    - dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
    - max_len (int, optional): Maximum sequence length for which positional encoding is pre-computed. Defaults to 5000.
    """
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
    """
    Implementation of a vanilla Transformer encoder-only model tailored for time-series data.
    
    This model consists of:
    - Embedding layer to map features to a higher-dimensional space.
    - Positional encoding to capture sequence order.
    - Multiple Transformer encoder layers.
    - A multi-layer perceptron (MLP) to produce the final output.
    
    Attributes:
    - num_features (int): Number of features in the input data.
    - d_model (int): Dimension of the model/embedding.
    - num_layers (int): Number of Transformer encoder layers.
    - num_heads (int): Number of attention heads.
    - dff (int): Depth of the feed-forward network inside the Transformer.
    - dropout_rate (float): Dropout rate used within the Transformer.
    - mlp_size (int): Size of the intermediate layer in the MLP.
    - mlp_dropout_rate (float): Dropout rate used in the MLP.
    - embedding (nn.Module): Linear layer used for embedding the input features.
    - pos_encoding (PE): Positional encoding module.
    - transformer_encoder (nn.Module): The Transformer encoder.
    - mlp (nn.Sequential): Multi-layer perceptron used at the end of the model.

    Parameters:
    **kwargs: Keyword arguments including:
        - num_features (int): Mandatory argument specifying number of input features.
        - d_model (int, optional): Dimension of model/embedding. Default is 64.
        - num_layers (int, optional): Number of Transformer encoder layers. Default is 6.
        - num_heads (int, optional): Number of attention heads. Default is 8.
        - dff (int, optional): Depth of feed-forward network in Transformer. Default is 1024.
        - dropout_rate (float, optional): Dropout rate for Transformer. Default is 0.2.
        - mlp_size (int, optional): Size of intermediate layer in MLP. Default is 64.
        - mlp_dropout_rate (float, optional): Dropout rate for MLP. Default is 0.2.
    """
    def __init__(self, **kwargs):
        super(VanillaTimeSeriesTransformer_EncoderOnly, self).__init__()
        print("model name is ","VanillaTimeSeriesTransformer_EncoderOnly")
        self.num_features = kwargs.get('num_features')
        self.d_model = kwargs.get('d_model', 64)
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

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.mlp_size),
            nn.ReLU(),
            nn.Dropout(self.mlp_dropout_rate),
            nn.Linear(self.mlp_size, 1),
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
    """
    Implementation of a vanilla Transformer model tailored for time-series data, including both encoder and decoder parts.
    
    This model consists of:
    - Embedding layers for both encoder and decoder parts.
    - Positional encoding to capture sequence order.
    - Multiple Transformer encoder layers to encode the input sequences.
    - Multiple Transformer decoder layers to decode the encoded sequences.
    - A multi-layer perceptron (MLP) to produce the final output.
    
    Attributes:
    - num_features (int): Number of features in the input data.
    - d_model (int): Dimension of the model/embedding.
    - teacher_forcing_ratio (float): Probability of using the ground truth as next input during training.
    - num_layers (int): Number of Transformer layers for both encoder and decoder.
    - num_heads (int): Number of attention heads.
    - dff (int): Depth of the feed-forward network inside the Transformer.
    - dropout_rate (float): Dropout rate used within the Transformer.
    - mlp_size (int): Size of the intermediate layer in the MLP.
    - mlp_dropout_rate (float): Dropout rate used in the MLP.
    - embedding (nn.Module): Linear layer used for embedding the input features for the encoder.
    - decoder_embedding (nn.Module): Linear layer used for embedding the output features for the decoder.
    - pos_encoding (PE): Positional encoding module.
    - transformer_encoder (nn.Module): The Transformer encoder.
    - transformer_decoder (nn.Module): The Transformer decoder.
    - mlp (nn.Sequential): Multi-layer perceptron used at the end of the model.

    Parameters:
    **kwargs: Keyword arguments including:
        - num_features (int): Mandatory argument specifying number of input features.
        - d_model (int, optional): Dimension of model/embedding. Default is 64.
        - teacher_forcing_ratio (float, optional): Probability of using ground truth as next input. Default is 0.5.
        - num_layers (int, optional): Number of Transformer layers for encoder and decoder. Default is 6.
        - num_heads (int, optional): Number of attention heads. Default is 8.
        - dff (int, optional): Depth of feed-forward network in Transformer. Default is 1024.
        - dropout_rate (float, optional): Dropout rate for Transformer. Default is 0.2.
        - mlp_size (int, optional): Size of intermediate layer in MLP. Default is 64.
        - mlp_dropout_rate (float, optional): Dropout rate for MLP. Default is 0.2.
    """
    def __init__(self, **kwargs):
        super(VanillaTimeSeriesTransformer, self).__init__()
        print("model name is ","VanillaTimeSeriesTransformer")
        self.num_features = kwargs.get('num_features')
        self.d_model = kwargs.get('d_model', 64)
        self.teacher_forcing_ratio = kwargs.get('teacher_forcing_ratio', 0.5)
        self.num_layers = kwargs.get('num_layers', 6)
        self.num_heads = kwargs.get('num_heads', 8)
        self.dff = kwargs.get('dff', 1024)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.mlp_size = kwargs.get('mlp_size', 64)
        self.mlp_dropout_rate = kwargs.get('mlp_dropout_rate', 0.2)

        self.embedding = nn.Linear(self.num_features, self.d_model)
        self.decoder_embedding = nn.Linear(1, self.d_model)
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
        # print("X Shape after embedding:", x.shape)
        
        x = self.pos_encoding(x)
        # print("X Shape after positional encoding:", x.shape)
        
        x = x.transpose(0, 1)  # transformer expect input as (seq_len, batch_size, num_features) by default
        # print("X Shape after transpose:", x.shape)

        encoder_output = self.transformer_encoder(x) # encoded input
        # print("Encoder output shape:", encoder_output.shape)

        target_variable = torch.zeros(1, x.size(1), self.d_model, device=x.device)
        # print("Target variable shape:", target_variable.shape)

        decoder_input = target_variable

        # if it is during training and y is provded and the probability of teacher forcing is higher than set treshold
        if self.training and 'y' in kwargs and torch.rand(1).item() < self.teacher_forcing_ratio:
            # print("y shape:", kwargs['y'].shape)
            decoder_input = self.decoder_embedding(kwargs['y']).view(1, -1, self.d_model)  # view is used to match the expected shape of the decoder
            # print("Decoder input shape (after teacher forcing):", decoder_input.shape)

        output = self.transformer_decoder(decoder_input, encoder_output)
        # print("Decoder output shape:", output.shape)

        x_last = output[-1, :, :]
        # print("Last output shape:", x_last.shape)

        x_mean = torch.mean(output, dim=0)
        # print("Mean output shape:", x_mean.shape)

        x_res = x_mean + x_last
        # print("Residual connection shape:", x_res.shape)
        
        x = self.mlp(x_res)
        # print("Final output shape:", x.shape)

        return x