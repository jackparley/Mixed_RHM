import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import MLP

class MultiHeadAttention(nn.Module):
    """
    Multiple Attention Heads.
    
    Args:
        input_dim: The dimension of input tokens.
        num_heads: The number of heads.
        head_dim: The dimension of each head.
        out_dim: The dimension of output tokens.
        
    """
    def __init__(
        self, input_dim, num_heads, head_dim, out_dim
    ):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        self.key=nn.Parameter(
            torch.randn( self.inner_dim, self.input_dim)
        )
        self.query=nn.Parameter(
            torch.randn( self.inner_dim, self.input_dim)
        )
        self.value=nn.Parameter(
            torch.randn( self.inner_dim, self.input_dim)
        )
        self.projection=nn.Parameter(
            torch.randn( self.out_dim, self.inner_dim)
        )

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, input_size, input_dim).
        
        Returns:
            The output of a multi-head attention layer,
            of size (batch_size, input_size, output_dim)
        """
        B,T,C = x.size()
        k = F.linear( x, self.key, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) *C**-.5    # [bs, num_heads, seq_len, head_dim]
        q = F.linear( x, self.query, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) *C**-.5  # [bs, num_heads, seq_len, head_dim]
        v = F.linear( x, self.value, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) *C**-.5  # [bs, num_heads, seq_len, head_dim]

        weight = q @ k.transpose(-2,-1) * self.head_dim**-.5             # [bs, num_heads, seq_len, seq_len]
        weight = F.softmax(weight, dim=-1)                              #  //

        out = (weight @ v).transpose(1,2).reshape(B,T,-1) # [bs, seq_len, inner_dim]
        out = F.linear( out, self.projection, bias=None) * self.projection.size(-1)**-.5

        return out


class AttentionBlock(nn.Module):
    def __init__(
        self, embedding_dim, num_heads
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.sa = MultiHeadAttention(
            input_dim=embedding_dim,
            num_heads=num_heads,
            head_dim=embedding_dim//num_heads,
            out_dim=embedding_dim
        )

    def forward(self, x):
        x = self.sa(x)
        return x


class EncoderBlock(nn.Module):
    """
    One Decoder Block.
    
    Args:
        embedding_dim: The dimension of the tokens (kept constant past embedding).
        num_heads: The number of attention heads.
        ffwd_size: Size of the MLP is ffwd_size*embedding_dim.
    """
    def __init__(
        self, embedding_dim, num_heads, ffwd_size=4
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding dim. must be multiple of num. heads"

        self.sa = MultiHeadAttention(
            input_dim=embedding_dim,
            num_heads=num_heads,
            head_dim=embedding_dim//num_heads,
            out_dim=embedding_dim
        )
        self.ffwd = MLP(
            input_dim=embedding_dim, 
            nn_dim=ffwd_size*embedding_dim, 
            out_dim=embedding_dim, 
            num_layers=1
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MLA(nn.Module):
    """
    Multi-Layer Attention classifier using EncoderBlocks and first-token readout.

    Args:
        vocab_size: Size of the vocabulary (including special CLS token).
        block_size: Maximum number of input tokens (including CLS token).
        embedding_dim: Embedding dimensionality.
        num_heads: Number of attention heads.
        num_layers: Number of encoder layers.
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers):
        super().__init__()

        self.vocab_size = vocab_size
        if num_layers==2:
            dmax=9
            block_size=dmax+1
        elif num_layers==3:
            dmax=27
            block_size=dmax+1
        elif num_layers==4:
            dmax=81
            block_size=dmax+1
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Token embedding: projects one-hot vectors (vocab_size) to embedding_dim
        self.token_embedding = nn.Parameter(
            torch.randn(embedding_dim, vocab_size)
        )

        # Positional embedding: learnable position encodings
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        # Transformer encoder stack
        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads
                ) for _ in range(num_layers)
            ]
        )

        # Final readout: from embedding_dim back to vocab_size
        self.readout = nn.Parameter(
            torch.randn(vocab_size, embedding_dim)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, vocab_size) — one-hot encoded input

        Returns:
            Logits of shape (batch_size, vocab_size) from position 0 ([CLS]).
        """
        B, T, C = x.size()
        assert C == self.vocab_size, f"Expected vocab_size={self.vocab_size}, got {C}"
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        # Project one-hot tokens to embeddings
        token_emb = F.linear(x, self.token_embedding, bias=None) * C**-0.5  # (B, T, D)

        # Add position embeddings
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos_ids)  # (1, T, D)
        x = token_emb + pos_emb  # (B, T, D)

        # Pass through Transformer layers
        x = self.blocks(x)  # (B, T, D)

        # Read from first position (assumed to be CLS)
        x_cls = x[:, 0, :]  # (B, D)

        # Project to logits
        logits = F.linear(x_cls, self.readout, bias=None) * self.readout.size(-1)**-0.5  # (B, vocab_size)
        return logits


class MLA_old(nn.Module):
    """
    Multi-Layer Multi-Head Attention for last token prediction

    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.
    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, num_layers
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_embedding=nn.Parameter(
            torch.randn( self.embedding_dim, self.vocab_size)
        )
        self.position_embedding = nn.Embedding(self.block_size, self.embedding_dim)

        self.blocks = nn.Sequential(
            *[
                AttentionBlock(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                ) for _ in range(self.num_layers)
            ]
        )
        self.readout = nn.Parameter(
            torch.randn(self.vocab_size, self.embedding_dim)
        )


    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, seq_len, vocab_size).
        
        Returns:
            Output of multilayer self-attention, tensor of size (batch_size, seq_len, vocab_size)
        """
        B,T,C = x.size()
        token_emb = F.linear( x, self.token_embedding, bias=None) *C**-.5 # [bs, seq_len, embedding_dim]
        pos_emb = self.position_embedding(torch.arange(T, device=x.device)) # [seq_len, embedding_dim]
        x = token_emb + pos_emb  # [bs, seq_len, embedding_dim]
        x = self.blocks(x)
        logits = F.linear( x[:,-1,:], self.readout, bias=None) * self.readout.size(-1)**-.5

        return logits
