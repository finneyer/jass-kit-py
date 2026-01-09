import torch
import torch.nn as nn
import torch.nn.functional as F


class JassTransformer(nn.Module):
    def __init__(
        self,
        input_dim=332,
        embed_dim=128,
        num_heads=4,
        ff_dim=512,
        num_layers=3,
        dropout=0.1,
        max_seq_len=100
    ):
        """
        Transformer-based Neural Network for Jass AlphaZero.
        
        Args:
            input_dim: Input feature dimension (332 for current encoding)
            embed_dim: Embedding/hidden dimension for transformer
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension in transformer blocks
            num_layers: Number of transformer encoder layers
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
        """
        super(JassTransformer, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 36)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 332)
            
        Returns:
            policy: Probability distribution over 36 cards (batch_size, 36)
            value: Expected outcome from current player's perspective (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        x = x.unsqueeze(1)
        x = x + self.pos_encoding
        
        x = self.transformer(x)
        
        x = x.squeeze(1)
        x = self.output_norm(x)
        x = self.output_dropout(x)
        
        policy_logits = self.policy_head(x)
        policy = F.softmax(policy_logits, dim=1)
        value = self.value_head(x)
        
        return policy, value


class JassTransformerSequence(nn.Module):
    """
    Alternative transformer that treats the game state as a sequence.
    This splits the 332 features into meaningful chunks.
    """
    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        ff_dim=512,
        num_layers=3,
        dropout=0.1
    ):
        super(JassTransformerSequence, self).__init__()
        
        self.chunk_sizes = [36] * 9 + [8]
        self.num_chunks = len(self.chunk_sizes)
        
        self.chunk_embeddings = nn.ModuleList([
            nn.Linear(size, embed_dim) for size in self.chunk_sizes
        ])
        
        self.pos_embedding = nn.Embedding(self.num_chunks, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 36)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
                nn.init.ones_(module.weight) if hasattr(module, 'weight') else None
                nn.init.zeros_(module.bias) if hasattr(module, 'bias') else None
    
    def forward(self, x):
        """
        Forward pass treating input as a sequence of feature chunks.
        
        Args:
            x: Input tensor of shape (batch_size, 332)
            
        Returns:
            policy: Probability distribution over 36 cards (batch_size, 36)
            value: Expected outcome (batch_size, 1)
        """
        batch_size = x.shape[0]
        
        chunks = []
        offset = 0
        for i, size in enumerate(self.chunk_sizes):
            chunk = x[:, offset:offset+size]
            chunk_emb = self.chunk_embeddings[i](chunk)
            chunks.append(chunk_emb)
            offset += size
        
        seq = torch.stack(chunks, dim=1)
        
        positions = torch.arange(self.num_chunks, device=x.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0)
        seq = seq + pos_emb
        
        seq = self.transformer(seq)
        
        pooled = seq.mean(dim=1)
        pooled = self.output_norm(pooled)
        pooled = self.output_dropout(pooled)
        
        policy_logits = self.policy_head(pooled)
        policy = F.softmax(policy_logits, dim=1)
        value = self.value_head(pooled)
        
        return policy, value
