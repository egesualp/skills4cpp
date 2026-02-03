import torch
import torch.nn as nn
import math

class CareerPathAggregator(nn.Module):
    def __init__(
        self, 
        input_dim=1024, 
        model_dim=512,  # Internal dimension for Transformer
        n_heads=8, 
        n_layers=2, 
        dropout=0.1,
        max_seq_len=100
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        # Dimension reduction/projection layer
        # If input_dim != model_dim, we project it.
        if input_dim != model_dim:
            self.input_proj = nn.Linear(input_dim, model_dim)
            self.output_proj = nn.Linear(model_dim, input_dim) # Project back to embedding space
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()

        # 1. Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, model_dim))

        # 2. The Transformer-Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=n_heads, 
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 3. Prediction Head 
        # We project the output of transformer (model_dim) back to input_dim space 
        # (which is the embedding space of next job)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, input_dim) 
        )

    def forward(self, job_embeddings, mask=None):
        """
        job_embeddings: Tensor of shape (Batch_Size, Seq_Len, input_dim)
        mask: Tensor of shape (Batch_Size, Seq_Len) for padding masking (True where padded)
              Transformer expects mask where True values are ignored.
        """
        batch_size, seq_len, _ = job_embeddings.size()
        
        # Project input if necessary
        x = self.input_proj(job_embeddings) # (B, L, model_dim)

        # Add positional info
        # Slice pos_encoder to current sequence length
        if seq_len > self.pos_encoder.size(1):
             # Handle case where sequence is longer than max (should be handled in dataset)
             # But as safety, we can repeat or fail.
             x = x + self.pos_encoder[:, :self.pos_encoder.size(1), :] # This would fail shape check
             raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.pos_encoder.size(1)}")
             
        x = x + self.pos_encoder[:, :seq_len, :]

        # Pass through Transformer
        # mask needs to be compatible with src_key_padding_mask:
        # If mask is boolean: True indicates elements that should NOT be attended to (padded positions).
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Aggregation Strategy: Take the vector corresponding to the LAST job
        # because it contains the context of all previous jobs via attention.
        # Note: If sequences are padded on the right, x[:, -1, :] takes the padding token's output.
        # This is generally incorrect for variable length sequences unless handled.
        # We assume the caller handles picking the right token or uses forward_with_lengths.
        last_hidden_state = x[:, -1, :]

        # Project to target space
        predicted_next_job_vector = self.head(last_hidden_state)

        return predicted_next_job_vector

    def forward_with_lengths(self, job_embeddings, lengths=None, mask=None):
         # Helper to handle lengths correctly
        batch_size, seq_len, _ = job_embeddings.size()
        
        x = self.input_proj(job_embeddings)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        if lengths is not None:
             # Gather the last valid hidden state for each sequence
             # lengths is (B,)
             # We want x[b, lengths[b]-1, :]
             idx = (lengths - 1).view(-1, 1).expand(batch_size, x.size(2)).unsqueeze(1)
             # idx shape (B, 1, model_dim)
             last_hidden_state = x.gather(1, idx.to(x.device)).squeeze(1)
        else:
             last_hidden_state = x[:, -1, :]
             
        predicted_next_job_vector = self.head(last_hidden_state)
        return predicted_next_job_vector

