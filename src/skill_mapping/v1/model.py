# model.py

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, binary_cross_entropy_with_logits
from typing import List, Dict, Callable, Tuple

# -----------------------------
# 1. Reusable Components
# -----------------------------

class TextEncoderWrapper(nn.Module):
    """
    A simple nn.Module wrapper for a text encoding function.
    
    This allows us to pass a simple callable (like a lambda for
    SentenceTransformer.encode) into our models and have it
    be part of the model graph, enabling fine-tuning.
    """
    def __init__(self, encoder_callable: Callable[[List[str]], torch.Tensor]):
        super().__init__()
        # This backbone is expected to be a function, e.g.:
        # lambda texts: model.encode(texts, convert_to_tensor=True)
        self.backbone = encoder_callable

    def forward(self, text_batch: List[str]) -> torch.Tensor:
        """
        Args:
            text_batch: A list of raw strings (length B).
        Returns:
            A [B, D] tensor of embeddings.
        """
        # The callable is assumed to handle encoding and tensor conversion
        return self.backbone(text_batch)

class MultiLabelClassifierHead(nn.Module):
    """
    A simple linear classification head for multi-label prediction.
    Takes embeddings [B, D] and outputs logits [B, C].
    """
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, D]
        Returns:
            logits: [B, C]
        """
        # Clone the embeddings to create a normal tensor that can be used in autograd
        # This is needed when embeddings come from inference mode (e.g., SentenceTransformer.encode)
        if not embeddings.requires_grad:
            embeddings = embeddings.clone().detach().requires_grad_(True)
        return self.fc(embeddings)

# -----------------------------
# 2. Architecture 1 (Smarter Hybrid) - Model Components
# -----------------------------

class CategoryPredictor(nn.Module):
    """
    This is the **Step 1 Model**.
    
    Task: job_text -> multi-label category prediction.
    Architecture: Encoder + Classifier Head
    """
    def __init__(self, encoder: TextEncoderWrapper, hidden_dim: int, num_categories: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = MultiLabelClassifierHead(hidden_dim, num_categories)

    def forward(self, text_batch: List[str]) -> torch.Tensor:
        """
        Args:
            text_batch: A list of raw job texts (length B).
        Returns:
            Logits [B, C] where C = num_categories.
        """
        emb = self.encoder(text_batch)      # [B, D]
        logits = self.classifier(emb)       # [B, C]
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for a batch.
        Args:
            logits: [B, C] raw scores from forward()
            targets: [B, C] multi-hot {0,1} labels
        """
        return binary_cross_entropy_with_logits(logits, targets)

    @torch.no_grad()
    def predict_proba(self, text_batch: List[str]) -> torch.Tensor:
        """
        Returns probabilities [B, C] for inference.
        """
        self.eval()
        logits = self.forward(text_batch)
        self.train()
        return torch.sigmoid(logits)


class SkillSimilarityModel(nn.Module):
    """
    This is the **Step 2 Model**.
    
    Task: Learn a shared embedding space for jobs and skills.
    Architecture: Bi-Encoder (a single, shared encoder).
    """
    def __init__(self, encoder: TextEncoderWrapper):
        super().__init__()
        # A bi-encoder uses *one* encoder for both inputs
        self.encoder = encoder

    def forward(self, job_texts: List[str], skill_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes both job and skill texts using the shared encoder.
        """
        job_emb = self.encoder(job_texts)
        skill_emb = self.encoder(skill_texts)
        return job_emb, skill_emb

    def compute_loss(self, job_emb: torch.Tensor, skill_emb: torch.Tensor) -> torch.Tensor:
        """
        Computes the Multiple Negatives Ranking Loss (MNRL).
        
        We assume job_emb[i] and skill_emb[i] are a positive pair,
        and all other (i, j) pairs in the batch are negatives.
        
        Args:
            job_emb: [B, D] tensor of job embeddings
            skill_emb: [B, D] tensor of *corresponding positive* skill embeddings
        """
        # Calculate cosine similarity matrix
        # sim_matrix[i, j] = similarity(job_emb[i], skill_emb[j])
        sim_matrix = cosine_similarity(job_emb.unsqueeze(1), skill_emb.unsqueeze(0), dim=-1)
        
        # The labels are the diagonal (0, 1, 2, ... B-1)
        labels = torch.arange(job_emb.size(0), device=job_emb.device)
        
        # Calculate cross-entropy loss. This pushes the diagonal (positive)
        # scores to be high and off-diagonal (negative) scores to be low.
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(sim_matrix, labels)
        
        return loss

    @torch.no_grad()
    def predict_similarity(self, job_texts: List[str], skill_texts: List[str]) -> torch.Tensor:
        """
        Returns a similarity matrix [Num_Jobs, Num_Skills] for inference.
        """
        self.eval()
        job_emb, skill_emb = self.forward(job_texts, skill_texts)
        self.train()
        
        # L2 normalize for stable cosine similarity
        job_emb = job_emb / job_emb.norm(dim=1, keepdim=True)
        skill_emb = skill_emb / skill_emb.norm(dim=1, keepdim=True)
        
        # Compute the similarity matrix
        return torch.matmul(job_emb, skill_emb.t())