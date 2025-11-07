import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, binary_cross_entropy_with_logits
from typing import List, Dict, Callable, Tuple
from sentence_transformers import SentenceTransformer, models # <-- 1. ADD 'models'
from loguru import logger

# -----------------------------
# 1. NEW: Encoder Factory Function
# -----------------------------

def build_encoder(encoder_name_or_path: str, device: str = "cpu") -> SentenceTransformer:
    """
    Factory function to load a SentenceTransformer.
    
    It first attempts the standard, direct loading method.
    If that fails, it falls back to a more explicit, manual assembly method.
    If both fail, it raises the final error.
    """
    
    # --- 1. Try the standard loading method first ---
    try:
        logger.info(f"Attempting standard load for: {encoder_name_or_path}")
        return SentenceTransformer(encoder_name_or_path, device=device, trust_remote_code=True) # fix 1
    
    except Exception as e_standard:
        logger.warning(
            f"Standard loading failed for {encoder_name_or_path} with error: {e_standard}. "
            f"Falling back to manual assembly..."
        )
        
        # --- 2. Fallback: Try the explicit manual assembly method ---
        try:
            logger.info(f"Attempting manual assembly for: {encoder_name_or_path}")
            
            # These args are common for BGE-style models
            config_args = {
                "model_type": "bert",
                "trust_remote_code": True
            }
            model_args = {
                "trust_remote_code": True
            }

            word_embedding_model = models.Transformer(
                encoder_name_or_path, 
                model_args=model_args,
                config_args=config_args # fix 2
            )

            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode='cls'
            )

            normalize_model = models.Normalize()

            return SentenceTransformer(
                modules=[word_embedding_model, pooling_model, normalize_model],
                device=device
            )
            
        except Exception as e_manual:
            # --- 3. If both methods fail, break ---
            logger.error(
                f"Manual assembly also failed for {encoder_name_or_path} with error: {e_manual}. "
                f"Both loading methods failed."
            )
            # Re-raise the most recent exception, as it's the one from the
            # failed fallback attempt.
            raise e_manual

# -----------------------------
# 2. Reusable Components
# -----------------------------

class TextEncoderWrapper(nn.Module):
    """
    A proper nn.Module wrapper for a SentenceTransformer model
    that enables fine-tuning.
    """
    def __init__(self, sbert_model: SentenceTransformer):
        super().__init__()
        # Hold the actual SentenceTransformer model
        self.sbert_model = sbert_model

    def forward(self, text_batch: List[str]) -> torch.Tensor:
        """
        Performs a gradient-tracking forward pass.
        """
        features = self.sbert_model.tokenize(text_batch)
        for k, v in features.items():
            features[k] = v.to(self.sbert_model.device)
            
        output = self.sbert_model(features)
        return output['sentence_embedding']

class MultiLabelClassifierHead(nn.Module):
    """
    A simple linear classification head for multi-label prediction.
    """
    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(embeddings)

# -----------------------------
# 3. Model Architectures
# -----------------------------

class CategoryPredictor(nn.Module):
    """
    This is the **Step 1 Model**.
    """
    def __init__(self, encoder: TextEncoderWrapper, hidden_dim: int, num_categories: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = MultiLabelClassifierHead(hidden_dim, num_categories)

    # ... (forward, compute_loss, predict_proba are unchanged) ...
    def compute_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        pos_weight: torch.Tensor = None
    ) -> torch.Tensor:
        return binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight=pos_weight
        )
        
    def forward(self, text_batch: List[str]) -> torch.Tensor:
        emb = self.encoder(text_batch)
        logits = self.classifier(emb)
        return logits
    
    @torch.no_grad()
    def predict_proba(self, text_batch: List[str]) -> torch.Tensor:
        self.eval()
        logits = self.forward(text_batch)
        self.train()
        return torch.sigmoid(logits)


class SkillSimilarityModel(nn.Module):
    """
    This is the **Step 2 Model**.
    """
    def __init__(self, encoder: TextEncoderWrapper):
        super().__init__()
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