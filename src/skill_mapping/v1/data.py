# data.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple

# Mapping of hierarchy levels to the column names you provided
HIER_COL_MAP = {
    0: "level0_label",
    1: "level1_label",
    2: "level2_label",
    3: "level3_label",
}

class LabelEncoder:
    """
    Minimal label encoder for multi-label classification.
    - For Stage 1: encodes category strings (e.g. pillars or level1 groups).
    """
    def __init__(self, values: List[str]):
        unique_vals = sorted(list(set(values)))
        self.str2idx = {v: i for i, v in enumerate(unique_vals)}
        self.idx2str = {i: v for v, i in self.str2idx.items()}

    def encode_multi(self, items: List[str]) -> torch.Tensor:
        """
        Turn a list of item strings into a multi-hot vector.
        """
        y = torch.zeros(len(self.str2idx), dtype=torch.float32)
        for it in items:
            if it in self.str2idx:
                y[self.str2idx[it]] = 1.0
        return y

    def __len__(self):
        return len(self.str2idx)

def process_raw_description(raw_description: str) -> str:
    """
    Cleans raw job descriptions (e.g., from DECORTE) by:
    1. Splitting into bullet items
    2. Cleaning each bullet (strip -, trim spaces)
    3. Re-joining into one readable sentence or short pseudo-paragraph
    """
    raw_description = str(raw_description)
    if not raw_description or raw_description.strip() == "" or raw_description == "nan":
        return ""
    
    # Split by newlines to get individual bullet points
    bullets = raw_description.split('\n')
    
    # Clean each bullet point
    cleaned_bullets = []
    for bullet in bullets:
        bullet = bullet.strip()
        if bullet:
            # Remove leading dash and spaces
            if bullet.startswith('- '):
                bullet = bullet[2:]
            elif bullet.startswith('-'):
                bullet = bullet[1:]
            
            bullet = bullet.strip()
            if bullet:
                cleaned_bullets.append(bullet)
    
    # Re-join into readable paragraph
    if cleaned_bullets:
        return '. '.join(cleaned_bullets) + '.'
    else:
        return ""

def build_job_text(
    row: pd.Series,
    text_fields: str = "title",
    is_structured: bool = False
) -> str:
    """
    Creates the input text string for a job, using either
    raw job data (DECORTE/Karrierewege) or canonical ESCO data.

    Args:
        row: A pandas Series. Must contain *either*
             (raw_job_title, raw_job_description) OR
             (occupationLabel, occupationDescription, occupationAltLabels).
        text_fields: "title", "title+desc", or "title+desc+alt"
        is_structured: If True, prepends "Job Title:", "Description:", etc.
    """
    
    if "raw_job_title" in row and pd.notna(row["raw_job_title"]):
        # This is a free-text job (DECORTE or Karrierewege)
        title = str(row.get("raw_job_title", "")).strip()
        raw_desc = str(row.get("raw_job_description", ""))
        desc = process_raw_description(raw_desc)
        alt = "" # Free-text jobs don't have alt labels in this schema
    else:
        # This is a canonical ESCO occupation row
        title = str(row.get("occupationLabel", "")).strip()
        desc = str(row.get("occupationDescription", "")).strip()
        alt = str(row.get("occupationAltLabels", "")).strip()

    # Build the final text
    if not is_structured:
        if text_fields == "title":
            return title
        elif text_fields == "title+desc":
            return f"{title} [SEP] {desc}" if desc else title
        elif text_fields == "title+desc+alt":
            text = title
            if desc: text += f" [SEP] {desc}"
            if alt: text += f" [SEP] {alt}"
            return text
    else:
        # Structured text for instruction-tuned models
        if text_fields == "title":
            return f"Job Title: {title}"
        elif text_fields == "title+desc":
            return f"Job Title: {title} [SEP] Description: {desc}" if desc else f"Job Title: {title}"
        elif text_fields == "title+desc+alt":
            text = f"Job Title: {title}"
            if desc: text += f" [SEP] Description: {desc}"
            if alt: text += f" [SEP] Alternative Labels: {alt}"
            return text
            
    raise ValueError(f"Unknown text_fields={text_fields}")

def build_skill_text(
    row: pd.Series,
    text_fields: str = "title",
    is_structured: bool = False
) -> str:
    """
    Creates the input text string for a skill from the ESCO master dataset.

    Args:
        row: A pandas Series containing skill info 
             (skillLabel, description, skillAltLabels).
        text_fields: "title", "title+desc", or "title+desc+alt"
        is_structured: If True, prepends "Skill:", "Description:", etc.
    """
    title = str(row.get("skillLabel", "")).strip()
    desc = str(row.get("description", "")).strip()
    alt = str(row.get("skillAltLabels", "")).strip()

    # Build the final text
    if not is_structured:
        if text_fields == "title":
            return title
        elif text_fields == "title+desc":
            return f"{title} [SEP] {desc}" if desc else title
        elif text_fields == "title+desc+alt":
            text = title
            if desc: text += f" [SEP] {desc}"
            if alt: text += f" [SEP] {alt}"
            return text
    else:
        if text_fields == "title":
            return f"Skill: {title}"
        elif text_fields == "title+desc":
            return f"Skill: {title} [SEP] Description: {desc}" if desc else f"Skill: {title}"
        elif text_fields == "title+desc+alt":
            text = f"Skill: {title}"
            if desc: text += f" [SEP] Description: {desc}"
            if alt: text += f" [SEP] Alternative Labels: {alt}"
            return text
            
    raise ValueError(f"Unknown text_fields={text_fields}")


class CategoryDataset(Dataset):
    """
    Dataset for **Stage 1: Category Prediction**.

    Maps a job text (from free-text OR canonical ESCO) to a
    multi-hot vector of its associated skill categories.
    """
    def __init__(
        self,
        esco_df: pd.DataFrame,
        job_df: Optional[pd.DataFrame] = None,
        hier_level: int = 0,
        text_fields: str = "title+desc",
        is_structured: bool = False,
        use_canonical_jobs: bool = True
    ):
        self.cat_col = HIER_COL_MAP[hier_level]
        self.text_fields = text_fields
        self.is_structured = is_structured
        self.samples = []

        esco_df = esco_df[~esco_df[self.cat_col].isna()]

        # 1. Process free-text jobs (DECORTE, Karrierewege)
        if job_df is not None:
            # Reset index to ensure we can track original job indices
            job_df_with_idx = job_df.reset_index().rename(columns={'index': 'original_job_idx'})
            
            # Link jobs to their skills and categories
            merged_df = job_df_with_idx.merge(
                esco_df,
                left_on="esco_id",
                right_on="occupationUri",
                how="left",
                suffixes=("_job", "_skill")
            )
            
            # Group by the original job index to get one sample per job
            for job_idx, skills_for_job in merged_df.groupby('original_job_idx'):
                if skills_for_job.empty:
                    continue
                
                first_row = skills_for_job.iloc[0]
                # Build text from the *job's* columns
                text = build_job_text(first_row, text_fields, is_structured)
                
                # Get unique categories from all *skills* linked to this job
                cats = skills_for_job[self.cat_col].dropna().astype(str).unique().tolist()
                
                self.samples.append({
                    "text": text,
                    "categories": cats,
                })

        # 2. Process canonical ESCO jobs
        if use_canonical_jobs:
            for occ_id, skills_for_occ in esco_df.groupby("occupationUri"):
                if skills_for_occ.empty:
                    continue
                    
                first_row = skills_for_occ.iloc[0]
                # Build text from the *occupation's* columns
                text = build_job_text(first_row, text_fields, is_structured)
                
                # Get unique categories from all skills linked to this occupation
                cats = skills_for_occ[self.cat_col].dropna().astype(str).unique().tolist()
                
                self.samples.append({
                    "text": text,
                    "categories": cats,
                })

        # 3. Build the label encoder
        # Use *all* categories from ESCO to ensure a complete, stable mapping
        all_cats = esco_df[self.cat_col].dropna().astype(str).unique().tolist()
        self.label_encoder = LabelEncoder(all_cats)

        # 4. Pre-encode all target vectors
        for s in self.samples:
            s["target_vec"] = self.label_encoder.encode_multi(s["categories"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        return {
            "text": row["text"],
            "y": row["target_vec"],  # torch.FloatTensor [num_categories]
        }

    def num_classes(self) -> int:
        return len(self.label_encoder)

    def get_vocab(self) -> Dict[int, str]:
        return self.label_encoder.idx2str


class GlobalSkillDataset(Dataset):
    """
    Dataset for **Stage 2: Global Skill Similarity (Contrastive Loss)**.

    Provides pairs of (job_text, list_of_positive_skill_texts).
    This is used to train the `SimilarityModel` (Bi-Encoder).
    """
    def __init__(
        self,
        esco_df: pd.DataFrame,
        job_df: Optional[pd.DataFrame] = None,
        text_fields: str = "title+desc",
        is_structured: bool = False,
        use_canonical_jobs: bool = True
    ):
        self.text_fields = text_fields
        self.is_structured = is_structured
        self.skill_label_col = "skillLabel" # Use label for simplicity
        self.samples = []

        # 1. Process free-text jobs (DECORTE, Karrierewege)
        if job_df is not None:
            # Reset index to ensure we can track original job indices
            job_df_with_idx = job_df.reset_index().rename(columns={'index': 'original_job_idx'})
            esco_df = esco_df[~esco_df.level0_uri.isna()]
            
            merged_df = job_df_with_idx.merge(
                esco_df,
                left_on="esco_id",
                right_on="occupationUri",
                how="left",
                suffixes=("_job", "_skill")
            )
            
            for job_idx, skills_for_job in merged_df.groupby('original_job_idx'):
                if skills_for_job.empty:
                    continue
                
                first_row = skills_for_job.iloc[0]
                job_text = build_job_text(first_row, text_fields, is_structured)
                
                # Get unique skill labels and build their texts
                skill_labels = skills_for_job[self.skill_label_col].dropna().astype(str).unique()
                skill_texts = []
                for sl in skill_labels:
                    # Find the first row in this group that matches the skill label
                    skill_row_matches = skills_for_job[skills_for_job[self.skill_label_col] == sl]
                    if not skill_row_matches.empty:
                        skill_texts.append(build_skill_text(skill_row_matches.iloc[0], text_fields, is_structured))
                
                if job_text and skill_texts:
                    self.samples.append({
                        "text": job_text,
                        "skill_texts": skill_texts,
                    })

        # 2. Process canonical ESCO jobs
        if use_canonical_jobs:
            # We need a de-duplicated esco_df for skills to build texts
            skill_lookup_df = esco_df.drop_duplicates(subset=[self.skill_label_col]).set_index(self.skill_label_col)

            for occ_id, skills_for_occ in esco_df.groupby("occupationUri"):
                if skills_for_occ.empty:
                    continue
                    
                first_row = skills_for_occ.iloc[0]
                job_text = build_job_text(first_row, text_fields, is_structured)
                
                skill_labels = skills_for_occ[self.skill_label_col].dropna().astype(str).unique()
                skill_texts = []
                for sl in skill_labels:
                    if sl in skill_lookup_df.index:
                        skill_row = skill_lookup_df.loc[sl]
                        skill_texts.append(build_skill_text(skill_row, text_fields, is_structured))

                if job_text and skill_texts:
                    self.samples.append({
                        "text": job_text,
                        "skill_texts": skill_texts,
                    })
        
        # This dataset doesn't need a label encoder, as the skill_texts
        # are the target for the contrastive loss.

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        return {
            "text": row["text"],            # The job text (anchor)
            "skill_texts": row["skill_texts"],  # List of skill texts (positives)
        }


def build_skill_lookups(
    esco_df: pd.DataFrame,
    hier_level: int,
    text_fields: str = "title+desc",
    is_structured: bool = False
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Creates the essential mappings needed for inference (indexing and re-ranking).

    Args:
        esco_df: The master ESCO dataframe.
        hier_level: The category level (0-3) to map against.
        text_fields: How to build the skill text.
        is_structured: Whether to use structured text.

    Returns:
        A tuple containing:
        - all_skill_labels: A list of all unique skill labels.
        - all_skill_texts: A list of all corresponding skill texts.
        - skill_to_cat_map: A dict mapping skill_label -> category_label.
    """
    cat_col = HIER_COL_MAP[hier_level]
    skill_label_col = "skillLabel"
    
    # De-duplicate skills, keeping the first occurrence
    unique_skills_df = esco_df.drop_duplicates(subset=[skill_label_col])
    
    all_skill_labels = []
    all_skill_texts = []
    skill_to_cat_map = {}
    
    for _, row in unique_skills_df.iterrows():
        label = row[skill_label_col]
        cat = row[cat_col]
        
        if pd.isna(label) or pd.isna(cat):
            continue
            
        label = str(label)
        cat = str(cat)
        
        all_skill_labels.append(label)
        all_skill_texts.append(build_skill_text(row, text_fields, is_structured))
        skill_to_cat_map[label] = cat
        
    return all_skill_labels, all_skill_texts, skill_to_cat_map