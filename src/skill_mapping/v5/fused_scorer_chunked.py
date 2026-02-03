"""
Fused Scorer with Chunked Processing for Large-Scale Data

Handles 400k+ jobs with 45GB+ Task B data through:
1. Streaming JSON parsing with ijson (no full file load)
2. Chunked evaluation to bound memory usage
3. Parallel chunk processing with joblib
4. Optional Parquet conversion for repeated runs

Usage:
    # One-time conversion (recommended for repeated experiments)
    python fused_scorer_chunked.py --convert-to-parquet \
        --task_b data/task_b.json \
        --parquet_output data/task_b_parquet/

    # Run with streaming JSON
    python fused_scorer_chunked.py \
        --esco_dir data/esco_datasets \
        --label_encoder models/label_encoder.json \
        --task_a data/task_a.jsonl \
        --task_b data/task_b.json \
        --isco_preds data/isco_preds.json \
        --decorte_map data/decorte_map.csv \
        --output_dir results/ \
        --chunk_size 5000

    # Run with pre-converted Parquet
    python fused_scorer_chunked.py \
        --esco_dir data/esco_datasets \
        --label_encoder models/label_encoder.json \
        --task_a data/task_a.jsonl \
        --task_b_parquet data/task_b_parquet/ \
        --isco_preds data/isco_preds.json \
        --decorte_map data/decorte_map.csv \
        --output_dir results/ \
        --chunk_size 5000
"""

import json
import logging
import gc
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple, Literal, Iterator, Any
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import itertools
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Type aliases
FusionStrategy = Literal['multiplicative', 'linear']
NormalizationMethod = Literal['minmax', 'zscore', 'rank']


# =============================================================================
# Statistics Classes (unchanged from original)
# =============================================================================

@dataclass
class JobScoringStats:
    """Statistics collected during job scoring for debugging purposes."""
    num_task_a_occupations: int = 0
    num_valid_task_a_occupations: int = 0
    num_task_a_candidates: int = 0
    num_with_task_b_scores: int = 0
    num_imputed: int = 0
    num_with_isco_mapping: int = 0
    num_without_isco_mapping: int = 0
    isco_probs_valid: bool = False
    fallback_mode: bool = False

    def to_dict(self) -> Dict:
        return {
            'task_a': {
                'num_occupations': self.num_task_a_occupations,
                'num_valid_occupations': self.num_valid_task_a_occupations,
                'num_candidates': self.num_task_a_candidates,
            },
            'task_b': {
                'num_with_scores': self.num_with_task_b_scores,
                'num_imputed': self.num_imputed,
                'coverage_pct': (self.num_with_task_b_scores / self.num_task_a_candidates * 100
                                 if self.num_task_a_candidates > 0 else 0.0),
            },
            'isco': {
                'num_with_mapping': self.num_with_isco_mapping,
                'num_without_mapping': self.num_without_isco_mapping,
                'probs_valid': self.isco_probs_valid,
                'coverage_pct': (self.num_with_isco_mapping / self.num_task_a_candidates * 100
                                 if self.num_task_a_candidates > 0 else 0.0),
            },
            'fallback_mode': self.fallback_mode,
        }


@dataclass
class AggregatedStats:
    """Aggregated statistics across multiple jobs."""
    num_jobs: int = 0
    total_candidates: int = 0
    total_with_task_b: int = 0
    total_imputed: int = 0
    total_with_isco: int = 0
    total_without_isco: int = 0
    num_fallback: int = 0
    _candidates_per_job: List[int] = field(default_factory=list)
    _task_b_coverage: List[float] = field(default_factory=list)
    _isco_coverage: List[float] = field(default_factory=list)

    def add(self, stats: JobScoringStats):
        self.num_jobs += 1
        self.total_candidates += stats.num_task_a_candidates
        self.total_with_task_b += stats.num_with_task_b_scores
        self.total_imputed += stats.num_imputed
        self.total_with_isco += stats.num_with_isco_mapping
        self.total_without_isco += stats.num_without_isco_mapping
        if stats.fallback_mode:
            self.num_fallback += 1
        self._candidates_per_job.append(stats.num_task_a_candidates)
        if stats.num_task_a_candidates > 0:
            self._task_b_coverage.append(stats.num_with_task_b_scores / stats.num_task_a_candidates)
            self._isco_coverage.append(stats.num_with_isco_mapping / stats.num_task_a_candidates)

    def summary(self) -> Dict:
        return {
            'num_jobs': self.num_jobs,
            'avg_candidates_per_job': np.mean(self._candidates_per_job) if self._candidates_per_job else 0,
            'median_candidates_per_job': np.median(self._candidates_per_job) if self._candidates_per_job else 0,
            'total_candidates': self.total_candidates,
            'task_b_coverage': {
                'total_with_scores': self.total_with_task_b,
                'total_imputed': self.total_imputed,
                'avg_coverage_pct': np.mean(self._task_b_coverage) * 100 if self._task_b_coverage else 0,
            },
            'isco_coverage': {
                'total_with_mapping': self.total_with_isco,
                'total_without_mapping': self.total_without_isco,
                'avg_coverage_pct': np.mean(self._isco_coverage) * 100 if self._isco_coverage else 0,
            },
            'num_fallback_jobs': self.num_fallback,
            'fallback_pct': self.num_fallback / self.num_jobs * 100 if self.num_jobs > 0 else 0,
        }

    def log_summary(self, level: int = logging.INFO):
        s = self.summary()
        logger.log(level, "=" * 60)
        logger.log(level, "SCORING STATISTICS SUMMARY")
        logger.log(level, "=" * 60)
        logger.log(level, f"Jobs processed: {s['num_jobs']}")
        logger.log(level, f"Candidates per job: avg={s['avg_candidates_per_job']:.1f}, "
                         f"median={s['median_candidates_per_job']:.0f}")
        logger.log(level, f"Task B coverage: {s['task_b_coverage']['avg_coverage_pct']:.1f}% avg")
        logger.log(level, f"ISCO coverage: {s['isco_coverage']['avg_coverage_pct']:.1f}% avg")
        logger.log(level, f"Fallback mode: {s['num_fallback_jobs']} jobs ({s['fallback_pct']:.1f}%)")
        logger.log(level, "=" * 60)


@dataclass 
class ChunkMetrics:
    """Metrics from a single chunk evaluation."""
    num_jobs: int
    sum_ap: float
    sum_r10: float
    sum_r50: float
    sum_r100: float
    
    def __add__(self, other: 'ChunkMetrics') -> 'ChunkMetrics':
        return ChunkMetrics(
            num_jobs=self.num_jobs + other.num_jobs,
            sum_ap=self.sum_ap + other.sum_ap,
            sum_r10=self.sum_r10 + other.sum_r10,
            sum_r50=self.sum_r50 + other.sum_r50,
            sum_r100=self.sum_r100 + other.sum_r100,
        )
    
    def to_means(self) -> Dict[str, float]:
        if self.num_jobs == 0:
            return {"mAP": 0.0, "R@10": 0.0, "R@50": 0.0, "R@100": 0.0}
        return {
            "mAP": self.sum_ap / self.num_jobs,
            "R@10": self.sum_r10 / self.num_jobs,
            "R@50": self.sum_r50 / self.num_jobs,
            "R@100": self.sum_r100 / self.num_jobs,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def clean_isco_code(code: str) -> Optional[str]:
    """Normalize ISCO code to a 4-digit string; returns None if invalid."""
    if code is None or str(code).lower() == "nan":
        return None
    s = str(code).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(4)[:4]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Streaming Data Loaders
# =============================================================================

def stream_task_b_json(
    json_path: Path,
    job_ids_filter: Optional[Set[str]] = None,
    chunk_size: int = 5000,
    top_k_skills: Optional[int] = None,
) -> Iterator[Dict[str, Dict[str, float]]]:
    """
    Stream Task B JSON file in chunks using ijson.
    
    Args:
        json_path: Path to the large JSON file.
        job_ids_filter: Optional set of job IDs to include (others skipped).
        chunk_size: Number of jobs per chunk.
        top_k_skills: If set, only keep top-k skills per job (reduces memory).
        
    Yields:
        Dict of {job_id: {skill_uri: score}} for each chunk.
    """
    try:
        import ijson
    except ImportError:
        raise ImportError("ijson is required for streaming. Install with: pip install ijson")
    
    logger.info(f"Streaming Task B from {json_path} (chunk_size={chunk_size})")
    
    current_chunk = {}
    total_yielded = 0
    
    with open(json_path, 'rb') as f:
        # ijson.kvitems streams key-value pairs from a JSON object
        parser = ijson.kvitems(f, '')
        
        for job_id, skills_data in parser:
            job_id_str = str(job_id)
            
            # Skip if not in filter set
            if job_ids_filter is not None and job_id_str not in job_ids_filter:
                continue
            
            # Parse skills data
            if isinstance(skills_data, list):
                # Format: [{"skill_uri": ..., "score": ...}, ...]
                if top_k_skills is not None:
                    # Sort by score and take top-k
                    skills_data = sorted(skills_data, key=lambda x: x.get('score', 0), reverse=True)[:top_k_skills]
                scores = {item['skill_uri']: float(item['score']) for item in skills_data}
            elif isinstance(skills_data, dict):
                # Format: {skill_uri: score, ...}
                scores = {k: float(v) for k, v in skills_data.items()}
                if top_k_skills is not None:
                    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k_skills]
                    scores = dict(sorted_items)
            else:
                logger.warning(f"Unexpected format for job {job_id_str}, skipping")
                continue
            
            current_chunk[job_id_str] = scores
            
            # Yield chunk when full
            if len(current_chunk) >= chunk_size:
                total_yielded += len(current_chunk)
                logger.info(f"Yielding chunk: {len(current_chunk)} jobs (total: {total_yielded})")
                yield current_chunk
                current_chunk = {}
                gc.collect()  # Help with memory
    
    # Yield remaining
    if current_chunk:
        total_yielded += len(current_chunk)
        logger.info(f"Yielding final chunk: {len(current_chunk)} jobs (total: {total_yielded})")
        yield current_chunk


def stream_task_b_parquet(
    parquet_path: Path,
    job_ids_filter: Optional[Set[str]] = None,
    chunk_size: int = 5000,
    job_id_remap: Optional[Dict[str, str]] = None,
) -> Iterator[Dict[str, Dict[str, float]]]:
    """
    Stream Task B from Parquet format in chunks.
    
    Expected Parquet schema:
        - job_id: string
        - skill_uri: string  
        - score: float
    
    Args:
        parquet_path: Path to Parquet file or directory.
        job_ids_filter: Optional set of job IDs to include (after remapping).
        chunk_size: Number of jobs per chunk.
        job_id_remap: Optional dict to remap job IDs (e.g., {"0": "203500", "1": "198735"}).
                      Used when Task B was indexed by line numbers instead of actual job IDs.
        
    Yields:
        Dict of {job_id: {skill_uri: score}} for each chunk.
    """
    import pyarrow.parquet as pq
    
    logger.info(f"Streaming Task B from Parquet: {parquet_path}")
    if job_id_remap:
        logger.info(f"Using job_id remapping ({len(job_id_remap)} mappings)")
    
    # Read Parquet in batches
    parquet_file = pq.ParquetFile(parquet_path) if parquet_path.is_file() else None
    
    if parquet_file:
        # Single file
        batches = parquet_file.iter_batches(batch_size=chunk_size * 100)  # Read more rows, group later
    else:
        # Directory of partitioned files
        dataset = pq.ParquetDataset(parquet_path)
        batches = dataset.to_batches(batch_size=chunk_size * 100)
    
    current_chunk = {}
    total_yielded = 0
    
    for batch in batches:
        df = batch.to_pandas()
        
        # Remap job IDs if provided
        if job_id_remap is not None:
            df['job_id'] = df['job_id'].map(lambda x: job_id_remap.get(str(x), str(x)))
        
        # Filter job IDs if specified
        if job_ids_filter is not None:
            df = df[df['job_id'].isin(job_ids_filter)]
        
        # Group by job_id
        for job_id, group in df.groupby('job_id'):
            job_id_str = str(job_id)
            scores = dict(zip(group['skill_uri'], group['score'].astype(float)))
            current_chunk[job_id_str] = scores
            
            if len(current_chunk) >= chunk_size:
                total_yielded += len(current_chunk)
                logger.info(f"Yielding chunk: {len(current_chunk)} jobs (total: {total_yielded})")
                yield current_chunk
                current_chunk = {}
                gc.collect()
    
    if current_chunk:
        total_yielded += len(current_chunk)
        logger.info(f"Yielding final chunk: {len(current_chunk)} jobs (total: {total_yielded})")
        yield current_chunk


def convert_json_to_parquet(
    json_path: Path,
    output_path: Path,
    chunk_size: int = 10000,
) -> None:
    """
    Convert large Task B JSON to Parquet format (one-time operation).
    
    Args:
        json_path: Path to input JSON file.
        output_path: Path for output Parquet file.
        chunk_size: Number of jobs to process before writing.
    """
    try:
        import ijson
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("ijson and pyarrow required. Install with: pip install ijson pyarrow")
    
    logger.info(f"Converting {json_path} to Parquet at {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    schema = pa.schema([
        ('job_id', pa.string()),
        ('skill_uri', pa.string()),
        ('score', pa.float32()),
    ])
    
    writer = None
    records = []
    total_jobs = 0
    
    with open(json_path, 'rb') as f:
        parser = ijson.kvitems(f, '')
        
        for job_id, skills_data in tqdm(parser, desc="Converting"):
            job_id_str = str(job_id)
            
            if isinstance(skills_data, list):
                for item in skills_data:
                    records.append({
                        'job_id': job_id_str,
                        'skill_uri': item['skill_uri'],
                        'score': float(item['score']),
                    })
            elif isinstance(skills_data, dict):
                for skill_uri, score in skills_data.items():
                    records.append({
                        'job_id': job_id_str,
                        'skill_uri': skill_uri,
                        'score': float(score),
                    })
            
            total_jobs += 1
            
            # Write batch
            if total_jobs % chunk_size == 0:
                df = pd.DataFrame(records)
                table = pa.Table.from_pandas(df, schema=schema)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression='snappy')
                
                writer.write_table(table)
                logger.info(f"Written {total_jobs} jobs...")
                records = []
                gc.collect()
    
    # Final write
    if records:
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema, compression='snappy')
        writer.write_table(table)
    
    if writer:
        writer.close()
    
    logger.info(f"Conversion complete: {total_jobs} jobs written to {output_path}")


# =============================================================================
# FusedScorer Class (with chunked evaluation)
# =============================================================================

class FusedScorer:
    def __init__(
        self,
        esco_dir: Path,
        isco_label_encoder_path: Path,
        essentials_only: bool = False,
        isco_level: Optional[int] = None,
    ):
        self.esco_dir = Path(esco_dir)
        self.isco_label_encoder_path = Path(isco_label_encoder_path)
        self.isco_level = isco_level
        self.essentials_only = essentials_only

        # Lookup tables
        self.occ_to_skills: Dict[str, Set[str]] = {}
        self.skill_to_isco: Dict[str, Set[str]] = {}
        self.skill_to_isco_counts: Dict[str, Dict[str, int]] = {}
        self.isco_index: List[str] = []

        # Affinity matrix
        self.skill_index: List[str] = []
        self.skill_uri_to_idx: Dict[str, int] = {}
        self.affinity_matrix: Optional[np.ndarray] = None
        self.affinity_mode: str = 'uniform'
        self.progress_log_every: int = 100

    def build_lookup_tables(self):
        """Builds the three required reference structures from ESCO taxonomy."""
        logger.info("Building lookup tables...")
        self._load_isco_index()
        self._build_mappings()
        logger.info(f"Built lookups: {len(self.occ_to_skills)} occupations, "
                   f"{len(self.skill_to_isco)} skills, {len(self.isco_index)} ISCO groups")

    def _load_isco_index(self):
        """Load ordered list of ISCO codes from classifier's label encoder."""
        if not self.isco_label_encoder_path.exists():
            raise FileNotFoundError(f"ISCO label encoder not found at {self.isco_label_encoder_path}")

        with open(self.isco_label_encoder_path, 'r') as f:
            data = json.load(f)

        if "idx2str" in data:
            self.isco_index = [data["idx2str"][str(i)] for i in range(len(data["idx2str"]))]
        elif "str2idx" in data:
            self.isco_index = sorted(data["str2idx"], key=data["str2idx"].get)
        else:
            raise ValueError("Invalid label encoder format")

        logger.info(f"Loaded {len(self.isco_index)} ISCO labels from encoder")

    def _build_mappings(self):
        """Build occupation-skills and skill-ISCO mappings."""
        occupations_path = self.esco_dir / "occupations_en.csv"
        relations_path = self.esco_dir / "occupationSkillRelations_en.csv"

        if not occupations_path.exists():
            raise FileNotFoundError(f"ESCO occupations file not found at {occupations_path}")
        if not relations_path.exists():
            raise FileNotFoundError(f"ESCO relations file not found at {relations_path}")

        logger.info(f"Loading occupations from {occupations_path}")
        occs_df = pd.read_csv(occupations_path, usecols=['conceptUri', 'iscoGroup'])

        valid_labels = set(self.isco_index)
        if self.isco_level is not None:
            label_len = self.isco_level
        elif not self.isco_index:
            label_len = 4
        else:
            label_len = len(self.isco_index[0])

        occ_isco_map = {}
        for _, row in occs_df.iterrows():
            uri = row['conceptUri']
            raw_code = clean_isco_code(row['iscoGroup'])
            if not raw_code:
                continue
            candidate = raw_code[:label_len]
            if candidate in valid_labels:
                occ_isco_map[uri] = candidate

        logger.info(f"Mapped {len(occ_isco_map)} occupations to known ISCO groups")

        logger.info(f"Loading relations from {relations_path}")
        rels_df = pd.read_csv(relations_path, usecols=['occupationUri', 'skillUri', 'relationType'])

        self.occ_to_skills = rels_df.groupby('occupationUri')['skillUri'].apply(set).to_dict()

        if self.essentials_only:
            essential_rels = rels_df[rels_df['relationType'] == 'essential'].copy()
            logger.info(f"Only essential skills: {essential_rels.shape}")
        else:
            essential_rels = rels_df.copy()
            logger.info(f"All related skills: {essential_rels.shape}")

        essential_rels['iscoGroup'] = essential_rels['occupationUri'].map(occ_isco_map)
        essential_rels = essential_rels.dropna(subset=['iscoGroup'])

        self.skill_to_isco = essential_rels.groupby('skillUri')['iscoGroup'].apply(set).to_dict()

        skill_isco_counts = essential_rels.groupby(['skillUri', 'iscoGroup']).size().reset_index(name='count')
        self.skill_to_isco_counts = {}
        for _, row in skill_isco_counts.iterrows():
            skill_uri = row['skillUri']
            isco_group = row['iscoGroup']
            count = row['count']
            if skill_uri not in self.skill_to_isco_counts:
                self.skill_to_isco_counts[skill_uri] = {}
            self.skill_to_isco_counts[skill_uri][isco_group] = count

        logger.info(f"Built skill-to-ISCO mapping for {len(self.skill_to_isco)} skills")

    def build_affinity_matrix(self, mode: str = 'uniform'):
        """Build Skill-ISCO Affinity Matrix."""
        logger.info(f"Building Skill-ISCO Affinity Matrix (mode={mode})...")
        self.affinity_mode = mode

        if not self.skill_to_isco:
            logger.warning("Skill-to-ISCO mapping is empty. Run build_lookup_tables() first.")
            return

        self.skill_index = sorted(self.skill_to_isco.keys())
        self.skill_uri_to_idx = {uri: i for i, uri in enumerate(self.skill_index)}

        n_skills = len(self.skill_index)
        n_groups = len(self.isco_index)

        self.affinity_matrix = np.zeros((n_skills, n_groups), dtype=np.float32)
        isco_label_to_idx = {label: i for i, label in enumerate(self.isco_index)}

        for i, skill_uri in enumerate(self.skill_index):
            groups = self.skill_to_isco.get(skill_uri, set())
            valid_groups = [(g, isco_label_to_idx[g]) for g in groups if g in isco_label_to_idx]

            if not valid_groups:
                continue

            if mode == 'uniform':
                prob = 1.0 / len(valid_groups)
                for _, col_idx in valid_groups:
                    self.affinity_matrix[i, col_idx] = prob
            elif mode == 'frequency':
                counts = self.skill_to_isco_counts.get(skill_uri, {})
                raw_weights = [(col_idx, counts.get(group, 1)) for group, col_idx in valid_groups]
                total = sum(w for _, w in raw_weights)
                if total > 0:
                    for col_idx, weight in raw_weights:
                        self.affinity_matrix[i, col_idx] = weight / total
            elif mode == 'binary':
                for _, col_idx in valid_groups:
                    self.affinity_matrix[i, col_idx] = 1.0
            else:
                raise ValueError(f"Unknown affinity mode: {mode}")

        logger.info(f"Built affinity matrix: {self.affinity_matrix.shape}")

    def get_skill_affinity_vector(self, skill_uri: str) -> Optional[np.ndarray]:
        """Returns the affinity vector for a given skill, or None if not found."""
        idx = self.skill_uri_to_idx.get(skill_uri)
        if idx is not None:
            return self.affinity_matrix[idx]
        return None

    # =========================================================================
    # Normalization Methods
    # =========================================================================

    @staticmethod
    def _normalize_minmax(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        min_val, max_val = np.min(values), np.max(values)
        if max_val - min_val < 1e-10:
            return np.full_like(values, 0.5)
        return (values - min_val) / (max_val - min_val)

    @staticmethod
    def _normalize_zscore(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        mean, std = np.mean(values), np.std(values)
        if std < 1e-10:
            return np.full_like(values, 0.5)
        z_scores = (values - mean) / std
        return 1.0 / (1.0 + np.exp(-z_scores))

    @staticmethod
    def _normalize_rank(values: np.ndarray) -> np.ndarray:
        if len(values) <= 1:
            return np.full_like(values, 0.5) if len(values) == 1 else values
        ranks = rankdata(values, method='average')
        return (ranks - 1) / (len(values) - 1)

    def normalize_scores(self, values: np.ndarray, method: NormalizationMethod = 'minmax') -> np.ndarray:
        if method == 'minmax':
            return self._normalize_minmax(values)
        elif method == 'zscore':
            return self._normalize_zscore(values)
        elif method == 'rank':
            return self._normalize_rank(values)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    # =========================================================================
    # Fusion Methods
    # =========================================================================

    def _fuse_multiplicative(
        self,
        base_scores: np.ndarray,
        isco_weights: np.ndarray,
        alpha: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 0.0
    ) -> np.ndarray:
        if epsilon > 0:
            isco_weights = np.maximum(isco_weights, epsilon)
        return np.power(base_scores, alpha) * np.power(isco_weights, gamma)

    def _fuse_linear(
        self,
        base_scores: np.ndarray,
        isco_weights: np.ndarray,
        alpha: float = 0.5,
        normalization: NormalizationMethod = 'minmax'
    ) -> np.ndarray:
        base_normalized = self.normalize_scores(base_scores, method=normalization)
        isco_normalized = self.normalize_scores(isco_weights, method=normalization)
        return alpha * base_normalized + (1 - alpha) * isco_normalized

    # =========================================================================
    # Core Scoring
    # =========================================================================

    def score_job(
        self,
        task_a_occupations: List[str],
        task_b_scores: Dict[str, float],
        isco_probs: np.ndarray,
        fusion_strategy: FusionStrategy = 'multiplicative',
        alpha: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 0.0,
        normalization: NormalizationMethod = 'minmax',
        collect_stats: bool = False
    ) -> Tuple[List[Tuple[str, float]], Optional[JobScoringStats]]:
        """Score and rank skills for a single job query."""
        stats = JobScoringStats() if collect_stats else None

        # Step 1: Generate candidate skills
        candidate_skills = set()
        num_valid_occs = 0
        for occ_uri in task_a_occupations:
            if occ_uri in self.occ_to_skills:
                candidate_skills.update(self.occ_to_skills[occ_uri])
                num_valid_occs += 1

        if stats:
            stats.num_task_a_occupations = len(task_a_occupations)
            stats.num_valid_task_a_occupations = num_valid_occs

        fallback_mode = False
        if not candidate_skills:
            fallback_mode = True
            candidate_skills = set(task_b_scores.keys())

        if stats:
            stats.fallback_mode = fallback_mode
            stats.num_task_a_candidates = len(candidate_skills)

        # Step 2: Get base scores
        min_score = min(task_b_scores.values()) if task_b_scores else 0.0

        skill_uris = []
        base_scores_list = []
        num_with_task_b = 0
        num_imputed = 0

        if fallback_mode:
            for skill_uri, score in task_b_scores.items():
                skill_uris.append(skill_uri)
                base_scores_list.append(score)
                num_with_task_b += 1
        else:
            for skill_uri in candidate_skills:
                skill_uris.append(skill_uri)
                if skill_uri in task_b_scores:
                    base_scores_list.append(task_b_scores[skill_uri])
                    num_with_task_b += 1
                else:
                    base_scores_list.append(min_score)
                    num_imputed += 1

        if stats:
            stats.num_with_task_b_scores = num_with_task_b
            stats.num_imputed = num_imputed

        if not skill_uris:
            return ([], stats)

        base_scores = np.array(base_scores_list, dtype=np.float32)

        # Step 3: Compute ISCO weights
        isco_valid = isco_probs is not None and len(isco_probs) > 0 and np.sum(isco_probs) > 0

        if stats:
            stats.isco_probs_valid = isco_valid

        num_with_isco = 0
        num_without_isco = 0

        if not isco_valid:
            isco_weights = np.ones(len(skill_uris), dtype=np.float32)
            num_without_isco = len(skill_uris)
        else:
            isco_weights = np.zeros(len(skill_uris), dtype=np.float32)
            for i, skill_uri in enumerate(skill_uris):
                skill_affinity = self.get_skill_affinity_vector(skill_uri)
                if skill_affinity is None:
                    isco_weights[i] = 1.0
                    num_without_isco += 1
                else:
                    isco_weights[i] = np.dot(skill_affinity, isco_probs)
                    num_with_isco += 1

        if stats:
            stats.num_with_isco_mapping = num_with_isco
            stats.num_without_isco_mapping = num_without_isco

        # Step 4: Apply fusion
        if fusion_strategy == 'multiplicative':
            final_scores_arr = self._fuse_multiplicative(base_scores, isco_weights, alpha, gamma, epsilon)
        elif fusion_strategy == 'linear':
            if epsilon > 0:
                isco_weights = np.maximum(isco_weights, epsilon)
            final_scores_arr = self._fuse_linear(base_scores, isco_weights, alpha, normalization)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # Step 5: Sort and return
        final_scores = list(zip(skill_uris, final_scores_arr.tolist()))
        final_scores.sort(key=lambda x: x[1], reverse=True)

        return (final_scores, stats)

    # =========================================================================
    # Chunked Evaluation (Main Addition)
    # =========================================================================

    def _compute_chunk_metrics(
        self,
        task_a_preds: Dict[str, List[str]],
        task_b_chunk: Dict[str, Dict[str, float]],
        isco_preds: Dict[str, np.ndarray],
        ground_truth: Dict[str, Set[str]],
        task_a_k: int,
        fusion_strategy: FusionStrategy,
        alpha: float,
        gamma: float,
        temperature: float,
        epsilon: float,
        normalization: NormalizationMethod,
        mode: str,
    ) -> ChunkMetrics:
        """Compute metrics for a single chunk of Task B data."""
        # Find common job IDs in this chunk
        chunk_job_ids = set(task_b_chunk.keys()) & set(ground_truth.keys())
        
        if mode != 'task_b_only':
            chunk_job_ids &= set(task_a_preds.keys())
            if mode == 'full':
                chunk_job_ids &= set(isco_preds.keys())
        
        if not chunk_job_ids:
            return ChunkMetrics(0, 0.0, 0.0, 0.0, 0.0)
        
        sum_ap = 0.0
        sum_r10 = 0.0
        sum_r50 = 0.0
        sum_r100 = 0.0
        num_jobs = 0
        
        for job_id in chunk_job_ids:
            skill_scores = task_b_chunk[job_id]
            true_skills = ground_truth[job_id]
            
            if mode == 'task_b_only':
                sorted_items = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)
                predicted_uris = [x[0] for x in sorted_items]
            elif mode == 'task_a_filter_only':
                occ_list = task_a_preds[job_id][:task_a_k]
                ranked_results, _ = self.score_job(
                    task_a_occupations=occ_list,
                    task_b_scores=skill_scores,
                    isco_probs=np.zeros(len(self.isco_index)),
                    fusion_strategy='multiplicative',
                    alpha=1.0,
                    gamma=0.0,
                    epsilon=0.0,
                    collect_stats=False
                )
                predicted_uris = [r[0] for r in ranked_results]
            elif mode == 'full':
                occ_list = task_a_preds[job_id][:task_a_k]
                probs = isco_preds[job_id].copy()
                
                if temperature != 1.0 and temperature > 0:
                    probs = np.power(probs, 1.0 / temperature)
                    prob_sum = np.sum(probs)
                    if prob_sum > 0:
                        probs = probs / prob_sum
                
                ranked_results, _ = self.score_job(
                    task_a_occupations=occ_list,
                    task_b_scores=skill_scores,
                    isco_probs=probs,
                    fusion_strategy=fusion_strategy,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    normalization=normalization,
                    collect_stats=False
                )
                predicted_uris = [r[0] for r in ranked_results]
            else:
                continue
            
            # Compute metrics
            if true_skills:
                # Recall@K
                for k, metric_list in [(10, 'r10'), (50, 'r50'), (100, 'r100')]:
                    top_k = set(predicted_uris[:k])
                    recall = len(top_k & true_skills) / len(true_skills)
                    if metric_list == 'r10':
                        sum_r10 += recall
                    elif metric_list == 'r50':
                        sum_r50 += recall
                    else:
                        sum_r100 += recall
                
                # AP
                hits = 0
                sum_precs = 0.0
                for rank, uri in enumerate(predicted_uris):
                    if uri in true_skills:
                        hits += 1
                        sum_precs += hits / (rank + 1)
                sum_ap += sum_precs / len(true_skills)
            
            num_jobs += 1
        
        return ChunkMetrics(num_jobs, sum_ap, sum_r10, sum_r50, sum_r100)

    def evaluate_chunked_streaming(
        self,
        task_a_preds: Dict[str, List[str]],
        task_b_iterator: Iterator[Dict[str, Dict[str, float]]],
        isco_preds: Dict[str, np.ndarray],
        ground_truth: Dict[str, Set[str]],
        task_a_k: int = 5,
        fusion_strategy: FusionStrategy = 'multiplicative',
        alpha: float = 1.0,
        gamma: float = 1.0,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        normalization: NormalizationMethod = 'minmax',
        mode: str = 'full',
    ) -> Dict[str, float]:
        """
        Evaluate with streaming Task B data.
        
        Args:
            task_a_preds: Full Task A predictions (should fit in memory).
            task_b_iterator: Iterator yielding chunks of Task B data.
            isco_preds: Full ISCO predictions (should fit in memory).
            ground_truth: Full ground truth (should fit in memory).
            Other args: Fusion parameters.
            
        Returns:
            Dict of metrics (mAP, R@10, R@50, R@100).
        """
        logger.info(f"Starting chunked evaluation (mode={mode}, strategy={fusion_strategy})")
        
        total_metrics = ChunkMetrics(0, 0.0, 0.0, 0.0, 0.0)
        chunk_num = 0
        
        for task_b_chunk in task_b_iterator:
            chunk_num += 1
            chunk_metrics = self._compute_chunk_metrics(
                task_a_preds=task_a_preds,
                task_b_chunk=task_b_chunk,
                isco_preds=isco_preds,
                ground_truth=ground_truth,
                task_a_k=task_a_k,
                fusion_strategy=fusion_strategy,
                alpha=alpha,
                gamma=gamma,
                temperature=temperature,
                epsilon=epsilon,
                normalization=normalization,
                mode=mode,
            )
            total_metrics = total_metrics + chunk_metrics
            logger.info(f"Chunk {chunk_num}: {chunk_metrics.num_jobs} jobs processed "
                       f"(cumulative: {total_metrics.num_jobs})")
            gc.collect()
        
        results = total_metrics.to_means()
        logger.info(f"Final metrics: {results}")
        return results

    def evaluate_chunked_with_predictions(
        self,
        task_a_preds: Dict[str, List[str]],
        task_b_iterator: Iterator[Dict[str, Dict[str, float]]],
        isco_preds: Dict[str, np.ndarray],
        ground_truth: Dict[str, Set[str]],
        output_predictions_path: Path,
        task_a_k: int = 5,
        fusion_strategy: FusionStrategy = 'multiplicative',
        alpha: float = 1.0,
        gamma: float = 1.0,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        normalization: NormalizationMethod = 'minmax',
        top_k_output: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate and stream predictions to file (avoids holding all predictions in memory).
        
        Predictions are written incrementally to a JSONL file.
        """
        logger.info(f"Starting chunked evaluation with prediction output to {output_predictions_path}")
        
        output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_metrics = ChunkMetrics(0, 0.0, 0.0, 0.0, 0.0)
        chunk_num = 0
        
        with open(output_predictions_path, 'w') as pred_file:
            for task_b_chunk in task_b_iterator:
                chunk_num += 1
                chunk_job_ids = (set(task_b_chunk.keys()) & set(ground_truth.keys()) &
                                set(task_a_preds.keys()) & set(isco_preds.keys()))
                
                chunk_sum_ap = 0.0
                chunk_sum_r10 = 0.0
                chunk_sum_r50 = 0.0
                chunk_sum_r100 = 0.0
                chunk_num_jobs = 0
                
                for job_id in chunk_job_ids:
                    skill_scores = task_b_chunk[job_id]
                    true_skills = ground_truth[job_id]
                    
                    occ_list = task_a_preds[job_id][:task_a_k]
                    probs = isco_preds[job_id].copy()
                    
                    if temperature != 1.0 and temperature > 0:
                        probs = np.power(probs, 1.0 / temperature)
                        prob_sum = np.sum(probs)
                        if prob_sum > 0:
                            probs = probs / prob_sum
                    
                    ranked_results, _ = self.score_job(
                        task_a_occupations=occ_list,
                        task_b_scores=skill_scores,
                        isco_probs=probs,
                        fusion_strategy=fusion_strategy,
                        alpha=alpha,
                        gamma=gamma,
                        epsilon=epsilon,
                        normalization=normalization,
                        collect_stats=False
                    )
                    
                    # Write prediction (top-k only)
                    pred_entry = {
                        'job_id': job_id,
                        'predictions': [{'skill_uri': uri, 'score': score} 
                                       for uri, score in ranked_results[:top_k_output]]
                    }
                    pred_file.write(json.dumps(pred_entry) + '\n')
                    
                    # Compute metrics
                    predicted_uris = [r[0] for r in ranked_results]
                    
                    if true_skills:
                        for k in [10, 50, 100]:
                            top_k = set(predicted_uris[:k])
                            recall = len(top_k & true_skills) / len(true_skills)
                            if k == 10:
                                chunk_sum_r10 += recall
                            elif k == 50:
                                chunk_sum_r50 += recall
                            else:
                                chunk_sum_r100 += recall
                        
                        hits = 0
                        sum_precs = 0.0
                        for rank, uri in enumerate(predicted_uris):
                            if uri in true_skills:
                                hits += 1
                                sum_precs += hits / (rank + 1)
                        chunk_sum_ap += sum_precs / len(true_skills)
                    
                    chunk_num_jobs += 1
                
                chunk_metrics = ChunkMetrics(chunk_num_jobs, chunk_sum_ap, 
                                            chunk_sum_r10, chunk_sum_r50, chunk_sum_r100)
                total_metrics = total_metrics + chunk_metrics
                
                logger.info(f"Chunk {chunk_num}: {chunk_metrics.num_jobs} jobs "
                           f"(cumulative: {total_metrics.num_jobs})")
                gc.collect()
        
        results = total_metrics.to_means()
        logger.info(f"Final metrics: {results}")
        logger.info(f"Predictions written to {output_predictions_path}")
        return results

    def grid_search_chunked(
        self,
        task_a_preds: Dict[str, List[str]],
        task_b_path: Path,
        isco_preds: Dict[str, np.ndarray],
        ground_truth: Dict[str, Set[str]],
        valid_job_ids: Set[str],
        chunk_size: int = 5000,
        k_values: List[int] = [3, 5, 7, 10],
        fusion_strategies: List[FusionStrategy] = ['multiplicative'],
        alpha_values: List[float] = [1.0],
        gamma_values: List[float] = [0.0, 0.5, 1.0],
        temperatures: List[float] = [1.0],
        affinity_modes: List[str] = ['uniform'],
        epsilon_values: List[float] = [0.0],
        normalization_methods: List[NormalizationMethod] = ['minmax'],
        metric: str = 'mAP',
        use_parquet: bool = False,
        job_id_remap: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Grid search with chunked Task B loading.
        
        NOTE: For large grid searches, this reloads Task B for each hyperparameter
        combination. For faster iteration, consider:
        1. Reducing search space
        2. Using a validation subset
        3. Converting to Parquet for faster loading
        """
        results = []
        best_score = -1.0
        best_params = None
        
        # Build parameter combinations
        param_combinations = []
        
        for aff_mode in affinity_modes:
            for k in k_values:
                for strategy in fusion_strategies:
                    if strategy == 'multiplicative':
                        for alpha in alpha_values:
                            for gamma in gamma_values:
                                for temp in temperatures:
                                    for eps in epsilon_values:
                                        param_combinations.append({
                                            'k': k, 'fusion_strategy': strategy,
                                            'alpha': alpha, 'gamma': gamma,
                                            'temp': temp, 'affinity_mode': aff_mode,
                                            'epsilon': eps, 'normalization': 'minmax'
                                        })
                    else:  # linear
                        for alpha in alpha_values:
                            for temp in temperatures:
                                for eps in epsilon_values:
                                    for norm in normalization_methods:
                                        param_combinations.append({
                                            'k': k, 'fusion_strategy': strategy,
                                            'alpha': alpha, 'gamma': 0.0,
                                            'temp': temp, 'affinity_mode': aff_mode,
                                            'epsilon': eps, 'normalization': norm
                                        })
        
        logger.info(f"Grid search: {len(param_combinations)} parameter combinations")
        
        current_affinity_mode = None
        
        for i, params in enumerate(tqdm(param_combinations, desc="Grid search")):
            # Switch affinity mode if needed
            if params['affinity_mode'] != current_affinity_mode:
                logger.info(f"Switching affinity mode to '{params['affinity_mode']}'")
                self.build_affinity_matrix(mode=params['affinity_mode'])
                current_affinity_mode = params['affinity_mode']
            
            # Create iterator for this evaluation
            if use_parquet:
                task_b_iter = stream_task_b_parquet(task_b_path, valid_job_ids, chunk_size, job_id_remap)
            else:
                task_b_iter = stream_task_b_json(task_b_path, valid_job_ids, chunk_size)
            
            # Evaluate
            metrics = self.evaluate_chunked_streaming(
                task_a_preds=task_a_preds,
                task_b_iterator=task_b_iter,
                isco_preds=isco_preds,
                ground_truth=ground_truth,
                task_a_k=params['k'],
                fusion_strategy=params['fusion_strategy'],
                alpha=params['alpha'],
                gamma=params['gamma'],
                temperature=params['temp'],
                epsilon=params['epsilon'],
                normalization=params['normalization'],
                mode='full',
            )
            
            metrics['params'] = params
            results.append(metrics)
            
            score = metrics.get(metric, 0.0)
            if score > best_score:
                best_score = score
                best_params = params
            
            logger.info(f"[{i+1}/{len(param_combinations)}] {params} -> {metric}={score:.4f}")
        
        logger.info(f"Best {metric}: {best_score:.4f} with params {best_params}")
        return results, best_params


# =============================================================================
# Main Data Loading and Running Function
# =============================================================================

def load_data_and_run_chunked(
    scorer: FusedScorer,
    task_a_path: Path,
    task_b_path: Path,
    isco_path: Path,
    ground_truth_paths: Dict[str, Path],
    output_dir: Path,
    args,
):
    """
    Load data and run evaluation with chunked Task B processing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    logger.info("=" * 60)
    logger.info("LOADING DATA (Task A, ISCO, Ground Truth - fits in memory)")
    logger.info("=" * 60)

    # 1. Load Ground Truth FIRST - we need job_id ordering for Task A
    logger.info("Loading Ground Truth & Splits...")
    decorte_df = pd.read_csv(ground_truth_paths['decorte_map'])
    decorte_df['job_id'] = decorte_df['job_id'].astype(str)
    
    # Create a list of job_ids in order (for aligning Task A predictions)
    ordered_job_ids = decorte_df['job_id'].tolist()
    logger.info(f"Ground Truth has {len(ordered_job_ids)} rows, {decorte_df['job_id'].nunique()} unique job_ids")
    logger.info(f"Sample Ground Truth job IDs: {ordered_job_ids[:5]}")

    job_to_occ = dict(zip(decorte_df['job_id'], decorte_df['esco_id']))

    splits = {}
    if 'split' in decorte_df.columns:
        splits = dict(zip(decorte_df['job_id'], decorte_df['split']))
        logger.info(f"Found splits: {decorte_df['split'].unique()}")

    ground_truth = {}
    for job_id, occ_uri in job_to_occ.items():
        if occ_uri in scorer.occ_to_skills:
            ground_truth[job_id] = scorer.occ_to_skills[occ_uri]
        else:
            ground_truth[job_id] = set()
    logger.info(f"Built Ground Truth for {len(ground_truth)} jobs")

    # 2. Load Task A (JSONL) - use job_id from ground truth CSV for alignment
    logger.info(f"Loading Task A from {task_a_path}")
    task_a_preds = {}
    with open(task_a_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line)
            # Use job_id from the data if available, otherwise use aligned job_id from ground truth
            if 'job_id' in data:
                job_id = str(data['job_id'])
            elif i < len(ordered_job_ids):
                job_id = ordered_job_ids[i]
            else:
                # Fallback to line number if we've exceeded ground truth
                job_id = str(i)
                logger.warning(f"Task A line {i} exceeds ground truth length, using line number as job_id")
            task_a_preds[job_id] = data.get('predicted_esco_ids', [])
    logger.info(f"Loaded Task A: {len(task_a_preds)} jobs")
    logger.info(f"Sample Task A job IDs: {list(task_a_preds.keys())[:5]}")

    # 3. Load ISCO Predictions (JSON) - typically manageable
    logger.info(f"Loading ISCO preds from {isco_path}")
    with open(isco_path, 'r') as f:
        raw_isco = json.load(f)

    label_to_idx = {lbl: i for i, lbl in enumerate(scorer.isco_index)}
    n_classes = len(scorer.isco_index)

    isco_preds = {}
    for job_id, data in raw_isco.items():
        prob_vec = np.zeros(n_classes, dtype=np.float32)
        if "topk" in data:
            for item in data["topk"]:
                lbl = item["label"]
                score = item["score"]
                if lbl in label_to_idx:
                    prob_vec[label_to_idx[lbl]] = score
        s = np.sum(prob_vec)
        if s > 0:
            prob_vec /= s
        isco_preds[str(job_id)] = prob_vec
    logger.info(f"Loaded ISCO preds: {len(isco_preds)} jobs")
    logger.info(f"Sample ISCO job IDs: {list(isco_preds.keys())[:5]}")

    # 4. Create job_id remap for Task B (which may use line indices instead of actual job IDs)
    # This maps "0" -> "203500", "1" -> "198735", etc. based on ground truth ordering
    job_id_remap = {str(i): job_id for i, job_id in enumerate(ordered_job_ids)}
    logger.info(f"Created job_id remap with {len(job_id_remap)} entries")
    logger.info(f"Sample remap: '0' -> '{job_id_remap.get('0', 'N/A')}', '1' -> '{job_id_remap.get('1', 'N/A')}'")

    # 5. Determine valid job IDs (intersection of all datasets)
    # We need a first pass to know which job IDs are in Task B
    # For very large files, you might have this pre-computed
    logger.info("Determining valid job IDs...")
    
    all_job_ids = set(task_a_preds.keys()) & set(isco_preds.keys()) & set(ground_truth.keys())
    logger.info(f"Jobs with Task A, ISCO, and Ground Truth: {len(all_job_ids)}")

    # Split determination
    if splits:
        train_keys = {'train', 'training'}
        val_keys = {'val', 'validation', 'valid', 'dev'}
        test_keys = {'test', 'testing'}

        train_ids = {jid for jid in all_job_ids if splits.get(jid, '').lower() in train_keys}
        val_ids = {jid for jid in all_job_ids if splits.get(jid, '').lower() in val_keys}
        test_ids = {jid for jid in all_job_ids if splits.get(jid, '').lower() in test_keys}

        search_split_arg = args.grid_search_split if hasattr(args, 'grid_search_split') else 'validation'

        if search_split_arg == 'train':
            search_ids = train_ids
        elif search_split_arg == 'train+validation':
            search_ids = train_ids | val_ids
        else:
            search_ids = val_ids

        if not search_ids:
            logger.warning(f"No IDs for grid search split '{search_split_arg}'! Using all.")
            search_ids = all_job_ids
        else:
            logger.info(f"Using {len(search_ids)} {search_split_arg.upper()} jobs for Grid Search")

        eval_ids = test_ids if test_ids else search_ids
        logger.info(f"Using {len(eval_ids)} TEST jobs for Final Evaluation")
    else:
        search_ids = all_job_ids
        eval_ids = all_job_ids

    # 5. Grid Search (if enabled)
    chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else 5000
    use_parquet = hasattr(args, 'task_b_parquet') and args.task_b_parquet is not None
    
    if use_parquet:
        task_b_source = Path(args.task_b_parquet)
        logger.info(f"Using Parquet source: {task_b_source}")
    else:
        task_b_source = task_b_path
        logger.info(f"Using JSON source: {task_b_source}")

    # Prepare hyperparameter ranges
    k_values = [args.task_a_k] if args.task_a_k else [3, 5, 7, 10]
    fusion_strategies = [args.fusion_strategy] if args.fusion_strategy else ['multiplicative', 'linear']
    alpha_values = [args.alpha] if args.alpha else [0.3, 0.5, 0.7, 0.9, 1.0]
    gamma_values = [args.gamma] if args.gamma else [0.0, 0.25, 0.5, 0.75, 1.0]
    temperatures = [args.temperature] if args.temperature else [1.0]
    affinity_modes = [args.affinity_mode] if args.affinity_mode else ['uniform']
    epsilon_values = [args.epsilon] if args.epsilon else [0.0]
    normalization_methods = [args.normalization] if args.normalization else ['minmax']

    if args.skip_grid_search:
        logger.info("Skipping grid search, using provided parameters")
        best_params = {
            'k': args.task_a_k or 5,
            'fusion_strategy': args.fusion_strategy or 'multiplicative',
            'alpha': args.alpha or 1.0,
            'gamma': args.gamma or 0.5,
            'temp': args.temperature or 1.0,
            'affinity_mode': args.affinity_mode or 'uniform',
            'epsilon': args.epsilon or 0.0,
            'normalization': args.normalization or 'minmax',
        }
        grid_results = []
    else:
        logger.info("=" * 60)
        logger.info("GRID SEARCH (Chunked)")
        logger.info("=" * 60)
        
        grid_results, best_params = scorer.grid_search_chunked(
            task_a_preds=task_a_preds,
            task_b_path=task_b_source,
            isco_preds=isco_preds,
            ground_truth=ground_truth,
            valid_job_ids=search_ids,
            chunk_size=chunk_size,
            k_values=k_values,
            fusion_strategies=fusion_strategies,
            alpha_values=alpha_values,
            gamma_values=gamma_values,
            temperatures=temperatures,
            affinity_modes=affinity_modes,
            epsilon_values=epsilon_values,
            normalization_methods=normalization_methods,
            use_parquet=use_parquet,
            job_id_remap=job_id_remap if use_parquet else None,
        )

    # 6. Final Evaluation on Test Set
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION (TEST SET)")
    logger.info("=" * 60)

    # Ensure correct affinity mode
    if scorer.affinity_mode != best_params['affinity_mode']:
        scorer.build_affinity_matrix(mode=best_params['affinity_mode'])

    # Baseline: Task B Only
    logger.info("Evaluating baseline: Task B Only")
    task_b_iter = (stream_task_b_parquet(task_b_source, eval_ids, chunk_size, job_id_remap if use_parquet else None) if use_parquet
                   else stream_task_b_json(task_b_source, eval_ids, chunk_size))
    res_b = scorer.evaluate_chunked_streaming(
        task_a_preds=task_a_preds,
        task_b_iterator=task_b_iter,
        isco_preds=isco_preds,
        ground_truth=ground_truth,
        mode='task_b_only',
    )
    logger.info(f"Baseline (Task B Only): {res_b}")

    # Baseline: Task A Filter Only
    logger.info("Evaluating baseline: Task A Filter Only")
    task_b_iter = (stream_task_b_parquet(task_b_source, eval_ids, chunk_size, job_id_remap if use_parquet else None) if use_parquet
                   else stream_task_b_json(task_b_source, eval_ids, chunk_size))
    res_a = scorer.evaluate_chunked_streaming(
        task_a_preds=task_a_preds,
        task_b_iterator=task_b_iter,
        isco_preds=isco_preds,
        ground_truth=ground_truth,
        task_a_k=best_params['k'],
        mode='task_a_filter_only',
    )
    logger.info(f"Baseline (Task A Filter Only): {res_a}")

    # Full Pipeline with Best Params
    logger.info("Evaluating Full Pipeline with best params")
    task_b_iter = (stream_task_b_parquet(task_b_source, eval_ids, chunk_size, job_id_remap if use_parquet else None) if use_parquet
                   else stream_task_b_json(task_b_source, eval_ids, chunk_size))
    res_full = scorer.evaluate_chunked_streaming(
        task_a_preds=task_a_preds,
        task_b_iterator=task_b_iter,
        isco_preds=isco_preds,
        ground_truth=ground_truth,
        task_a_k=best_params['k'],
        fusion_strategy=best_params['fusion_strategy'],
        alpha=best_params['alpha'],
        gamma=best_params['gamma'],
        temperature=best_params['temp'],
        epsilon=best_params['epsilon'],
        normalization=best_params['normalization'],
        mode='full',
    )
    logger.info(f"Full Pipeline: {res_full}")

    # 7. Generate Predictions for ALL Data
    logger.info("=" * 60)
    logger.info("GENERATING PREDICTIONS (ALL DATA)")
    logger.info("=" * 60)

    predictions_path = output_dir / "fused_predictions.jsonl"
    task_b_iter = (stream_task_b_parquet(task_b_source, all_job_ids, chunk_size, job_id_remap if use_parquet else None) if use_parquet
                   else stream_task_b_json(task_b_source, all_job_ids, chunk_size))
    
    res_all = scorer.evaluate_chunked_with_predictions(
        task_a_preds=task_a_preds,
        task_b_iterator=task_b_iter,
        isco_preds=isco_preds,
        ground_truth=ground_truth,
        output_predictions_path=predictions_path,
        task_a_k=best_params['k'],
        fusion_strategy=best_params['fusion_strategy'],
        alpha=best_params['alpha'],
        gamma=best_params['gamma'],
        temperature=best_params['temp'],
        epsilon=best_params['epsilon'],
        normalization=best_params['normalization'],
        top_k_output=args.top_k_output if hasattr(args, 'top_k_output') else 100,
    )

    # 8. Save Results
    final_results = {
        "grid_search": grid_results,
        "best_params": best_params,
        "test_split_metrics": {
            "baselines": {
                "task_b_only": res_b,
                "task_a_filter_only": res_a
            },
            "full_pipeline": res_full
        },
        "all_data_metrics": res_all
    }

    results_path = output_dir / "fused_scorer_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved results to {results_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fused Scorer with Chunked Processing for Large-Scale Data"
    )

    # Mode selection
    parser.add_argument("--convert-to-parquet", action="store_true",
                        help="Convert Task B JSON to Parquet format (one-time)")

    # ESCO/ISCO configuration
    parser.add_argument("--esco_dir", type=str, default="data/esco_datasets")
    parser.add_argument("--label_encoder", type=str, required=False)
    parser.add_argument("--isco_level", type=int, default=None)
    parser.add_argument("--use_essentials", action='store_true')

    # Data paths
    parser.add_argument("--task_a", type=str, help="Path to Task A predictions (jsonl)")
    parser.add_argument("--task_b", type=str, help="Path to Task B predictions (json)")
    parser.add_argument("--task_b_parquet", type=str, help="Path to Task B Parquet (alternative)")
    parser.add_argument("--parquet_output", type=str, help="Output path for Parquet conversion")
    parser.add_argument("--isco_preds", type=str, help="Path to ISCO predictions (json)")
    parser.add_argument("--decorte_map", type=str, help="Path to job_id -> esco_id mapping CSV")
    parser.add_argument("--output_dir", type=str, default=".")

    # Chunking configuration
    parser.add_argument("--chunk_size", type=int, default=5000,
                        help="Number of jobs per chunk (default: 5000)")
    parser.add_argument("--top_k_skills", type=int, default=None,
                        help="Only load top-k skills per job (reduces memory)")
    parser.add_argument("--top_k_output", type=int, default=100,
                        help="Number of predictions to save per job")

    # Fusion hyperparameters
    parser.add_argument("--fusion_strategy", type=str, default=None,
                        choices=['multiplicative', 'linear'])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--normalization", type=str, default=None,
                        choices=['minmax', 'zscore', 'rank'])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--task_a_k", type=int, default=None)
    parser.add_argument("--affinity_mode", type=str, default=None,
                        choices=['uniform', 'frequency', 'binary'])

    # Execution configuration
    parser.add_argument("--skip_grid_search", action="store_true",
                        help="Skip grid search, use provided params directly")
    parser.add_argument("--grid_search_split", type=str, default="validation",
                        choices=["train", "validation", "train+validation"])

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Handle Parquet conversion mode
    if args.convert_to_parquet:
        if not args.task_b or not args.parquet_output:
            parser.error("--convert-to-parquet requires --task_b and --parquet_output")
        convert_json_to_parquet(
            json_path=Path(args.task_b),
            output_path=Path(args.parquet_output),
            chunk_size=args.chunk_size,
        )
        sys.exit(0)

    # Normal evaluation mode
    if not args.label_encoder:
        parser.error("--label_encoder is required for evaluation")

    scorer = FusedScorer(
        args.esco_dir,
        args.label_encoder,
        isco_level=args.isco_level,
        essentials_only=args.use_essentials
    )
    scorer.build_lookup_tables()
    scorer.build_affinity_matrix(mode=args.affinity_mode or 'uniform')

    if args.task_a and (args.task_b or args.task_b_parquet) and args.isco_preds and args.decorte_map:
        task_b_path = Path(args.task_b) if args.task_b else None
        load_data_and_run_chunked(
            scorer=scorer,
            task_a_path=Path(args.task_a),
            task_b_path=task_b_path,
            isco_path=Path(args.isco_preds),
            ground_truth_paths={'decorte_map': args.decorte_map},
            output_dir=Path(args.output_dir),
            args=args,
        )
    else:
        logger.error("Missing required data paths. Use --help for usage.")
        sys.exit(1)