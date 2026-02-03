import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple, Literal
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Type aliases for fusion configuration
FusionStrategy = Literal['multiplicative', 'linear']
NormalizationMethod = Literal['minmax', 'zscore', 'rank']


@dataclass
class JobScoringStats:
    """Statistics collected during job scoring for debugging purposes."""
    # Task A statistics
    num_task_a_occupations: int = 0
    num_valid_task_a_occupations: int = 0  # Occupations with skill mappings
    num_task_a_candidates: int = 0  # Total candidate skills from Task A
    
    # Task B statistics  
    num_with_task_b_scores: int = 0  # Candidates that had Task B scores
    num_imputed: int = 0  # Candidates that needed score imputation
    
    # ISCO statistics
    num_with_isco_mapping: int = 0  # Candidates with valid ISCO affinity vectors
    num_without_isco_mapping: int = 0  # Candidates without ISCO mapping (neutral weight)
    isco_probs_valid: bool = False  # Whether ISCO probs were valid
    
    # Mode flags
    fallback_mode: bool = False  # True if Task A produced no candidates
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
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
    
    def __str__(self) -> str:
        total = self.num_task_a_candidates
        task_b_pct = self.num_with_task_b_scores / total * 100 if total > 0 else 0
        isco_pct = self.num_with_isco_mapping / total * 100 if total > 0 else 0
        return (
            f"TaskA: {self.num_valid_task_a_occupations}/{self.num_task_a_occupations} valid occs → "
            f"{total} candidates | "
            f"TaskB: {self.num_with_task_b_scores} scored + {self.num_imputed} imputed ({task_b_pct:.1f}% coverage) | "
            f"ISCO: {self.num_with_isco_mapping} mapped ({isco_pct:.1f}% coverage)"
            + (" [FALLBACK]" if self.fallback_mode else "")
        )


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
    
    # For computing averages
    _candidates_per_job: List[int] = field(default_factory=list)
    _task_b_coverage: List[float] = field(default_factory=list)
    _isco_coverage: List[float] = field(default_factory=list)
    
    def add(self, stats: JobScoringStats):
        """Add a single job's stats to the aggregate."""
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
        """Return summary statistics."""
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
        """Log a summary of the aggregated statistics."""
        s = self.summary()
        logger.log(level, "="*60)
        logger.log(level, "SCORING STATISTICS SUMMARY")
        logger.log(level, "="*60)
        logger.log(level, f"Jobs processed: {s['num_jobs']}")
        logger.log(level, f"Candidates per job: avg={s['avg_candidates_per_job']:.1f}, median={s['median_candidates_per_job']:.0f}")
        logger.log(level, f"Task B coverage: {s['task_b_coverage']['avg_coverage_pct']:.1f}% avg "
                         f"({s['task_b_coverage']['total_with_scores']} scored, {s['task_b_coverage']['total_imputed']} imputed)")
        logger.log(level, f"ISCO coverage: {s['isco_coverage']['avg_coverage_pct']:.1f}% avg "
                         f"({s['isco_coverage']['total_with_mapping']} mapped, {s['isco_coverage']['total_without_mapping']} unmapped)")
        logger.log(level, f"Fallback mode: {s['num_fallback_jobs']} jobs ({s['fallback_pct']:.1f}%)")
        logger.log(level, "="*60)

def clean_isco_code(code: str) -> Optional[str]:
    """Normalize ISCO code to a 4-digit string; returns None if invalid."""
    if code is None or str(code).lower() == "nan":
        return None
    s = str(code).strip()
    # Remove trailing .0 if coming from float CSV
    if s.endswith(".0"):
        s = s[:-2]
    # Keep only digits
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(4)[:4]

class FusedScorer:
    def __init__(
        self,
        esco_dir: Path,
        isco_label_encoder_path: Path,
        essentials_only: Optional[int] = False,
        isco_level: Optional[int] = None,
    ):
        self.esco_dir = Path(esco_dir)
        self.isco_label_encoder_path = Path(isco_label_encoder_path)
        self.isco_level = isco_level
        self.essentials_only = essentials_only
        
        # Lookup tables
        self.occ_to_skills: Dict[str, Set[str]] = {}
        self.skill_to_isco: Dict[str, Set[str]] = {}
        self.skill_to_isco_counts: Dict[str, Dict[str, int]] = {}  # skill -> {isco_group -> occupation_count}
        self.isco_index: List[str] = []
        
        # Affinity matrix (built later)
        self.skill_index: List[str] = []
        self.skill_uri_to_idx: Dict[str, int] = {}
        self.affinity_matrix: Optional[np.ndarray] = None
        self.affinity_mode: str = 'uniform'  # 'uniform', 'frequency', or 'binary'
        # Default interval (in jobs) for periodic progress logging during batch evaluations.
        # Can be overridden by passing `progress_log_every` to `evaluate_batch`.
        self.progress_log_every: int = 100
        
    def build_lookup_tables(self):
        """Builds the three required reference structures from ESCO taxonomy."""
        logger.info("Building lookup tables...")
        
        # 1c. ISCO group index
        self._load_isco_index()
        
        # 1a. Occupation-to-Skills mapping
        # 1b. Skill-to-ISCO mapping
        self._build_mappings()
        
        logger.info(f"Built lookups: {len(self.occ_to_skills)} occupations, {len(self.skill_to_isco)} skills, {len(self.isco_index)} ISCO groups")

    def _load_isco_index(self):
        """Load ordered list of ISCO codes from classifier's label encoder."""
        if not self.isco_label_encoder_path.exists():
            raise FileNotFoundError(f"ISCO label encoder not found at {self.isco_label_encoder_path}")
            
        with open(self.isco_label_encoder_path, 'r') as f:
            data = json.load(f)
            
        # Standard SingleLabelEncoder format: {"str2idx": {...}, "idx2str": {...}}
        if "idx2str" in data:
            # keys are stringified integers, ensure we sort by integer key
            self.isco_index = [data["idx2str"][str(i)] for i in range(len(data["idx2str"]))]
        elif "str2idx" in data:
            # Sort by value to get ordered keys
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

        # --- Load Occupations & Map to ISCO Labels ---
        logger.info(f"Loading occupations from {occupations_path}")
        occs_df = pd.read_csv(occupations_path, usecols=['conceptUri', 'iscoGroup'])
        
        # We need to map occupation URI -> ISCO Group Label (that matches our classifier)
        # The classifier labels might be 1, 2, 3, or 4 digits. 
        # We assume the labels in isco_index are the target format.
        # We try to match the occupation's ISCO code to one of the labels.
        
        valid_labels = set(self.isco_index)
        
        # Create map: occupation_uri -> isco_label
        occ_isco_map = {}
        
        # Determine the length of labels to know how to truncate
        # Assuming all labels have same length (e.g. all level 3)
        if self.isco_level is not None:
            label_len = self.isco_level
        elif not self.isco_index:
            logger.warning("ISCO index is empty!")
            label_len = 4
        else:
            label_len = len(self.isco_index[0])
        
        for _, row in occs_df.iterrows():
            uri = row['conceptUri']
            raw_code = clean_isco_code(row['iscoGroup'])
            
            if not raw_code:
                continue
                
            # Truncate to label length
            candidate = raw_code[:label_len]
            
            if candidate in valid_labels:
                occ_isco_map[uri] = candidate
        
        logger.info(f"Mapped {len(occ_isco_map)} occupations to known ISCO groups")

        # --- Load Relations ---
        logger.info(f"Loading relations from {relations_path}")
        rels_df = pd.read_csv(relations_path, usecols=['occupationUri', 'skillUri', 'relationType'])
        
        # 1a. Occupation-to-Skills (All relations: essential + optional)
        # Using groupby is efficient
        self.occ_to_skills = rels_df.groupby('occupationUri')['skillUri'].apply(set).to_dict()
        
        # 1b. Skill-to-ISCO (Only essential relations?)
        # "A skill belongs to an ISCO group if any occupation in that group lists it as a required skill."
        # Assuming required == essential.
        
        if self.essentials_only:
            essential_rels = rels_df[rels_df['relationType'] == 'essential']
            logger.info(f"Only essential skills: {essential_rels.shape}")
        else:
            essential_rels = rels_df.copy()
            logger.info(f"All related skills: {essential_rels.shape}")

        # Merge with occupation ISCO info
        # We want: Skill -> Set of ISCO groups
        
        # Add isco group to relations
        # Map occupationUri to isco_label
        essential_rels = essential_rels.copy()
        essential_rels['iscoGroup'] = essential_rels['occupationUri'].map(occ_isco_map)
        
        # Drop rows where occupation didn't have a valid ISCO mapping
        essential_rels = essential_rels.dropna(subset=['iscoGroup'])
        
        self.skill_to_isco = essential_rels.groupby('skillUri')['iscoGroup'].apply(set).to_dict()
        
        # Also compute counts: how many occupations use each skill within each ISCO group
        # This is needed for frequency-based affinity (Option B)
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
        """
        Step 2: Build Skill-ISCO Affinity Matrix.
        Rows: Skills (sorted by URI)
        Columns: ISCO groups (ordered by self.isco_index)
        
        Args:
            mode: Affinity computation mode
                - 'uniform': P(g|s) = 1/N if skill is essential for N groups (equal weight)
                - 'frequency': P(g|s) weighted by occupation count within each group
                - 'binary': No normalization, just 1 if skill is in group, 0 otherwise
        """
        logger.info(f"Building Skill-ISCO Affinity Matrix (mode={mode})...")
        self.affinity_mode = mode
        
        if not self.skill_to_isco:
            logger.warning("Skill-to-ISCO mapping is empty. Run build_lookup_tables() first.")
            return

        # 1. Create stable skill index
        # We include all skills that have an ISCO mapping. 
        # Skills without mapping (row of zeros) are effectively ignored in this scoring component.
        self.skill_index = sorted(self.skill_to_isco.keys())
        self.skill_uri_to_idx = {uri: i for i, uri in enumerate(self.skill_index)}
        
        n_skills = len(self.skill_index)
        n_groups = len(self.isco_index)
        
        # 2. Initialize matrix
        self.affinity_matrix = np.zeros((n_skills, n_groups), dtype=np.float32)
        
        # Map ISCO label to column index for fast lookup
        isco_label_to_idx = {label: i for i, label in enumerate(self.isco_index)}
        
        # 3. Fill matrix based on mode
        for i, skill_uri in enumerate(self.skill_index):
            groups = self.skill_to_isco.get(skill_uri, set())
            
            valid_groups = [(g, isco_label_to_idx[g]) for g in groups if g in isco_label_to_idx]
            
            if not valid_groups:
                continue
            
            if mode == 'uniform':
                # Option A: Uniform probability over all groups
                prob = 1.0 / len(valid_groups)
                for _, col_idx in valid_groups:
                    self.affinity_matrix[i, col_idx] = prob
                    
            elif mode == 'frequency':
                # Option B: Weight by occupation count within each group
                counts = self.skill_to_isco_counts.get(skill_uri, {})
                raw_weights = []
                for group, col_idx in valid_groups:
                    count = counts.get(group, 1)  # Default to 1 if missing
                    raw_weights.append((col_idx, count))
                
                # Normalize to sum to 1
                total = sum(w for _, w in raw_weights)
                if total > 0:
                    for col_idx, weight in raw_weights:
                        self.affinity_matrix[i, col_idx] = weight / total
                        
            elif mode == 'binary':
                # Option C: Binary indicators (no normalization)
                for _, col_idx in valid_groups:
                    self.affinity_matrix[i, col_idx] = 1.0
            else:
                raise ValueError(f"Unknown affinity mode: {mode}")
                 
        logger.info(f"Built affinity matrix: {self.affinity_matrix.shape} (Skills x ISCO Groups)")
        logger.info(f"Affinity matrix ready; will log progress every {self.progress_log_every} jobs during batch evaluation (override with evaluate_batch(progress_log_every=...)).")
        
    def get_skill_affinity_vector(self, skill_uri: str) -> Optional[np.ndarray]:
        """Returns the affinity vector for a given skill, or None if not found."""
        idx = self.skill_uri_to_idx.get(skill_uri)
        if idx is not None:
            return self.affinity_matrix[idx]
        return None

    # =========================================================================
    # Normalization Methods for Linear Fusion
    # =========================================================================
    
    @staticmethod
    def _normalize_minmax(values: np.ndarray) -> np.ndarray:
        """
        Min-max normalization: scales values to [0, 1] range.
        
        Formula: (x - min) / (max - min)
        If all values are identical, returns array of 0.5.
        """
        if len(values) == 0:
            return values
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val - min_val < 1e-10:  # All values identical
            return np.full_like(values, 0.5)
        
        return (values - min_val) / (max_val - min_val)
    
    @staticmethod
    def _normalize_zscore(values: np.ndarray) -> np.ndarray:
        """
        Z-score normalization: standardizes to mean=0, std=1, then applies sigmoid.
        
        Formula: sigmoid((x - mean) / std)
        The sigmoid maps z-scores to [0, 1] for compatibility with other normalized scores.
        If std is near zero, returns array of 0.5.
        """
        if len(values) == 0:
            return values
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std < 1e-10:  # All values identical
            return np.full_like(values, 0.5)
        
        z_scores = (values - mean) / std
        # Apply sigmoid to map to [0, 1]
        return 1.0 / (1.0 + np.exp(-z_scores))
    
    @staticmethod
    def _normalize_rank(values: np.ndarray) -> np.ndarray:
        """
        Rank-based normalization: converts values to percentiles.
        
        Formula: (rank - 1) / (n - 1) where rank is 1-based ascending order.
        Highest value gets 1.0, lowest gets 0.0.
        Ties are handled by averaging ranks.
        """
        if len(values) <= 1:
            return np.full_like(values, 0.5) if len(values) == 1 else values
        
        # rankdata returns 1-based ranks, with 'average' for ties
        ranks = rankdata(values, method='average')
        n = len(values)
        
        # Convert to percentiles [0, 1]
        return (ranks - 1) / (n - 1)
    
    def normalize_scores(
        self, 
        values: np.ndarray, 
        method: NormalizationMethod = 'minmax'
    ) -> np.ndarray:
        """
        Apply normalization to a set of scores.
        
        Args:
            values: Array of scores to normalize.
            method: Normalization method ('minmax', 'zscore', 'rank').
            
        Returns:
            Normalized scores in [0, 1] range.
        """
        if method == 'minmax':
            return self._normalize_minmax(values)
        elif method == 'zscore':
            return self._normalize_zscore(values)
        elif method == 'rank':
            return self._normalize_rank(values)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    # =========================================================================
    # Score Fusion Methods
    # =========================================================================
    
    def _fuse_multiplicative(
        self,
        base_scores: np.ndarray,
        isco_weights: np.ndarray,
        alpha: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 0.0
    ) -> np.ndarray:
        """
        Multiplicative (Bayesian-style) fusion.
        
        Formula: Score(s|q) = BaseScore(s|q)^alpha × Weight_ISCO(s|q)^gamma
        
        Args:
            base_scores: Task B similarity scores.
            isco_weights: ISCO alignment weights.
            alpha: Exponent for base scores (default 1.0).
            gamma: Exponent for ISCO weights (0 means no ISCO effect).
            epsilon: Floor value for ISCO weights to prevent near-zero elimination.
            
        Returns:
            Fused scores.
        """
        # Apply epsilon floor to ISCO weights
        if epsilon > 0:
            isco_weights = np.maximum(isco_weights, epsilon)
        
        # Apply exponents and multiply
        return np.power(base_scores, alpha) * np.power(isco_weights, gamma)
    
    def _fuse_linear(
        self,
        base_scores: np.ndarray,
        isco_weights: np.ndarray,
        alpha: float = 0.5,
        normalization: NormalizationMethod = 'minmax'
    ) -> np.ndarray:
        """
        Linear (additive) fusion with normalization.
        
        Formula: Score(s|q) = alpha × BaseScore_tilde(s|q) + (1-alpha) × Weight_ISCO_tilde(s|q)
        
        Where tilde denotes normalized scores.
        
        Args:
            base_scores: Task B similarity scores (will be normalized).
            isco_weights: ISCO alignment weights (will be normalized).
            alpha: Interpolation weight [0, 1]. Higher values favor base scores.
            normalization: Normalization method ('minmax', 'zscore', 'rank').
            
        Returns:
            Fused scores (normalized and interpolated).
        """
        # Normalize both score arrays
        base_normalized = self.normalize_scores(base_scores, method=normalization)
        isco_normalized = self.normalize_scores(isco_weights, method=normalization)
        
        # Linear interpolation
        return alpha * base_normalized + (1 - alpha) * isco_normalized

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
        """
        Full Pipeline: Score and rank skills for a job query.
        
        Combines three signals:
        1. Task A filtering: Restricts candidates to skills from top-k predicted occupations
        2. Task B base scores: Semantic similarity from bi-encoder
        3. ISCO weights: Alignment scores from ISCO classification
        
        Args:
            task_a_occupations: List of top-k ESCO occupation URIs from Task A.
            task_b_scores: Dict of {skill_uri: score} from Task B (top-1000).
            isco_probs: Probability distribution over ISCO groups (numpy array).
            fusion_strategy: 'multiplicative' or 'linear'.
            alpha: For multiplicative: base score exponent (default 1.0).
                   For linear: interpolation weight [0,1], higher favors base scores.
            gamma: ISCO weight exponent (multiplicative only, 0 = no ISCO effect).
            epsilon: Floor for ISCO weights to prevent near-zero elimination.
            normalization: Normalization method for linear fusion ('minmax', 'zscore', 'rank').
            collect_stats: If True, collect and return debugging statistics.
            
        Returns:
            Tuple of (ranked_skills, stats) where:
            - ranked_skills: List of (skill_uri, final_score) sorted descending.
            - stats: JobScoringStats if collect_stats=True, else None.
        """
        # Initialize stats collector
        stats = JobScoringStats() if collect_stats else None
        
        # Step 1: Generate candidate skills from Task A occupations
        candidate_skills = set()
        num_valid_occs = 0
        for occ_uri in task_a_occupations:
            if occ_uri in self.occ_to_skills:
                candidate_skills.update(self.occ_to_skills[occ_uri])
                num_valid_occs += 1
        
        if stats:
            stats.num_task_a_occupations = len(task_a_occupations)
            stats.num_valid_task_a_occupations = num_valid_occs
                
        # Edge Case: Empty candidate set -> fall back to raw Task B scores
        fallback_mode = False
        if not candidate_skills:
            fallback_mode = True
            candidate_skills = set(task_b_scores.keys())
            
        if stats:
            stats.fallback_mode = fallback_mode
            stats.num_task_a_candidates = len(candidate_skills)

        # Step 2: Get base scores for candidates (filter Task B to candidates)
        # For candidates not in Task B, impute with min score
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

        # Step 3: Compute ISCO weights for each candidate
        isco_valid = (
            isco_probs is not None and 
            len(isco_probs) > 0 and 
            np.sum(isco_probs) > 0
        )
        
        if stats:
            stats.isco_probs_valid = isco_valid
        
        num_with_isco = 0
        num_without_isco = 0
        
        if not isco_valid:
            # No valid ISCO probs -> neutral weights (1.0)
            logger.debug("Empty or invalid ISCO probs, using neutral weights")
            isco_weights = np.ones(len(skill_uris), dtype=np.float32)
            num_without_isco = len(skill_uris)
        else:
            isco_weights = np.zeros(len(skill_uris), dtype=np.float32)
            for i, skill_uri in enumerate(skill_uris):
                skill_affinity = self.get_skill_affinity_vector(skill_uri)
                if skill_affinity is None:
                    # Missing ISCO mapping -> neutral weight
                    isco_weights[i] = 1.0
                    num_without_isco += 1
                else:
                    # Dot product: alignment between skill affinity and query ISCO probs
                    isco_weights[i] = np.dot(skill_affinity, isco_probs)
                    num_with_isco += 1
        
        if stats:
            stats.num_with_isco_mapping = num_with_isco
            stats.num_without_isco_mapping = num_without_isco
        
        # Step 4: Apply fusion strategy
        if fusion_strategy == 'multiplicative':
            final_scores_arr = self._fuse_multiplicative(
                base_scores=base_scores,
                isco_weights=isco_weights,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon
            )
        elif fusion_strategy == 'linear':
            # For linear fusion, apply epsilon floor before normalization
            if epsilon > 0:
                isco_weights = np.maximum(isco_weights, epsilon)
            final_scores_arr = self._fuse_linear(
                base_scores=base_scores,
                isco_weights=isco_weights,
                alpha=alpha,
                normalization=normalization
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            
        # Step 5: Build and sort final ranking
        final_scores = list(zip(skill_uris, final_scores_arr.tolist()))
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return (final_scores, stats)
    
    def score_job_full_pipeline(
        self,
        job_id: str,
        task_a_preds: Dict[str, List[str]],
        task_b_preds: Dict[str, Dict[str, float]],
        isco_preds: Dict[str, np.ndarray],
        task_a_k: int = 5,
        fusion_strategy: FusionStrategy = 'multiplicative',
        alpha: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 0.0,
        normalization: NormalizationMethod = 'minmax',
        temperature: float = 1.0,
        collect_stats: bool = False
    ) -> Tuple[List[Tuple[str, float]], Optional[JobScoringStats]]:
        """
        Full pipeline integration for a single job query.
        
        Given a job ID and prediction dictionaries, this method:
        1. Retrieves candidate skills from Task A top-k occupations
        2. Filters Task B scores to candidate skills
        3. Computes ISCO weights for each candidate
        4. Applies the selected fusion strategy
        5. Returns ranked skills with final scores
        
        Args:
            job_id: ID of the job to score.
            task_a_preds: Dict mapping job_id -> list of occupation URIs.
            task_b_preds: Dict mapping job_id -> {skill_uri: score}.
            isco_preds: Dict mapping job_id -> ISCO probability vector.
            task_a_k: Number of top occupations to use for candidate generation.
            fusion_strategy: 'multiplicative' or 'linear'.
            alpha: Base score influence (exponent for mult, weight for linear).
            gamma: ISCO weight exponent (multiplicative only).
            epsilon: Floor for ISCO weights.
            normalization: Method for linear fusion ('minmax', 'zscore', 'rank').
            temperature: Temperature for ISCO probability sharpening.
            collect_stats: If True, collect and return debugging statistics.
            
        Returns:
            Tuple of (ranked_skills, stats) where:
            - ranked_skills: List of (skill_uri, final_score) sorted descending.
            - stats: JobScoringStats if collect_stats=True, else None.
        """
        # Get predictions for this job
        task_a_occs = task_a_preds.get(job_id, [])[:task_a_k]
        task_b_scores = task_b_preds.get(job_id, {})
        isco_probs = isco_preds.get(job_id, np.array([]))
        
        # Apply temperature scaling to ISCO probs if needed
        if temperature != 1.0 and temperature > 0 and len(isco_probs) > 0:
            isco_probs = np.power(isco_probs, 1.0 / temperature)
            prob_sum = np.sum(isco_probs)
            if prob_sum > 0:
                isco_probs = isco_probs / prob_sum
        
        return self.score_job(
            task_a_occupations=task_a_occs,
            task_b_scores=task_b_scores,
            isco_probs=isco_probs,
            fusion_strategy=fusion_strategy,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            normalization=normalization,
            collect_stats=collect_stats
        )
    
    def evaluate_batch(
        self,
        task_a_preds: Dict[str, List[str]],  # job_id -> list of occupation URIs
        task_b_preds: Dict[str, Dict[str, float]],  # job_id -> {skill_uri: score}
        isco_preds: Dict[str, np.ndarray],  # job_id -> isco_prob_vector
        ground_truth: Dict[str, Set[str]],  # job_id -> set of true skill URIs
        task_a_k: int = 5,
        fusion_strategy: FusionStrategy = 'multiplicative',
        alpha: float = 1.0,
        gamma: float = 1.0,
        temperature: float = 1.0,
        mode: str = 'full',  # 'full', 'task_b_only', 'task_a_filter_only'
        epsilon: float = 0.0,
        normalization: NormalizationMethod = 'minmax',
        return_predictions: bool = False,
        collect_stats: bool = False,
        log_stats_every: int = 0,
        progress_log_every: Optional[int] = None
    ) -> Dict[str, object]:
        """
        Run scoring on a batch of jobs and compute metrics.
        
        Args:
            task_a_preds: Job ID -> list of predicted occupation URIs.
            task_b_preds: Job ID -> {skill_uri: similarity score}.
            isco_preds: Job ID -> ISCO probability vector.
            ground_truth: Job ID -> set of true skill URIs.
            task_a_k: Number of top occupations for candidate generation.
            fusion_strategy: 'multiplicative' or 'linear'.
            alpha: For multiplicative: base score exponent.
                   For linear: interpolation weight [0,1].
            gamma: ISCO weight exponent (multiplicative only).
            temperature: Temperature for ISCO probability sharpening.
            mode: 'full' (all signals), 'task_b_only', or 'task_a_filter_only'.
            epsilon: Floor for ISCO weights (prevents near-zero elimination).
            normalization: Method for linear fusion ('minmax', 'zscore', 'rank').
            return_predictions: If True, include ranked predictions in output.
            collect_stats: If True, collect per-job statistics for debugging.
            log_stats_every: Log individual job stats every N jobs (0 = don't log individual jobs).
        
        Returns:
            Dict of metrics (mAP, R@10, R@50, R@100).
            If return_predictions=True, includes 'predictions': {job_id: [(uri, score), ...]}
            If collect_stats=True, includes 'stats_summary': aggregated statistics dict.
        """
        all_aps = []
        all_recalls_at_10 = []
        all_recalls_at_50 = []
        all_recalls_at_100 = []
        
        predictions = {}  # job_id -> list of (skill_uri, score)
        agg_stats = AggregatedStats() if collect_stats else None
        
        # Intersection of all keys
        job_ids = set(task_b_preds.keys()) & set(ground_truth.keys())
        
        # If mode needs Task A or ISCO, intersect with those too
        if mode != 'task_b_only':
            if mode == 'full' or mode == 'task_a_filter_only':
                job_ids = job_ids & set(task_a_preds.keys())
            if mode == 'full':
                job_ids = job_ids & set(isco_preds.keys())
        
        job_ids = sorted(job_ids)  # Sort for deterministic iteration
        logger.info(f"Evaluating ({mode}, strategy={fusion_strategy}) on {len(job_ids)} common jobs...")
        # Determine progress logging interval
        prog_every = progress_log_every if progress_log_every is not None else self.progress_log_every
        
        for i, job_id in enumerate(job_ids):
            skill_scores = task_b_preds[job_id]
            true_skills = ground_truth[job_id]
            predicted_uris = []
            final_scores_list = []
            job_stats = None

            # --- Logic for different baselines ---
            if mode == 'task_b_only':
                # Baseline 1: Task B only (no filtering, no weighting)
                sorted_items = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)
                final_scores_list = sorted_items
                predicted_uris = [x[0] for x in sorted_items]
                
            elif mode == 'task_a_filter_only':
                # Baseline 2: Task A filtering only (no ISCO weighting)
                occ_list = task_a_preds[job_id][:task_a_k]
                ranked_results, job_stats = self.score_job(
                    task_a_occupations=occ_list,
                    task_b_scores=skill_scores,
                    isco_probs=np.zeros(len(self.isco_index)),
                    fusion_strategy='multiplicative',
                    alpha=1.0,
                    gamma=0.0,  # No ISCO effect
                    epsilon=0.0,
                    collect_stats=collect_stats
                )
                final_scores_list = ranked_results
                predicted_uris = [r[0] for r in ranked_results]
                
            elif mode == 'full':
                # Full Pipeline with selected fusion strategy
                occ_list = task_a_preds[job_id][:task_a_k]
                probs = isco_preds[job_id].copy()
                
                # Apply temperature scaling if needed
                if temperature != 1.0 and temperature > 0:
                    probs = np.power(probs, 1.0 / temperature)
                    prob_sum = np.sum(probs)
                    if prob_sum > 0:
                        probs = probs / prob_sum
                
                ranked_results, job_stats = self.score_job(
                    task_a_occupations=occ_list,
                    task_b_scores=skill_scores,
                    isco_probs=probs,
                    fusion_strategy=fusion_strategy,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    normalization=normalization,
                    collect_stats=collect_stats
                )
                final_scores_list = ranked_results
                predicted_uris = [r[0] for r in ranked_results]

            # Aggregate stats
            if collect_stats and job_stats is not None:
                agg_stats.add(job_stats)
                # Log individual job stats at DEBUG level or periodically
                if log_stats_every > 0 and (i + 1) % log_stats_every == 0:
                    logger.info(f"[Job {job_id}] {job_stats}")
            # Periodic progress logging (counts only)
            if prog_every and (i + 1) % prog_every == 0:
                pct = (i + 1) / len(job_ids) * 100 if len(job_ids) > 0 else 0.0
                logger.info(f"Processed {i+1}/{len(job_ids)} jobs ({pct:.1f}%)")
            
            if return_predictions:
                predictions[job_id] = final_scores_list

            # --- Metrics ---
            def calc_recall(k):
                top_k = set(predicted_uris[:k])
                if not true_skills:
                    return 0.0
                hits = len(top_k & true_skills)
                return hits / len(true_skills)
            
            all_recalls_at_10.append(calc_recall(10))
            all_recalls_at_50.append(calc_recall(50))
            all_recalls_at_100.append(calc_recall(100))
            
            # Average Precision (AP)
            hits = 0
            sum_precs = 0
            for rank, uri in enumerate(predicted_uris):
                if uri in true_skills:
                    hits += 1
                    sum_precs += hits / (rank + 1)
            
            ap = sum_precs / len(true_skills) if true_skills else 0.0
            all_aps.append(ap)
            
        results = {
            "mAP": np.mean(all_aps) if all_aps else 0.0,
            "R@10": np.mean(all_recalls_at_10) if all_recalls_at_10 else 0.0,
            "R@50": np.mean(all_recalls_at_50) if all_recalls_at_50 else 0.0,
            "R@100": np.mean(all_recalls_at_100) if all_recalls_at_100 else 0.0
        }
        
        if return_predictions:
            results['predictions'] = predictions
        
        if collect_stats and agg_stats is not None:
            agg_stats.log_summary()
            results['stats_summary'] = agg_stats.summary()
            
        return results

    def evaluate_batch_vectorized(
        self,
        task_a_preds: Dict[str, List[str]],
        task_b_preds: Dict[str, Dict[str, float]],
        isco_preds: Dict[str, np.ndarray],
        ground_truth: Dict[str, Set[str]],
        task_a_k: int = 5,
        fusion_strategy: FusionStrategy = 'multiplicative',
        alpha: float = 1.0,
        gamma: float = 1.0,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        normalization: NormalizationMethod = 'minmax',
        top_k: int = 100,
    ) -> Dict[str, object]:
        """
        Vectorized batch evaluation - significantly faster than per-job processing.
        
        Processes all jobs using matrix operations instead of Python loops.
        This provides 50-100x speedup compared to evaluate_batch() for large datasets.
        
        Args:
            task_a_preds: Job ID -> list of predicted occupation URIs.
            task_b_preds: Job ID -> {skill_uri: similarity score}.
            isco_preds: Job ID -> ISCO probability vector.
            ground_truth: Job ID -> set of true skill URIs.
            task_a_k: Number of top occupations for candidate generation.
            fusion_strategy: 'multiplicative' or 'linear'.
            alpha: For multiplicative: base score exponent.
                   For linear: interpolation weight [0,1].
            gamma: ISCO weight exponent (multiplicative only).
            temperature: Temperature for ISCO probability sharpening.
            epsilon: Floor for ISCO weights (prevents near-zero elimination).
            normalization: Method for linear fusion ('minmax', 'zscore', 'rank').
            top_k: Maximum number of skills to rank per job.
        
        Returns:
            Dict of metrics (mAP, R@10, R@50, R@100).
        """
        # 1. Get common job IDs
        job_ids = sorted(
            set(task_b_preds.keys()) & set(ground_truth.keys()) & 
            set(task_a_preds.keys()) & set(isco_preds.keys())
        )
        n_jobs = len(job_ids)
        logger.info(f"Vectorized evaluation on {n_jobs} jobs...")
        
        if n_jobs == 0:
            return {"mAP": 0.0, "R@10": 0.0, "R@50": 0.0, "R@100": 0.0}
        
        # 2. Build unified skill index from all Task B scores
        all_skills = set()
        for job_id in job_ids:
            all_skills.update(task_b_preds[job_id].keys())
        skill_list = sorted(all_skills)
        skill_to_idx = {s: i for i, s in enumerate(skill_list)}
        n_skills = len(skill_list)
        
        logger.info(f"  Building matrices: {n_jobs} jobs × {n_skills} skills")
        
        # 3. Build candidate mask [n_jobs, n_skills] - which skills are candidates per job
        candidate_mask = np.zeros((n_jobs, n_skills), dtype=bool)
        for i, job_id in enumerate(job_ids):
            task_a_occs = task_a_preds[job_id][:task_a_k]
            candidate_skills = set()
            for occ_uri in task_a_occs:
                if occ_uri in self.occ_to_skills:
                    candidate_skills.update(self.occ_to_skills[occ_uri])
            
            # If no candidates from Task A, fall back to all Task B skills
            if not candidate_skills:
                candidate_skills = set(task_b_preds[job_id].keys())
            
            for skill in candidate_skills:
                if skill in skill_to_idx:
                    candidate_mask[i, skill_to_idx[skill]] = True
        
        # 4. Build dense score matrix [n_jobs, n_skills]
        score_matrix = np.full((n_jobs, n_skills), -np.inf, dtype=np.float32)  # -inf for non-candidates
        for i, job_id in enumerate(job_ids):
            min_score = min(task_b_preds[job_id].values()) if task_b_preds[job_id] else 0.0
            for skill, score in task_b_preds[job_id].items():
                if skill in skill_to_idx:
                    score_matrix[i, skill_to_idx[skill]] = score
            # Impute missing candidates with min score
            for j in range(n_skills):
                if candidate_mask[i, j] and score_matrix[i, j] == -np.inf:
                    score_matrix[i, j] = min_score
        
        # 5. Build ISCO probability matrix [n_jobs, n_isco_groups]
        n_groups = len(self.isco_index)
        isco_matrix = np.zeros((n_jobs, n_groups), dtype=np.float32)
        for i, job_id in enumerate(job_ids):
            probs = isco_preds[job_id].copy()
            # Apply temperature scaling
            if temperature != 1.0 and temperature > 0 and len(probs) > 0:
                probs = np.power(probs, 1.0 / temperature)
                prob_sum = np.sum(probs)
                if prob_sum > 0:
                    probs = probs / prob_sum
            isco_matrix[i] = probs
        
        # 6. Build skill affinity matrix [n_skills, n_isco_groups]
        skill_affinity = np.ones((n_skills, n_groups), dtype=np.float32)  # Default neutral (1.0)
        skill_has_affinity = np.zeros(n_skills, dtype=bool)
        for skill, idx in skill_to_idx.items():
            aff = self.get_skill_affinity_vector(skill)
            if aff is not None:
                skill_affinity[idx] = aff
                skill_has_affinity[idx] = True
        
        # 7. Compute ISCO weights via matrix multiplication [n_jobs, n_skills]
        # isco_weights[i, j] = dot(isco_matrix[i], skill_affinity[j])
        isco_weights = isco_matrix @ skill_affinity.T  # [n_jobs, n_skills]
        
        # Skills without affinity mapping get neutral weight (1.0)
        isco_weights[:, ~skill_has_affinity] = 1.0
        
        # Apply epsilon floor
        if epsilon > 0:
            isco_weights = np.maximum(isco_weights, epsilon)
        
        # 8. Apply fusion strategy
        # First, extract only valid scores (not -inf) for normalization
        valid_mask = score_matrix > -np.inf
        
        if fusion_strategy == 'multiplicative':
            # fused = base^alpha × isco^gamma
            fused_scores = np.where(
                valid_mask,
                np.power(np.maximum(score_matrix, 1e-10), alpha) * np.power(isco_weights, gamma),
                -np.inf
            )
        else:  # linear
            # Need to normalize per-job for linear fusion
            fused_scores = np.full_like(score_matrix, -np.inf)
            for i in range(n_jobs):
                valid_idx = valid_mask[i]
                if not np.any(valid_idx):
                    continue
                base_valid = score_matrix[i, valid_idx]
                isco_valid = isco_weights[i, valid_idx]
                
                # Normalize
                base_norm = self.normalize_scores(base_valid, method=normalization)
                isco_norm = self.normalize_scores(isco_valid, method=normalization)
                
                fused_scores[i, valid_idx] = alpha * base_norm + (1 - alpha) * isco_norm
        
        # 9. Apply candidate mask (non-candidates stay at -inf)
        fused_scores = np.where(candidate_mask, fused_scores, -np.inf)
        
        # 10. Get top-k indices per job using argpartition (fast)
        # Handle case where some jobs have fewer than top_k candidates
        top_k_indices = np.zeros((n_jobs, top_k), dtype=np.int64)
        top_k_scores = np.full((n_jobs, top_k), -np.inf, dtype=np.float32)
        
        for i in range(n_jobs):
            valid_count = int(np.sum(fused_scores[i] > -np.inf))
            k = min(top_k, valid_count)
            if k > 0:
                if k < n_skills:
                    part_idx = np.argpartition(-fused_scores[i], k)[:k]
                    sorted_idx = part_idx[np.argsort(-fused_scores[i, part_idx])]
                else:
                    sorted_idx = np.argsort(-fused_scores[i])[:k]
                top_k_indices[i, :k] = sorted_idx
                top_k_scores[i, :k] = fused_scores[i, sorted_idx]
        
        # 11. Build gold sets for evaluation
        gold_sets = []
        for job_id in job_ids:
            gold_uris = ground_truth[job_id]
            gold_indices = {skill_to_idx[uri] for uri in gold_uris if uri in skill_to_idx}
            gold_sets.append(gold_indices)
        
        # 12. Compute metrics
        all_aps = []
        all_recalls_10 = []
        all_recalls_50 = []
        all_recalls_100 = []
        
        for i in range(n_jobs):
            gold = gold_sets[i]
            if not gold:
                all_aps.append(0.0)
                all_recalls_10.append(0.0)
                all_recalls_50.append(0.0)
                all_recalls_100.append(0.0)
                continue
            
            # Get predicted indices (filter out -inf scores)
            pred_indices = [idx for idx, score in zip(top_k_indices[i], top_k_scores[i]) if score > -np.inf]
            
            # Recall@K
            def calc_recall(k):
                top_k_set = set(pred_indices[:k])
                return len(top_k_set & gold) / len(gold)
            
            all_recalls_10.append(calc_recall(10))
            all_recalls_50.append(calc_recall(50))
            all_recalls_100.append(calc_recall(100))
            
            # Average Precision
            hits = 0
            sum_precs = 0.0
            for rank, idx in enumerate(pred_indices):
                if idx in gold:
                    hits += 1
                    sum_precs += hits / (rank + 1)
            all_aps.append(sum_precs / len(gold))
        
        return {
            "mAP": np.mean(all_aps) if all_aps else 0.0,
            "R@10": np.mean(all_recalls_10) if all_recalls_10 else 0.0,
            "R@50": np.mean(all_recalls_50) if all_recalls_50 else 0.0,
            "R@100": np.mean(all_recalls_100) if all_recalls_100 else 0.0,
        }

    def grid_search(
        self,
        task_a_preds, task_b_preds, isco_preds, ground_truth,
        k_values: List[int] = [3, 5, 7, 10],
        fusion_strategies: List[FusionStrategy] = ['multiplicative'],
        alpha_values: List[float] = [1.0],
        gamma_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        temperatures: List[float] = [1.0],
        affinity_modes: List[str] = ['uniform'],
        epsilon_values: List[float] = [0.0],
        normalization_methods: List[NormalizationMethod] = ['minmax'],
        metric: str = 'mAP',
        n_jobs: int = 1
    ):
        """
        Run hyperparameter optimization over fusion strategies and parameters.
        
        Args:
            k_values: Task A top-k values to try.
            fusion_strategies: List of strategies ('multiplicative', 'linear').
            alpha_values: Alpha values to try.
                If both 'multiplicative' and 'linear' are in fusion_strategies, this list
                will be used for both unless overridden by defaults inside.
                It is recommended to pass strategy-specific alpha lists if calling this manually.
                This function attempts to use appropriate defaults for each strategy if not constrained by user.
            gamma_values: Gamma values for multiplicative fusion (ISCO exponent).
            temperatures: Temperature values for ISCO probability sharpening.
            affinity_modes: Affinity matrix modes ('uniform', 'frequency', 'binary').
            epsilon_values: Epsilon floor values to try.
            normalization_methods: Normalization methods for linear fusion.
            metric: Metric to optimize ('mAP', 'R@10', 'R@50', 'R@100').
            n_jobs: Number of parallel jobs (-1 for all CPUs).
            
        Returns:
            Tuple of (all_results, best_params).
        """
        
        results = []
        best_score = -1.0
        best_params = None
        current_affinity_mode = self.affinity_mode
        
        # Generate parameter combinations based on fusion strategy
        multiplicative_params = []
        linear_params = []
        
        # Define strategy-specific alpha defaults if the passed list matches the "union" default
        # or if we want to enforce logic.
        # Heuristic: If alpha_values looks like our "default union" [0.3, 0.5, 0.7, 0.9, 1.0, 1.5],
        # we split it.
        # If it looks like a user provided list (e.g. single value or custom range), we use it as is for both.
        
        default_union = {0.3, 0.5, 0.7, 0.9, 1.0, 1.5}
        is_default_range = set(alpha_values) == default_union
        
        if 'multiplicative' in fusion_strategies:
            if is_default_range:
                mult_alphas = [0.5, 1.0, 1.5]
            else:
                mult_alphas = alpha_values
                
            multiplicative_params = list(itertools.product(
                k_values, ['multiplicative'], mult_alphas, gamma_values, 
                temperatures, affinity_modes, epsilon_values, ['minmax']  # norm not used
            ))
            
        if 'linear' in fusion_strategies:
            if is_default_range:
                lin_alphas = [0.3, 0.5, 0.7, 0.9]
            else:
                lin_alphas = alpha_values
                
            linear_params = list(itertools.product(
                k_values, ['linear'], lin_alphas, [0.0],  # gamma not used for linear
                temperatures, affinity_modes, epsilon_values, normalization_methods
            ))
        
        param_grid = multiplicative_params + linear_params
        logger.info(f"Starting grid search with {len(param_grid)} combinations (n_jobs={n_jobs})...")
        
        # Helper for parallel execution
        def evaluate_one_config(params_tuple):
            k, strategy, alpha, gamma, temp, aff_mode, eps, norm = params_tuple
            # Placeholder helper (actual evaluation uses grouped processing below)
            return (params_tuple, None)
        
        # Group by affinity mode to avoid race conditions on self.affinity_matrix.
        
        # Extract unique affinity modes
        unique_aff_modes = sorted(list(set(p[5] for p in param_grid)))
        
        for aff_mode in unique_aff_modes:
             # Set affinity matrix for this mode
            if aff_mode != self.affinity_mode:
                logger.info(f"Switching affinity mode to '{aff_mode}'...")
                self.build_affinity_matrix(mode=aff_mode)
            
            # Filter params for this mode
            sub_grid = [p for p in param_grid if p[5] == aff_mode]
            
            def process_params(p):
                k, strategy, alpha, gamma, temp, _, eps, norm = p
                
                # Use vectorized evaluation for speed
                metrics = self.evaluate_batch_vectorized(
                    task_a_preds, task_b_preds, isco_preds, ground_truth,
                    task_a_k=k,
                    fusion_strategy=strategy,
                    alpha=alpha,
                    gamma=gamma,
                    temperature=temp,
                    epsilon=eps,
                    normalization=norm,
                )
                
                params_dict = {
                    'k': k,
                    'fusion_strategy': strategy,
                    'alpha': alpha,
                    'gamma': gamma,
                    'temp': temp,
                    'affinity_mode': aff_mode,
                    'epsilon': eps,
                    'normalization': norm
                }
                metrics['params'] = params_dict
                return metrics

            # Run parallel (or sequential) evaluation for this affinity mode
            if n_jobs != 1:
                batch_results = Parallel(n_jobs=n_jobs)(
                    delayed(process_params)(p) for p in tqdm(sub_grid)
                )
            else:
                batch_results = [process_params(p) for p in tqdm(sub_grid)]
                
            # Process results
            for metrics in batch_results:
                params = metrics['params']
                logger.info(f"Evaluated: {params['fusion_strategy']}, K={params['k']}, alpha={params['alpha']}, gamma={params['gamma']} -> mAP: {metrics['mAP']:.4f}")
                
                results.append(metrics)
                score = metrics.get(metric, metrics['mAP'])
                if score > best_score:
                    best_score = score
                    best_params = params
                    
        logger.info("Grid search complete.")
        logger.info(f"Best {metric}: {best_score:.4f} with params {best_params}")
        
        return results, best_params

def load_data_and_run(
    scorer: FusedScorer,
    task_a_path: Path,
    task_b_path: Path,
    isco_path: Path,
    ground_truth_paths: Dict[str, Path],
    output_dir: Path,
    args=None
):
    """
    Load all datasets and run grid search.
    Supports train/val/test splits if 'split' column is present in decorte_map CSV.
    - Grid Search runs on VALIDATION split (if available)
    - Final Evaluation runs on TEST split (if available)
    - Final Predictions generated for ALL jobs (Train+Val+Test)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    logger.info("="*60)
    logger.info("LOADING DATA")
    logger.info("="*60)
    
    # 1. Build Ground Truth & Load Splits FIRST - we need job_id ordering for Task A
    logger.info("Building Ground Truth & Loading Splits...")
    # Load mapping: job_id -> esco_occupation_uri
    decorte_df = pd.read_csv(ground_truth_paths['decorte_map'])
    
    # Ensure job_id is string
    decorte_df['job_id'] = decorte_df['job_id'].astype(str)
    
    # Create a list of job_ids in order (for aligning Task A predictions)
    ordered_job_ids = decorte_df['job_id'].tolist()
    logger.info(f"Ground Truth has {len(ordered_job_ids)} rows, {decorte_df['job_id'].nunique()} unique job_ids")
    logger.info(f"Sample Ground Truth job IDs: {ordered_job_ids[:5]}")
    
    job_to_occ = dict(zip(decorte_df['job_id'], decorte_df['esco_id']))
    
    # Check for split column
    splits = {}
    if 'split' in decorte_df.columns:
        splits = dict(zip(decorte_df['job_id'], decorte_df['split']))
        logger.info(f"Found splits: {decorte_df['split'].unique()}")
        logger.info(f"Split counts: {decorte_df['split'].value_counts().to_dict()}")
    else:
        logger.warning("No 'split' column found in decorte_map. Using all data for everything.")
    
    # We already have scorer.occ_to_skills from build_lookup_tables()
    # So we can build job_id -> Set[skill_uri]
    
    ground_truth = {}
    for job_id, occ_uri in job_to_occ.items():
        if occ_uri in scorer.occ_to_skills:
            ground_truth[job_id] = scorer.occ_to_skills[occ_uri]
        else:
            # Maybe occupation not in ESCO relations? 
            # Or occupation URI format mismatch?
            ground_truth[job_id] = set()
            
    logger.info(f"Built Ground Truth for {len(ground_truth)} jobs")
    
    # 2. Load Task A (JSONL) - use job_id from ground truth CSV for alignment
    logger.info(f"Loading Task A from {task_a_path}")
    task_a_preds = {}
    with open(task_a_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
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
    
    # 3. Load Task B (JSON)
    logger.info(f"Loading Task B from {task_b_path}")
    with open(task_b_path, 'r') as f:
        raw_b = json.load(f)
    
    task_b_preds = {}
    for job_id, items in raw_b.items():
        # Convert list of dicts to dict of uri->score
        # Items might be list of {skill_uri, score, rank}
        scores = {}
        if isinstance(items, list):
            for item in items:
                scores[item['skill_uri']] = float(item['score'])
        elif isinstance(items, dict):
             scores = {k: float(v) for k, v in items.items()}
        task_b_preds[str(job_id)] = scores
    logger.info(f"Loaded Task B: {len(task_b_preds)} jobs")
    
    # 4. Load ISCO Predictions (JSON)
    logger.info(f"Loading ISCO preds from {isco_path}")
    with open(isco_path, 'r') as f:
        raw_isco = json.load(f)
        
    isco_preds = {}
    # Helper to map labels to indices
    label_to_idx = {lbl: i for i, lbl in enumerate(scorer.isco_index)}
    n_classes = len(scorer.isco_index)
    
    for job_id, data in raw_isco.items():
        # Data format: {"topk": [{"label": "...", "score": ...}, ...]}
        # Build dense probability vector
        prob_vec = np.zeros(n_classes, dtype=np.float32)
        
        if "topk" in data:
            for item in data["topk"]:
                lbl = item["label"]
                score = item["score"]
                if lbl in label_to_idx:
                    prob_vec[label_to_idx[lbl]] = score
        
        # Normalize just in case (though we might trust the model outputs)
        # If sum is > 0, normalize. If 0 (empty), leave as 0.
        s = np.sum(prob_vec)
        if s > 0:
            prob_vec /= s
            
        isco_preds[str(job_id)] = prob_vec
        
    logger.info(f"Loaded ISCO preds: {len(isco_preds)} jobs")
    
    # 5. Run Grid Search
    logger.info("="*60)
    logger.info("PREPARING DATASETS")
    logger.info("="*60)
    
    # Optional: Filter job_ids to common set first to speed up
    common_ids = set(task_b_preds.keys()) & set(ground_truth.keys()) & set(task_a_preds.keys()) & set(isco_preds.keys())
    if not common_ids:
        logger.error("No intersection between datasets! Check IDs.")
        return
        
    logger.info(f"Total common jobs (intersection): {len(common_ids)}")
    
    # Determine split sets
    if splits:
        train_keys = {'train', 'training'}
        val_keys = {'val', 'validation', 'valid', 'dev'}
        test_keys = {'test', 'testing'}
        
        train_ids = {jid for jid in common_ids if splits.get(jid, '').lower() in train_keys}
        val_ids = {jid for jid in common_ids if splits.get(jid, '').lower() in val_keys}
        test_ids = {jid for jid in common_ids if splits.get(jid, '').lower() in test_keys}
        
        # Determine Search IDs based on argument
        search_split_arg = args.grid_search_split if args else 'validation'
        
        if search_split_arg == 'train':
            search_ids = train_ids
        elif search_split_arg == 'train+validation':
            search_ids = train_ids | val_ids
        else: # validation (default)
            search_ids = val_ids
            
        if not search_ids:
            logger.warning(f"No IDs found for grid search split '{search_split_arg}'! Falling back to FULL set.")
            search_ids = common_ids
        else:
            logger.info(f"Using {len(search_ids)} {search_split_arg.upper()} jobs for Grid Search")
            
        if not test_ids:
            logger.warning("No test IDs found in intersection! Will evaluate on search set.")
            eval_ids = search_ids
        else:
            eval_ids = test_ids
            logger.info(f"Using {len(eval_ids)} TEST jobs for Final Evaluation")
    else:
        search_ids = common_ids
        eval_ids = common_ids
        logger.info(f"Using {len(search_ids)} jobs for Grid Search & Evaluation (No splits)")
    
    # Helper to slice dictionaries
    def get_subset(ids):
        return (
            {k: task_a_preds[k] for k in ids},
            {k: task_b_preds[k] for k in ids},
            {k: isco_preds[k] for k in ids},
            {k: ground_truth[k] for k in ids}
        )
    
    # Prepare Grid Search Data
    gs_a, gs_b, gs_isco, gs_gt = get_subset(search_ids)
    
    # Run Grid Search
    if args:
        k_values = [args.task_a_k] if args.task_a_k is not None else [3, 5, 7, 10]
        fusion_strategies = [args.fusion_strategy] if args.fusion_strategy is not None else ['multiplicative', 'linear']
        
        # Determine alpha/gamma/temperature/affinity/epsilon/normalization lists (from args or defaults).
        # We pass a union of values to grid_search; grid_search will handle strategy-specific filtering.
        if args.alpha is not None:
            alpha_values = [args.alpha]
        else:
            # Default ranges requested by user
            alpha_values = [0.3, 0.5, 0.7, 0.9, 1.0, 1.5] 
            
        gamma_values = [args.gamma] if args.gamma is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        temperatures = [args.temperature] if args.temperature is not None else [1.0]
        affinity_modes = [args.affinity_mode] if args.affinity_mode is not None else ['uniform', 'frequency', 'binary']
        epsilon_values = [args.epsilon] if args.epsilon is not None else [0.0, 0.01, 0.05, 0.1]
        normalization_methods = [args.normalization] if args.normalization is not None else ['minmax', 'zscore', 'rank']
    else:
        k_values = [3, 5, 7, 10]
        fusion_strategies = ['multiplicative', 'linear']
        alpha_values = [0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
        gamma_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        temperatures = [1.0]
        affinity_modes = ['uniform', 'frequency', 'binary']
        epsilon_values = [0.0, 0.01, 0.05, 0.1]
        normalization_methods = ['minmax', 'zscore', 'rank']

    # Note: grid_search receives these lists and applies sensible defaults per strategy.
    results, best_params = scorer.grid_search(
        gs_a, gs_b, gs_isco, gs_gt,
        k_values=k_values,
        fusion_strategies=fusion_strategies,
        alpha_values=alpha_values,
        gamma_values=gamma_values,
        temperatures=temperatures,
        affinity_modes=affinity_modes,
        epsilon_values=epsilon_values,
        normalization_methods=normalization_methods,
        n_jobs=args.n_jobs if args else 1
    )
    
    logger.info("="*60)
    logger.info("FINAL EVALUATION (TEST SET)")
    logger.info("="*60)
    
    # 6. Evaluation Comparison
    best_k = best_params['k']
    best_strategy = best_params['fusion_strategy']
    best_alpha = best_params['alpha']
    best_gamma = best_params['gamma']
    best_temp = best_params['temp']
    best_mode = best_params.get('affinity_mode', 'uniform')
    best_eps = best_params.get('epsilon', 0.0)
    best_norm = best_params.get('normalization', 'minmax')
    
    logger.info(f"Using best params: strategy={best_strategy}, K={best_k}, alpha={best_alpha}, "
                f"gamma={best_gamma}, temp={best_temp}, mode={best_mode}, eps={best_eps}, norm={best_norm}")
    
    # Ensure affinity matrix is built with best mode for final evaluation
    if scorer.affinity_mode != best_mode:
        scorer.build_affinity_matrix(mode=best_mode)
    
    # Prepare Test Data
    test_a, test_b, test_isco, test_gt = get_subset(eval_ids)

    # Baseline 1: Task B Only (no Task A filter, no ISCO)
    res_b = scorer.evaluate_batch(
        test_a, test_b, test_isco, test_gt,
        mode='task_b_only'
    )
    logger.info(f"Baseline (Task B Only): {res_b}")
    
    # Baseline 2: Task A Filter Only (no ISCO weighting)
    res_a = scorer.evaluate_batch(
        test_a, test_b, test_isco, test_gt,
        task_a_k=best_k,
        mode='task_a_filter_only'
    )
    logger.info(f"Baseline (Task A Filter Only): {res_a}")
    
    # Full Pipeline (Best Params) with statistics collection
    res_full = scorer.evaluate_batch(
        test_a, test_b, test_isco, test_gt,
        task_a_k=best_k,
        fusion_strategy=best_strategy,
        alpha=best_alpha,
        gamma=best_gamma,
        temperature=best_temp,
        epsilon=best_eps,
        normalization=best_norm,
        mode='full',
        return_predictions=False, # We'll do predictions on ALL data later
        collect_stats=True
    )
    
    logger.info(f"Full Pipeline ({best_strategy} fusion): {res_full}")
    
    # 7. Generate Predictions for ALL Data (Train+Val+Test)
    logger.info("="*60)
    logger.info("GENERATING FINAL PREDICTIONS (ALL DATA)")
    logger.info("="*60)
    
    all_ids = sorted(list(common_ids))
    logger.info(f"Generating scores for {len(all_ids)} total jobs...")
    
    all_a, all_b, all_isco, all_gt = get_subset(all_ids)
    
    # Run evaluation again on ALL data to get predictions and overall metrics
    res_all = scorer.evaluate_batch(
        all_a, all_b, all_isco, all_gt,
        task_a_k=best_k,
        fusion_strategy=best_strategy,
        alpha=best_alpha,
        gamma=best_gamma,
        temperature=best_temp,
        epsilon=best_eps,
        normalization=best_norm,
        mode='full',
        return_predictions=True,
        collect_stats=False
    )
    
    predictions = res_all.pop('predictions')
    
    # Save results
    
    # Helper to convert numpy types for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    final_results = {
        "grid_search": results,
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
    
    # Save predictions
    preds_path = output_dir / "fused_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved ranked predictions to {preds_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fused Scorer: Combines Task A filtering, Task B scores, and ISCO weights for skill ranking."
    )
    
    # ESCO/ISCO configuration
    parser.add_argument("--esco_dir", type=str, default="data/esco_datasets",
                        help="Path to ESCO dataset directory")
    parser.add_argument("--label_encoder", type=str, required=True, 
                        help="Path to label_encoder.json for ISCO labels")
    parser.add_argument("--isco_level", type=int, default=None, 
                        help="ISCO level (1-4 digits) for mapping ESCO occupations")
    parser.add_argument("--use_essentials", action='store_true',
                   help="Use only essential ISCO skills")
    
    
    # Data paths
    parser.add_argument("--task_a", type=str, help="Path to Task A predictions (jsonl)")
    parser.add_argument("--task_b", type=str, help="Path to Task B predictions (json)")
    parser.add_argument("--isco_preds", type=str, help="Path to ISCO classifier predictions (json)")
    parser.add_argument("--decorte_map", type=str, help="Path to CSV mapping job_id to esco_id")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")
    
    # Fusion hyperparameters
    parser.add_argument("--fusion_strategy", type=str, default=None,
                        choices=['multiplicative', 'linear'],
                        help="Fusion strategy: 'multiplicative' (Bayesian-style) or 'linear' (additive)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="For multiplicative: base score exponent (default 1.0). "
                             "For linear: interpolation weight [0,1], higher favors base scores.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="ISCO weight exponent (multiplicative only). 0 = no ISCO effect.")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Floor value for ISCO weights (0.0-1.0). Prevents near-zero elimination.")
    parser.add_argument("--normalization", type=str, default=None,
                        choices=['minmax', 'zscore', 'rank'],
                        help="Normalization method for linear fusion")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for ISCO probability sharpening. <1 sharpens, >1 smooths.")
    
    # Task A configuration
    parser.add_argument("--task_a_k", type=int, default=None,
                        help="Number of top occupations for candidate skill generation")
    
    # Affinity matrix configuration
    parser.add_argument("--affinity_mode", type=str, default=None, 
                        choices=['uniform', 'frequency', 'binary'],
                        help="Affinity mode: 'uniform' (1/N), 'frequency' (weighted by occ count), 'binary'")
    
    # Execution configuration
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs for grid search (-1 for all CPUs). Default: 1")
    
    parser.add_argument("--grid_search_split", type=str, 
                        choices=["train", "validation", "train+validation"],
                        default="validation",
                        help="Split to use for grid search optimization. Default: 'validation'.")

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    scorer = FusedScorer(args.esco_dir, args.label_encoder, isco_level=args.isco_level, essentials_only=args.use_essentials)
    scorer.build_lookup_tables()
    scorer.build_affinity_matrix(mode=args.affinity_mode if args.affinity_mode else 'uniform')
    
    if args.task_a and args.task_b and args.isco_preds and args.decorte_map:
        ground_truth_paths = {
            'decorte_map': args.decorte_map
        }
        load_data_and_run(
            scorer, 
            Path(args.task_a), 
            Path(args.task_b), 
            Path(args.isco_preds), 
            ground_truth_paths,
            Path(args.output_dir),
            args
        )
    else:
        logger.warning("No data paths provided. Running mock test only.")
        logger.info("Testing fusion pipeline with mock data...")
        
        # 1. Mock Task A (pick occupations that exist)
        if scorer.occ_to_skills:
            mock_occs = list(scorer.occ_to_skills.keys())[:5]
        else:
            mock_occs = []
            
        # 2. Mock Task B (pick skills relevant to those occupations + noise)
        mock_b_scores = {}
        if mock_occs:
            # Add some real skills
            real_skills = list(scorer.occ_to_skills[mock_occs[0]])[:10]
            for s in real_skills:
                mock_b_scores[s] = 0.9  # High base score
                
            # Add some noise skills (random ones from other occupations)
            noise_occs = list(scorer.occ_to_skills.keys())[10:11]
            if noise_occs:
                for s in list(scorer.occ_to_skills[noise_occs[0]])[:10]:
                    mock_b_scores[s] = 0.8  # High score but shouldn't match candidate set
        
        # 3. Mock ISCO probs (random distribution)
        n_groups = len(scorer.isco_index)
        mock_probs = np.random.dirichlet(np.ones(n_groups), size=1)[0]
        
        # Test multiplicative fusion with stats collection
        logger.info("\n--- Testing MULTIPLICATIVE fusion ---")
        results_mult, stats_mult = scorer.score_job(
            task_a_occupations=mock_occs,
            task_b_scores=mock_b_scores,
            isco_probs=mock_probs,
            fusion_strategy='multiplicative',
            alpha=args.alpha if args.alpha is not None else 1.0,
            gamma=args.gamma if args.gamma is not None else 1.0,
            epsilon=args.epsilon if args.epsilon is not None else 0.0,
            collect_stats=True
        )
        logger.info(f"Multiplicative output: {len(results_mult)} ranked skills")
        if results_mult:
            logger.info(f"Top 3: {results_mult[:3]}")
        if stats_mult:
            logger.info(f"Stats: {stats_mult}")
        
        # Test linear fusion with different normalizations
        for norm in ['minmax', 'zscore', 'rank']:
            logger.info(f"\n--- Testing LINEAR fusion ({norm} normalization) ---")
            results_linear, stats_linear = scorer.score_job(
                task_a_occupations=mock_occs,
                task_b_scores=mock_b_scores,
                isco_probs=mock_probs,
                fusion_strategy='linear',
                alpha=0.7,  # 70% base score, 30% ISCO
                normalization=norm,
                epsilon=args.epsilon if args.epsilon is not None else 0.0,
                collect_stats=True
            )
            logger.info(f"Linear ({norm}) output: {len(results_linear)} ranked skills")
            if results_linear:
                logger.info(f"Top 3: {results_linear[:3]}")
            if stats_linear:
                logger.info(f"Stats: {stats_linear}")

