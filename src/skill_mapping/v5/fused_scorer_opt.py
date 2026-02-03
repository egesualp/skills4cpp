
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple, Literal, Any
import numpy as np
from scipy.stats import rankdata
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import sqlite3
import pandas as pd

logger = logging.getLogger(__name__)

FusionStrategy = Literal['multiplicative', 'linear']
NormalizationMethod = Literal['minmax', 'zscore', 'rank']

class TaskBManager:
    """Manages Task B data (40GB+) using a SQLite cache to avoid OOM."""
    def __init__(self, json_path: Path, cache_dir: Optional[Path] = None):
        self.json_path = Path(json_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.json_path.parent
        self.conn = None
        self._keys = None
        self._unique_skills = None
        
        self.db_path, self.skills_path = self._get_cache_paths()
        self._init_db()

    def __getstate__(self):
        """Allow pickling by removing the SQLite connection object."""
        state = self.__dict__.copy()
        state['conn'] = None
        return state

    def __setstate__(self, state):
        """Restore state and reconnect to SQLite database."""
        self.__dict__.update(state)
        # Reconnect if db exists
        if self.db_path and self.db_path.exists():
            self.conn = sqlite3.connect(str(self.db_path))
        else:
            self.conn = None

    def _get_cache_paths(self):
        # Try alongside input first (best for reusability)
        try:
            base = self.json_path.with_suffix('')
            db = base.parent / (base.name + ".cache.db")
            skills = base.parent / (base.name + ".skills.json")
            # Test write access by checking parent
            if not db.exists():
                test_file = db.parent / ".test_write"
                test_file.touch()
                test_file.unlink()
            return db, skills
        except Exception as e:
            logger.warning(f"Cannot write cache to input dir ({e}). Using cache_dir: {self.cache_dir}")
            base = self.cache_dir / self.json_path.name
            return base.with_suffix(".cache.db"), base.with_suffix(".skills.json")

    def _init_db(self):
        if self.db_path.exists() and self.skills_path.exists():
            logger.info(f"Using existing Task B cache: {self.db_path}")
            self.conn = sqlite3.connect(str(self.db_path))
        else:
            self._build_db()

    def _build_db(self):
        logger.info(f"Building cache DB from {self.json_path} (this may take a while)...")
        if self.db_path.exists(): self.db_path.unlink()
        
        self.conn = sqlite3.connect(str(self.db_path))
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE task_b (job_id TEXT PRIMARY KEY, data TEXT)")
        cur.execute("PRAGMA synchronous = OFF")
        cur.execute("PRAGMA journal_mode = WAL")
        
        unique_skills = set()
        decoder = json.JSONDecoder()
        
        # Buffer-based parser for "Key": Value format
        # Assumes file structure is roughly { "0": [...], "1": [...] }
        file_size = self.json_path.stat().st_size
        
        with open(self.json_path, 'r') as f, tqdm(total=file_size, unit='B', unit_scale=True, desc="Building DB") as pbar:
            buf = ""
            count = 0
            while True:
                chunk = f.read(16 * 1024 * 1024) # 16MB chunks
                if not chunk:
                    break
                buf += chunk
                pbar.update(len(chunk))
                
                while True:
                    buf = buf.lstrip()
                    if not buf: break
                    
                    # Skip separators
                    if buf[0] in '{},':
                        buf = buf[1:]
                        continue
                        
                    # Expect "key":
                    if buf[0] != '"': break # Should be quote
                    
                    quote_end = buf.find('"', 1)
                    if quote_end == -1: break # Need more data
                    
                    key = buf[1:quote_end]
                    
                    colon_pos = buf.find(':', quote_end)
                    if colon_pos == -1: break # Need more data
                    
                    # Value starts after colon
                    search_start = colon_pos + 1
                    while search_start < len(buf) and buf[search_start].isspace():
                        search_start += 1
                        
                    if search_start >= len(buf): break
                    
                    try:
                        # Parse ONE JSON object (the value)
                        val, end_idx = decoder.raw_decode(buf, search_start)
                        
                        # Process value
                        scores = {}
                        if isinstance(val, list):
                            for item in val:
                                uri = item.get('skill_uri')
                                if uri:
                                    scores[uri] = float(item.get('score', 0))
                                    unique_skills.add(uri)
                        elif isinstance(val, dict):
                            for k, v in val.items():
                                scores[k] = float(v)
                                unique_skills.add(k)
                                
                        cur.execute("INSERT INTO task_b VALUES (?, ?)", (key, json.dumps(scores)))
                        count += 1
                        if count % 5000 == 0:
                            self.conn.commit()

                        # Advance buffer
                        buf = buf[end_idx:]
                        
                    except json.JSONDecodeError:
                        break # Need more data
        
            self.conn.commit()
            
        # Save unique skills
        with open(self.skills_path, "w") as f:
            json.dump(sorted(list(unique_skills)), f)
        logger.info(f"Cache built. Saved {len(unique_skills)} unique skills.")

    def keys(self) -> List[str]:
        if self._keys is None:
            cur = self.conn.cursor()
            cur.execute("SELECT job_id FROM task_b")
            self._keys = [row[0] for row in cur.fetchall()]
        return self._keys
    
    def get_unique_skills(self) -> List[str]:
        if self._unique_skills is None:
            with open(self.skills_path, 'r') as f:
                self._unique_skills = json.load(f)
        return self._unique_skills

    def get_batch(self, job_ids: List[str]) -> Dict[str, Dict[str, float]]:
        res = {}
        chunk_size_sql = 900
        cursor = self.conn.cursor()
        
        # Deduplicate and filter empty
        unique_ids = list(set(jid for jid in job_ids if jid))
        
        for i in range(0, len(unique_ids), chunk_size_sql):
             batch = unique_ids[i:i+chunk_size_sql]
             placeholders = ','.join(['?'] * len(batch))
             sql = f"SELECT job_id, data FROM task_b WHERE job_id IN ({placeholders})"
             cursor.execute(sql, batch)
             for jid, data_str in cursor.fetchall():
                 res[jid] = json.loads(data_str)
        return res

@dataclass
class JobScoringStats:
    # Minimal stats container for compatibility
    num_task_a_candidates: int = 0
    num_with_task_b_scores: int = 0

def clean_isco_code(code: str) -> Optional[str]:
    if code is None or str(code).lower() == "nan": return None
    s = str(code).strip()
    if s.endswith(".0"): s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    if not s: return None
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
        
        self.occ_to_skills: Dict[str, Set[str]] = {}
        self.skill_to_isco: Dict[str, Set[str]] = {}
        self.skill_to_isco_counts: Dict[str, Dict[str, int]] = {}
        self.isco_index: List[str] = []
        
        self.skill_index: List[str] = []
        self.skill_uri_to_idx: Dict[str, int] = {}
        self.affinity_matrix: Optional[np.ndarray] = None
        self.affinity_mode: str = 'uniform'
        
    def build_lookup_tables(self):
        logger.info("Building lookup tables...")
        self._load_isco_index()
        self._build_mappings()
        logger.info(f"Built lookups: {len(self.occ_to_skills)} occupations, {len(self.skill_to_isco)} skills")

    def _load_isco_index(self):
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

    def _build_mappings(self):
        occupations_path = self.esco_dir / "occupations_en.csv"
        relations_path = self.esco_dir / "occupationSkillRelations_en.csv"
        
        occs_df = pd.read_csv(occupations_path, usecols=['conceptUri', 'iscoGroup'])
        valid_labels = set(self.isco_index)
        occ_isco_map = {}
        label_len = self.isco_level if self.isco_level else (len(self.isco_index[0]) if self.isco_index else 4)
        
        for _, row in occs_df.iterrows():
            uri = row['conceptUri']
            raw_code = clean_isco_code(row['iscoGroup'])
            if not raw_code: continue
            candidate = raw_code[:label_len]
            if candidate in valid_labels:
                occ_isco_map[uri] = candidate
        
        rels_df = pd.read_csv(relations_path, usecols=['occupationUri', 'skillUri', 'relationType'])
        self.occ_to_skills = rels_df.groupby('occupationUri')['skillUri'].apply(set).to_dict()
        
        essential_rels = rels_df[rels_df['relationType'] == 'essential'] if self.essentials_only else rels_df.copy()
        essential_rels = essential_rels.copy()
        essential_rels['iscoGroup'] = essential_rels['occupationUri'].map(occ_isco_map)
        essential_rels = essential_rels.dropna(subset=['iscoGroup'])
        
        self.skill_to_isco = essential_rels.groupby('skillUri')['iscoGroup'].apply(set).to_dict()
        
        skill_isco_counts = essential_rels.groupby(['skillUri', 'iscoGroup']).size().reset_index(name='count')
        self.skill_to_isco_counts = {}
        for _, row in skill_isco_counts.iterrows():
            if row['skillUri'] not in self.skill_to_isco_counts: self.skill_to_isco_counts[row['skillUri']] = {}
            self.skill_to_isco_counts[row['skillUri']][row['iscoGroup']] = row['count']

    def build_affinity_matrix(self, mode: str = 'uniform'):
        logger.info(f"Building Affinity Matrix ({mode})...")
        self.affinity_mode = mode
        if not self.skill_to_isco: return
        self.skill_index = sorted(self.skill_to_isco.keys())
        self.skill_uri_to_idx = {uri: i for i, uri in enumerate(self.skill_index)}
        
        n_skills = len(self.skill_index)
        n_groups = len(self.isco_index)
        self.affinity_matrix = np.zeros((n_skills, n_groups), dtype=np.float32)
        isco_label_to_idx = {l: i for i, l in enumerate(self.isco_index)}
        
        for i, skill_uri in enumerate(self.skill_index):
            groups = self.skill_to_isco.get(skill_uri, set())
            valid_groups = [(g, isco_label_to_idx[g]) for g in groups if g in isco_label_to_idx]
            if not valid_groups: continue
            
            if mode == 'uniform':
                prob = 1.0 / len(valid_groups)
                for _, col in valid_groups: self.affinity_matrix[i, col] = prob
            elif mode == 'frequency':
                counts = self.skill_to_isco_counts.get(skill_uri, {})
                raw = [(col, counts.get(g, 1)) for g, col in valid_groups]
                total = sum(w for _, w in raw)
                if total > 0:
                    for col, w in raw: self.affinity_matrix[i, col] = w / total
            elif mode == 'binary':
                for _, col in valid_groups: self.affinity_matrix[i, col] = 1.0

    def get_skill_affinity_vector(self, skill_uri: str) -> Optional[np.ndarray]:
        idx = self.skill_uri_to_idx.get(skill_uri)
        return self.affinity_matrix[idx] if idx is not None else None

    def normalize_scores(self, values: np.ndarray, method: NormalizationMethod = 'minmax') -> np.ndarray:
        if len(values) == 0: return values
        if method == 'minmax':
            mn, mx = np.min(values), np.max(values)
            if mx - mn < 1e-10: return np.full_like(values, 0.5)
            return (values - mn) / (mx - mn)
        elif method == 'zscore':
            mn, std = np.mean(values), np.std(values)
            if std < 1e-10: return np.full_like(values, 0.5)
            return 1.0 / (1.0 + np.exp(-((values - mn) / std)))
        elif method == 'rank':
            if len(values) <= 1: return np.full_like(values, 0.5)
            return (rankdata(values, method='average') - 1) / (len(values) - 1)
        return values

    def evaluate_batch_vectorized(
        self,
        task_a_preds: Dict[str, List[str]],
        task_b_preds: Any, # Dict or TaskBManager
        isco_preds: Dict[str, np.ndarray],
        ground_truth: Dict[str, Set[str]],
        task_a_k: int = 5,
        fusion_strategy: FusionStrategy = 'multiplicative',
        alpha: float = 1.0,
        gamma: float = 1.0,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        normalization: NormalizationMethod = 'minmax',
        chunk_size: int = 20000,
        mode: str = 'full',
        return_predictions: bool = False,
    ) -> Dict[str, object]:
        """Optimized vectorized batch evaluation with chunking and prediction return."""
        
        # 1. Determine common IDs
        if hasattr(task_b_preds, 'keys'):
            # This works for both Dict and TaskBManager
            b_keys = set(task_b_preds.keys())
        else:
            b_keys = set() # Should not happen
            
        common_ids = b_keys & set(ground_truth.keys())
        if mode != 'task_b_only':
            if mode == 'full' or mode == 'task_a_filter_only':
                common_ids &= set(task_a_preds.keys())
            if mode == 'full':
                common_ids &= set(isco_preds.keys())
        
        job_ids = sorted(list(common_ids))
        n_total_jobs = len(job_ids)
        logger.info(f"Vectorized evaluation on {n_total_jobs} jobs (Chunk size {chunk_size}, Mode {mode})...")
        
        if n_total_jobs == 0:
            return {"mAP": 0.0, "R@10": 0.0, "R@50": 0.0, "R@100": 0.0, "predictions": {}} if return_predictions else {"mAP": 0.0, "R@10": 0.0, "R@50": 0.0, "R@100": 0.0}

        # 2. Build Unified Skill Index
        if hasattr(task_b_preds, 'get_unique_skills'):
             skill_list = sorted(list(task_b_preds.get_unique_skills()))
        else:
             all_skills = set()
             for job_id in task_b_preds.keys():
                 all_skills.update(task_b_preds[job_id].keys())
             skill_list = sorted(all_skills)
             
        skill_to_idx = {s: i for i, s in enumerate(skill_list)}
        idx_to_skill = {i: s for i, s in enumerate(skill_list)}
        n_skills = len(skill_list)
        logger.info(f"  Total unique skills in Task B: {n_skills}")
        
        # Precompute Skill Affinity Matrix
        n_groups = len(self.isco_index)
        skill_affinity = np.ones((n_skills, n_groups), dtype=np.float32)
        skill_has_affinity = np.zeros(n_skills, dtype=bool)
        
        if mode == 'full' and gamma != 0:
            for skill, idx in skill_to_idx.items():
                aff = self.get_skill_affinity_vector(skill)
                if aff is not None:
                    skill_affinity[idx] = aff
                    skill_has_affinity[idx] = True
            logger.info("  Built skill affinity matrix")

        # Accumulators
        all_aps = []
        all_recalls_10 = []
        all_recalls_50 = []
        all_recalls_100 = []
        final_predictions = {}
        
        for chunk_start in tqdm(range(0, n_total_jobs, chunk_size), desc="Processing Batches"):
            chunk_job_ids = job_ids[chunk_start : chunk_start + chunk_size]
            n_chunk = len(chunk_job_ids)
            
            # Fetch Batch Data
            if hasattr(task_b_preds, 'get_batch'):
                 batch_b_preds = task_b_preds.get_batch(chunk_job_ids)
            else:
                 batch_b_preds = {j: task_b_preds[j] for j in chunk_job_ids if j in task_b_preds}

            # A. ISCO Matrix
            if mode == 'full' and gamma != 0:
                isco_matrix = np.zeros((n_chunk, n_groups), dtype=np.float32)
                for i, job_id in enumerate(chunk_job_ids):
                    probs = isco_preds.get(job_id, np.zeros(n_groups))
                    # Temperature scaling
                    if temperature != 1.0 and temperature > 0 and np.sum(probs) > 0:
                         probs = np.power(probs, 1.0/temperature)
                         probs /= np.sum(probs)
                    isco_matrix[i] = probs
                
                isco_weights = isco_matrix @ skill_affinity.T
                isco_weights[:, ~skill_has_affinity] = 1.0
                if epsilon > 0: isco_weights = np.maximum(isco_weights, epsilon)
            else:
                isco_weights = None

            # B. Score Matrix & Candidate Mask
            score_matrix = np.full((n_chunk, n_skills), -np.inf, dtype=np.float32)
            candidate_mask = np.zeros((n_chunk, n_skills), dtype=bool)

            for i, job_id in enumerate(chunk_job_ids):
                b_scores = batch_b_preds.get(job_id, {})
                min_score = min(b_scores.values()) if b_scores else 0.0
                
                # Fill available
                for skill, sc in b_scores.items():
                    if skill in skill_to_idx:
                        score_matrix[i, skill_to_idx[skill]] = sc
                
                # Candidates
                candidates = set()
                if mode == 'task_b_only':
                    for s in b_scores.keys(): candidates.add(s)
                elif mode == 'task_a_filter_only':
                    occs = task_a_preds.get(job_id, [])[:task_a_k]
                    for occ in occs:
                        if occ in self.occ_to_skills: candidates.update(self.occ_to_skills[occ])
                    if not candidates: candidates = set(b_scores.keys())
                else: # mode == 'full' -> UNION
                    for s in b_scores.keys(): candidates.add(s)
                    occs = task_a_preds.get(job_id, [])[:task_a_k]
                    for occ in occs:
                        if occ in self.occ_to_skills: candidates.update(self.occ_to_skills[occ])

                for cand in candidates:
                    if cand in skill_to_idx:
                        c_idx = skill_to_idx[cand]
                        candidate_mask[i, c_idx] = True
                        if score_matrix[i, c_idx] == -np.inf:
                           score_matrix[i, c_idx] = min_score
            
            # C. Fusion
            valid_mask = score_matrix > -np.inf
            fused_scores = np.full_like(score_matrix, -np.inf)
            
            if mode == 'task_b_only' or mode == 'task_a_filter_only' or gamma == 0:
                fused_scores = np.where(valid_mask, score_matrix, -np.inf)
            elif fusion_strategy == 'multiplicative':
                base_term = np.where(valid_mask, score_matrix, 0.0)
                base_term = np.maximum(base_term, 1e-6)
                fused_scores = np.where(valid_mask, np.power(base_term, alpha) * np.power(isco_weights, gamma), -np.inf)
            else: # linear
                for i in range(n_chunk):
                    row_valid = valid_mask[i]
                    if not np.any(row_valid): continue
                    
                    base_vals = score_matrix[i, row_valid]
                    isco_vals = isco_weights[i, row_valid]
                    
                    b_norm = self.normalize_scores(base_vals, method=normalization)
                    i_norm = self.normalize_scores(isco_vals, method=normalization)
                    
                    fused_scores[i, row_valid] = alpha * b_norm + (1-alpha) * i_norm

            if mode != 'task_b_only':
                fused_scores = np.where(candidate_mask, fused_scores, -np.inf)

            # D. Ranking & Metrics
            gold_sets = [ground_truth[jid] for jid in chunk_job_ids]
            
            for i in range(n_chunk):
                row_scores = fused_scores[i]
                valid_indices = np.where(row_scores > -np.inf)[0]
                
                if len(valid_indices) == 0:
                    if return_predictions: final_predictions[chunk_job_ids[i]] = []
                    all_aps.append(0.0)
                    all_recalls_10.append(0.0)
                    all_recalls_50.append(0.0)
                    all_recalls_100.append(0.0)
                    continue

                sorted_indices = valid_indices[np.argsort(-row_scores[valid_indices])]
                pred_uris = [idx_to_skill[idx] for idx in sorted_indices]
                
                if return_predictions:
                    top_preds = [(idx_to_skill[idx], float(row_scores[idx])) for idx in sorted_indices[:100]]
                    final_predictions[chunk_job_ids[i]] = top_preds
                    
                gold = gold_sets[i]
                if not gold:
                    all_aps.append(0.0)
                    all_recalls_10.append(0.0)
                    all_recalls_50.append(0.0)
                    all_recalls_100.append(0.0)
                    continue
                
                hits = 0
                sum_precs = 0.0
                for rank, uri in enumerate(pred_uris):
                    if uri in gold:
                        hits += 1
                        sum_precs += hits / (rank + 1)
                all_aps.append(sum_precs / len(gold))
                
                def calc_recall(k):
                    top = set(pred_uris[:k])
                    return len(top & gold) / len(gold)
                
                all_recalls_10.append(calc_recall(10))
                all_recalls_50.append(calc_recall(50))
                all_recalls_100.append(calc_recall(100))
        
        results = {
            "mAP": np.mean(all_aps) if all_aps else 0.0,
            "R@10": np.mean(all_recalls_10) if all_recalls_10 else 0.0,
            "R@50": np.mean(all_recalls_50) if all_recalls_50 else 0.0,
            "R@100": np.mean(all_recalls_100) if all_recalls_100 else 0.0
        }
        
        if return_predictions: results['predictions'] = final_predictions
        return results

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
        n_jobs: int = 1,
        chunk_size: int = 20000
    ):
         results = []
         best_score = -1.0
         best_params = None
         
         default_union = {0.3, 0.5, 0.7, 0.9, 1.0, 1.5}
         is_default_range = set(alpha_values) == default_union
         
         multiplicative_params = []
         linear_params = []
         
         if 'multiplicative' in fusion_strategies:
            mult_alphas = [0.5, 1.0, 1.5] if is_default_range else alpha_values
            multiplicative_params = list(itertools.product(
                k_values, ['multiplicative'], mult_alphas, gamma_values, 
                temperatures, affinity_modes, epsilon_values, ['minmax']
            ))
            
         if 'linear' in fusion_strategies:
            lin_alphas = [0.3, 0.5, 0.7, 0.9] if is_default_range else alpha_values
            linear_params = list(itertools.product(
                k_values, ['linear'], lin_alphas, [0.0],
                temperatures, affinity_modes, epsilon_values, normalization_methods
            ))
            
         param_grid = multiplicative_params + linear_params
         logger.info(f"Starting grid search with {len(param_grid)} combinations (n_jobs={n_jobs})...")
         
         unique_aff_modes = sorted(list(set(p[5] for p in param_grid)))
         
         for aff_mode in unique_aff_modes:
            if aff_mode != self.affinity_mode:
                logger.info(f"Switching affinity mode to '{aff_mode}'...")
                self.build_affinity_matrix(mode=aff_mode)
            
            sub_grid = [p for p in param_grid if p[5] == aff_mode]
            
            def process_params(p):
                k, strategy, alpha, gamma, temp, _, eps, norm = p
                # Vectorized search
                metrics = self.evaluate_batch_vectorized(
                    task_a_preds, task_b_preds, isco_preds, ground_truth,
                    task_a_k=k, fusion_strategy=strategy, alpha=alpha, gamma=gamma,
                    temperature=temp, epsilon=eps, normalization=norm, chunk_size=chunk_size
                )
                params_dict = {
                    'k': k, 'fusion_strategy': strategy, 'alpha': alpha, 'gamma': gamma,
                    'temp': temp, 'affinity_mode': aff_mode, 'epsilon': eps, 'normalization': norm
                }
                metrics['params'] = params_dict
                return metrics

            if n_jobs != 1:
                batch_results = Parallel(n_jobs=n_jobs)(delayed(process_params)(p) for p in tqdm(sub_grid))
            else:
                batch_results = [process_params(p) for p in tqdm(sub_grid)]
                
            for metrics in batch_results:
                params = metrics['params']
                logger.info(f"Evaluated: {params['fusion_strategy']}, K={params['k']}, a={params['alpha']}, g={params['gamma']} -> mAP: {metrics['mAP']:.4f}")
                results.append(metrics)
                score = metrics.get(metric, 0.0)
                if score > best_score:
                    best_score = score
                    best_params = params
                    
         logger.info(f"Grid search complete. Best {metric}: {best_score:.4f}")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    logger.info("Loading Task A...")
    task_a_preds = {}
    with open(task_a_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                data = json.loads(line)
                task_a_preds[str(i)] = data.get('predicted_esco_ids', [])
            except: pass
            
    # Load Task B using Manager
    logger.info("Initializing Task B Manager...")
    # Use output_dir as potential cache location to ensure write access if input is read-only
    task_b_manager = TaskBManager(task_b_path, cache_dir=output_dir)
        
    logger.info("Loading ISCO...")
    with open(isco_path, 'r') as f:
        raw_isco = json.load(f)
    isco_preds = {}
    label_to_idx = {lbl: i for i, lbl in enumerate(scorer.isco_index)}
    n_classes = len(scorer.isco_index)
    for job_id, data in raw_isco.items():
        prob_vec = np.zeros(n_classes, dtype=np.float32)
        if "topk" in data:
            for item in data["topk"]:
                if item["label"] in label_to_idx:
                    prob_vec[label_to_idx[item["label"]]] = item["score"]
        if np.sum(prob_vec) > 0: prob_vec /= np.sum(prob_vec)
        isco_preds[str(job_id)] = prob_vec

    logger.info("Loading Ground Truth...")
    decorte_df = pd.read_csv(ground_truth_paths['decorte_map'])
    decorte_df['job_id'] = decorte_df['job_id'].astype(str)
    job_to_occ = dict(zip(decorte_df['job_id'], decorte_df['esco_id']))
    
    splits = {}
    if 'split' in decorte_df.columns:
        splits = dict(zip(decorte_df['job_id'], decorte_df['split']))
        
    ground_truth = {jid: scorer.occ_to_skills.get(uri, set()) for jid, uri in job_to_occ.items()}
    
    common_ids = set(task_b_manager.keys()) & set(ground_truth.keys()) & set(task_a_preds.keys()) & set(isco_preds.keys())
    logger.info(f"Common jobs: {len(common_ids)}")
    
    train_ids = {jid for jid in common_ids if splits.get(jid, '').lower() in {'train', 'training'}}
    val_ids = {jid for jid in common_ids if splits.get(jid, '').lower() in {'val', 'validation', 'valid', 'dev'}}
    test_ids = {jid for jid in common_ids if splits.get(jid, '').lower() in {'test', 'testing'}}
    
    search_split = args.grid_search_split if args else 'validation'
    if search_split == 'train': search_ids = train_ids
    elif search_split == 'train+validation': search_ids = train_ids | val_ids
    else: search_ids = val_ids
    if not search_ids: search_ids = common_ids
    
    eval_ids = test_ids if test_ids else search_ids
    chunk_size = args.chunk_size if args else 20000

    def get_subset(ids, full_manager=False):
        subs_a = {k: task_a_preds[k] for k in ids if k in task_a_preds}
        subs_isco = {k: isco_preds[k] for k in ids if k in isco_preds}
        subs_gt = {k: ground_truth[k] for k in ids if k in ground_truth}
        
        if full_manager:
            return subs_a, task_b_manager, subs_isco, subs_gt
        else:
            subs_b = task_b_manager.get_batch(list(ids))
            return subs_a, subs_b, subs_isco, subs_gt

    # Grid Search
    logger.info("Running Grid Search on Search Split...")
    # For Grid Search (subset), we can load dict to avoid SQL overhead in inner loop, or pass manager.
    # Passing manager is safer for memory, but potentially slower due to repetitive SQL?
    # Actually evaluate_batch_vectorized loads batch by batch.
    # Let's pass manager.
    gs_data = get_subset(search_ids, full_manager=True)
    
    k_vals = [args.task_a_k] if args and args.task_a_k else [3, 5, 7, 10]
    # Configure other params from args or defaults...
    alpha_vals = [args.alpha] if args and args.alpha is not None else [0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
    gamma_vals = [args.gamma] if args and args.gamma is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    strategies = [args.fusion_strategy] if args and args.fusion_strategy else ['multiplicative', 'linear']
    
    results, best_params = scorer.grid_search(
        *gs_data, 
        k_values=k_vals,
        fusion_strategies=strategies,
        alpha_values=alpha_vals,
        gamma_values=gamma_vals,
        n_jobs=args.n_jobs if args else 1,
        chunk_size=chunk_size
    )
    
    # Final Eval
    best_k = best_params['k']
    best_strategy = best_params['fusion_strategy']
    best_alpha = best_params['alpha']
    best_gamma = best_params['gamma']
    best_temp = best_params['temp']
    best_mode = best_params.get('affinity_mode', 'uniform')
    best_eps = best_params.get('epsilon', 0.0)
    best_norm = best_params.get('normalization', 'minmax')
    
    logger.info(f"Optimal Params: {best_params}")
    if scorer.affinity_mode != best_mode:
        scorer.build_affinity_matrix(mode=best_mode)
        
    test_data = get_subset(eval_ids, full_manager=True)
    
    logger.info("Evaluating Baselines...")
    res_b = scorer.evaluate_batch_vectorized(*test_data, mode='task_b_only', chunk_size=chunk_size)
    logger.info(f"Task B Only: {res_b['mAP']:.4f}")

    res_a = scorer.evaluate_batch_vectorized(*test_data, mode='task_a_filter_only', task_a_k=best_k, chunk_size=chunk_size)
    logger.info(f"Task A Filter: {res_a['mAP']:.4f}")
    
    res_full = scorer.evaluate_batch_vectorized(
        *test_data, mode='full', 
        task_a_k=best_k, fusion_strategy=best_strategy, alpha=best_alpha, 
        gamma=best_gamma, temperature=best_temp, epsilon=best_eps, normalization=best_norm, chunk_size=chunk_size
    )
    logger.info(f"Full Pipeline: {res_full['mAP']:.4f}")
    
    # Final Predictions (ALL DATA)
    logger.info("Generating Predictions for ALL jobs...")
    all_data = get_subset(sorted(list(common_ids)), full_manager=True)
    
    res_all = scorer.evaluate_batch_vectorized(
        *all_data, mode='full',
        task_a_k=best_k, fusion_strategy=best_strategy, alpha=best_alpha, 
        gamma=best_gamma, temperature=best_temp, epsilon=best_eps, normalization=best_norm,
        chunk_size=chunk_size,
        return_predictions=True
    )
    
    predictions = res_all.pop('predictions')
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
            
    final_results = {
        "grid_search": results,
        "best_params": best_params,
        "test_metrics": {"task_b": res_b, "task_a": res_a, "full": res_full},
        "all_data_metrics": res_all
    }
    
    with open(output_dir / "fused_scorer_results.json", "w") as f:
        json.dump(final_results, f, indent=2, cls=NumpyEncoder)
        
    with open(output_dir / "fused_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2, cls=NumpyEncoder)
        
    logger.info("Finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--esco_dir", type=str, default="data/esco_datasets")
    parser.add_argument("--label_encoder", type=str, required=True)
    parser.add_argument("--isco_level", type=int, default=None)
    parser.add_argument("--use_essentials", action='store_true')
    parser.add_argument("--task_a", type=str)
    parser.add_argument("--task_b", type=str)
    parser.add_argument("--isco_preds", type=str)
    parser.add_argument("--decorte_map", type=str)
    parser.add_argument("--output_dir", type=str, default=".")
    
    parser.add_argument("--fusion_strategy", type=str)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--normalization", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--task_a_k", type=int)
    parser.add_argument("--affinity_mode", type=str)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--grid_search_split", type=str, default="validation")
    parser.add_argument("--chunk_size", type=int, default=20000)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    scorer = FusedScorer(args.esco_dir, args.label_encoder, isco_level=args.isco_level, essentials_only=args.use_essentials)
    scorer.build_lookup_tables()
    scorer.build_affinity_matrix(mode=args.affinity_mode if args.affinity_mode else 'uniform')
    
    if args.task_a:
        ground_truth_paths = {'decorte_map': args.decorte_map}
        load_data_and_run(
            scorer, Path(args.task_a), Path(args.task_b), Path(args.isco_preds),
            ground_truth_paths, Path(args.output_dir), args
        )
