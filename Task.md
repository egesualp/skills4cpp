💡 Goal: Set up the foundational taxonomy and mapping components for Hierarchical Skill Prediction on top of my existing TalentCLEF/DECORTE evaluation repo.
Create the following modules and scripts so we can later train category and skill models.
✅ Tasks
1️⃣ Taxonomy builder
In build_taxonomy.py, explicitly support these filenames under data/taxonomy/raw/:
skills_en.csv
skillGroups_en.csv
skillsHierarchy_en.csv
skillSkillRelations_en.csv
(optional) occupations_en.csv, occupationSkillRelations_en.csv, ISCOGroups_en.csv
Implement header auto-detection helpers because ESCO releases sometimes tweak column names:
# inside build_taxonomy.py
CANDIDATE_COLS = {
    "uri": ["conceptUri","skillUri","groupUri","occupationUri","uri"],
    "label": ["preferredLabel","prefLabel","label","title"],
    "alt": ["altLabels","altLabel","alternativeLabels"],
    "desc": ["description","scopeNote","definition","note"],
    "broader": ["broaderUri","broaderConceptUri","broaderSkillUri","broaderGroupUri","broader"]
}

def pick(df, keys):
    for k in keys:
        if k in df.columns: 
            return k
    # fuzzy fallback
    lowered = {c.lower(): c for c in df.columns}
    for k in keys:
        for c in lowered:
            if k.lower() in c:
                return lowered[c]
    raise KeyError(f"None of {keys} found in columns: {list(df.columns)}")
Then:
# load skills
skills = pd.read_csv(raw/"skills_en.csv")
uri_col = pick(skills, CANDIDATE_COLS["uri"])
lbl_col = pick(skills, CANDIDATE_COLS["label"])
alt_col = pick(skills, CANDIDATE_COLS["alt"])
desc_col = pick(skills, CANDIDATE_COLS["desc"])
# normalize & keep {uri, name, altLabels, desc}

# load groups (categories)
groups = pd.read_csv(raw/"skillGroups_en.csv")
g_uri = pick(groups, CANDIDATE_COLS["uri"])
g_label = pick(groups, CANDIDATE_COLS["label"])
g_desc = pick(groups, CANDIDATE_COLS["desc"])

# load hierarchy: skill -> broader (group or skill)
hier = pd.read_csv(raw/"skillsHierarchy_en.csv")
h_child = pick(hier, ["skillUri","conceptUri","narrowerUri","childUri","uri"])
h_parent = pick(hier, CANDIDATE_COLS["broader"])
Build a parent map and climb to the nearest group ancestor to assign category_id:
parent = dict(zip(hier[h_child], hier[h_parent]))
group_uris = set(groups[g_uri])

def nearest_group(u):
    seen = set()
    cur = u
    for _ in range(10):  # cap depth
        p = parent.get(cur)
        if p is None or p in seen: 
            return None
        if p in group_uris:
            return p
        seen.add(p); cur = p
    return None
Explode altLabels safely (they’re usually pipe- or comma-separated). Keep language code if present like "Java; Java programming@en".
Write the four processed TSVs exactly as above.
(Optional) From skillSkillRelations_en.csv, add edges skill↔skill to a skill_edges.tsv for graph methods later:
src_skill_id	dst_skill_id	relation_type
2️⃣ DECORTE → taxonomy mapping
Create src/datasets/decorte_to_taxonomy.py.
It should:
Read DECORTE skill/label schema.
Load processed taxonomy files from data/taxonomy/processed/.
Fuzzy/alias match DECORTE labels to skills/categories using:
exact alias match
fallback: cosine similarity between embeddings + Jaccard overlap
Output data/decorte/label_mapping.tsv with columns:
decorte_label, category_id, skill_id (nullable), match_score, match_method
If similarity < 0.75, flag in an “unreviewed” section or file for manual checking.
3️⃣ Hierarchical dataset view
Create src/datasets/decorte_hier_views.py.
It should:
Use the mapping above to expand each DECORTE example into:
y_cat: multi-hot category labels.
y_skill[c]: per-category multi-hot skill labels.
Save to data/decorte/processed/{train,dev,test}.jsonl, format:
{"text": "...", "y_cat": ["cat_id1", "cat_id2"], "y_skill": {"cat_id1": ["skillA", "skillB"]}}
📦 Expected repo structure after execution
src/
  taxonomy/
    build_taxonomy.py
    embeddings.py
    hard_negatives.py
  datasets/
    decorte_to_taxonomy.py
    decorte_hier_views.py
data/
  taxonomy/
    raw/
    processed/
  decorte/
    processed/
Makefile
🧠 Constraints
Use argparse CLIs for each script.
Use only standard Python + pandas, numpy, scikit-learn, torch, and tqdm.
Each script should be runnable independently (no repo-specific imports).
Keep clean docstrings and minimal comments so I can audit logic easily.
Deliverable: All scripts and Makefile modifications created and staged, ready to run locally. Do not start model training yet.