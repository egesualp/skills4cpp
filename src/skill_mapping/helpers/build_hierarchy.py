# src/skill_mapping/build_hierarchy.py
from pathlib import Path
import pandas as pd
import json
from collections import defaultdict, deque

# ==== CONFIG (edit these) ====
PATH_ESCO = Path("data/esco_datasets")   # folder with ESCO CSVs
PATH_OUT  = Path("data/processed")
LANG = "en"  # use your ESCO language code suffix (e.g., 'en', 'de', ...)
# =============================

PILLAR_NAMES_EN = {
    "skills",
    "knowledge",
    "language skills and knowledge",
    "transversal skills and competences",
}

def read_csv_smart(path):
    return pd.read_csv(path, low_memory=False)

def extract_skill_to_groups(broader_relations_df):
    """
    Extract skill -> [group_id,...] mappings from broaderRelationsSkillPillar_<lang>.csv.
    Multiple parents are preserved.
    """
    df = broader_relations_df[
        broader_relations_df["conceptType"] == "KnowledgeSkillCompetence"
    ].rename(columns={"conceptUri": "skill_id", "broaderUri": "group_id"})[["skill_id", "group_id"]]

    # keep one-to-many
    # some CSVs may contain duplicates; drop duplicates to keep lists clean
    df = df.drop_duplicates()
    skill2groups = df.groupby("skill_id")["group_id"].apply(list).to_dict()
    return df, skill2groups

def build_group_parents(broader_relations_df):
    """
    Build group_id -> [parent_id, ...] adjacency (parents can be groups or pillar nodes).
    Multiple parents are preserved.
    """
    grp = broader_relations_df[broader_relations_df["conceptType"] == "SkillGroup"][
        ["conceptUri", "broaderUri"]
    ].rename(columns={"conceptUri": "group_id", "broaderUri": "parent_id"}).drop_duplicates()

    group2parents = grp.groupby("group_id")["parent_id"].apply(list).to_dict()
    return grp, group2parents

def find_pillar_ids(groups_df, pillar_names=PILLAR_NAMES_EN):
    """
    Identify pillar nodes (their URIs) by label match against known ESCO pillar names (English).
    """
    # Some ESCO exports contain pillars alongside groups in the same file.
    # We detect pillars by their preferredLabel.
    lut = groups_df.set_index("conceptUri")["preferredLabel"].to_dict()
    pillar_ids = {gid for gid, lbl in lut.items() if isinstance(lbl, str) and lbl.strip() in pillar_names}
    return pillar_ids

def climb_to_pillars(group_id, group2parents, pillar_ids):
    """
    Multi-hop, multi-parent upward traversal from a group_id to all reachable pillar_ids.
    Returns a list of unique pillar IDs.
    """
    if group_id in pillar_ids:
        return [group_id]
    out = set()
    q = deque([group_id])
    seen = set()
    while q:
        g = q.popleft()
        if g in seen:
            continue
        seen.add(g)
        if g in pillar_ids:
            out.add(g)
            # don't stop; there shouldn't be parents above pillars, but we keep it safe
        for p in group2parents.get(g, []):
            if p not in seen:
                q.append(p)
    return sorted(out)

def min_depth_to_pillar(group_id, group2parents, pillar_ids, max_depth=50):
    """
    Optional: compute minimum hop distance from group_id to any pillar.
    Returns an integer depth or None if no path exists.
    """
    if group_id in pillar_ids:
        return 0
    q = deque([(group_id, 0)])
    seen = {group_id}
    while q:
        g, d = q.popleft()
        for p in group2parents.get(g, []):
            if p in pillar_ids:
                return d + 1
            if p not in seen and d + 1 < max_depth:
                seen.add(p)
                q.append((p, d + 1))
    return None

def main():
    PATH_OUT.mkdir(parents=True, exist_ok=True)

    # Filenames (language-aware)
    skills_fp   = PATH_ESCO / f"skills_{LANG}.csv"
    groups_fp   = PATH_ESCO / f"skillGroups_{LANG}.csv"
    rel_fp      = PATH_ESCO / f"broaderRelationsSkillPillar_{LANG}.csv"

    print("Loading ESCO data...")
    skills = read_csv_smart(skills_fp)
    groups = read_csv_smart(groups_fp)
    broader_relations = read_csv_smart(rel_fp)

    print(f"  ✓ Loaded {len(skills):,} skills")
    print(f"  ✓ Loaded {len(groups):,} skill groups (and pillar nodes)")
    print(f"  ✓ Loaded {len(broader_relations):,} broader relations")

    # --- Build skill -> groups (multi-parent preserved)
    print("\nExtracting skill-to-group mappings (multi-parent)...")
    s2g_df, skill2groups = extract_skill_to_groups(broader_relations)
    print(f"  ✓ Skills with at least one parent group: {len(skill2groups):,}")

    # --- Build group -> parents (can be groups or pillars, multi-parent preserved)
    print("Extracting group-to-parent mappings (multi-parent)...")
    grp_edges_df, group2parents = build_group_parents(broader_relations)
    print(f"  ✓ Groups with at least one parent: {len(group2parents):,}")

    # --- Identify pillar nodes
    pillar_ids = find_pillar_ids(groups)
    if not pillar_ids:
        print("  [!] No pillars detected by label. Double-check language / pillar names.")
    else:
        print(f"  ✓ Detected {len(pillar_ids)} pillar nodes")

    # --- Build group -> pillars (climb multi-hop)
    print("\nClimbing from groups to pillars (multi-hop, multi-parent)...")
    group_ids = groups["conceptUri"].tolist()
    group2pillars = {gid: climb_to_pillars(gid, group2parents, pillar_ids) for gid in group_ids}

    # --- Build skill -> pillars via any group
    print("Linking skills to pillars via their groups...")
    skill2pillars = {}
    for sid, gids in skill2groups.items():
        pill_set = set()
        for g in gids:
            for p in group2pillars.get(g, []):
                pill_set.add(p)
        if pill_set:
            skill2pillars[sid] = sorted(pill_set)
        else:
            # keep empty if no path to pillar was found (orphan/edge case)
            skill2pillars[sid] = []

    # --- Labels LUTs
    skill_labels = skills.rename(columns={
        "conceptUri": "skill_id",
        "preferredLabel": "skill_label"
    })[["skill_id", "skill_label"]]

    group_tbl = groups.rename(columns={
        "conceptUri": "group_id",
        "preferredLabel": "group_label"
    })[["group_id", "group_label"]]

    pillar_tbl = groups.rename(columns={
        "conceptUri": "pillar_id",
        "preferredLabel": "pillar_label"
    })[["pillar_id", "pillar_label"]]

    # --- Build a flat lookup table (explode multi-parents)
    print("\nBuilding complete multi-parent skill lookup table...")
    # skill_id, group_id rows (may repeat skill_id)
    s2g_flat = s2g_df.merge(skill_labels, on="skill_id", how="left").merge(group_tbl, on="group_id", how="left")

    # attach pillar list per group, then explode
    g2p_df = pd.DataFrame(
        [(g, p) for g, ps in group2pillars.items() for p in (ps if ps else [None])],
        columns=["group_id", "pillar_id"]
    )
    lookup = (s2g_flat
              .merge(g2p_df, on="group_id", how="left")
              .merge(pillar_tbl, on="pillar_id", how="left"))

    # Save expanded CSV (skill, group, pillar rows)
    PATH_OUT.mkdir(parents=True, exist_ok=True)
    out_csv = PATH_OUT / "skill_to_group_pillar.csv"
    lookup.to_csv(out_csv, index=False)
    print(f"  ✓ Saved {out_csv.name} ({len(lookup):,} rows)")

    # --- JSON artifacts (lists preserved)
    out_skill2group = PATH_OUT / "skill2group.json"
    out_group2label = PATH_OUT / "group2label.json"
    out_group2parent = PATH_OUT / "group2parent.json"
    out_group2pillar = PATH_OUT / "group2pillar.json"
    out_skill2pillar = PATH_OUT / "skill2pillar.json"

    # group labels
    group2label = dict(zip(group_tbl["group_id"], group_tbl["group_label"]))

    # Convert keys/values to plain lists for JSON
    def _to_plain_dict_list(d):
        return {k: list(v) if isinstance(v, (set, list, tuple)) else ([v] if pd.notna(v) else [])
                for k, v in d.items()}

    with open(out_skill2group, "w") as f:
        json.dump(_to_plain_dict_list(skill2groups), f, indent=2)
    with open(out_group2label, "w") as f:
        json.dump(group2label, f, indent=2)
    with open(out_group2parent, "w") as f:
        json.dump(_to_plain_dict_list(group2parents), f, indent=2)
    with open(out_group2pillar, "w") as f:
        json.dump(_to_plain_dict_list(group2pillars), f, indent=2)
    with open(out_skill2pillar, "w") as f:
        json.dump(_to_plain_dict_list(skill2pillars), f, indent=2)

    print(f"  ✓ Saved {out_skill2group.name} ({len(skill2groups):,} keys)")
    print(f"  ✓ Saved {out_group2label.name} ({len(group2label):,} labels)")
    print(f"  ✓ Saved {out_group2parent.name} ({len(group2parents):,} keys)")
    print(f"  ✓ Saved {out_group2pillar.name} ({len(group2pillars):,} keys)")
    print(f"  ✓ Saved {out_skill2pillar.name} ({len(skill2pillars):,} keys)")

    # --- Summary stats
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total_skills = skills.shape[0]
    total_groups = groups.shape[0]
    skills_with_groups = len(skill2groups)
    skills_with_pillars = sum(1 for v in skill2pillars.values() if v)
    orphans_skills = skills_with_groups - skills_with_pillars

    print(f"Total skills in ESCO file: {total_skills:,}")
    print(f"Total groups/pillars rows in groups file: {total_groups:,}")
    print(f"Skills mapped to ≥1 group: {skills_with_groups:,}")
    print(f"Skills with ≥1 pillar (via any group): {skills_with_pillars:,}  (orphans: {orphans_skills:,})")

    # Pillar distribution over skills (count a skill once per pillar)
    pillar_name_lut = dict(zip(pillar_tbl["pillar_id"], pillar_tbl["pillar_label"]))
    pillar_counts = defaultdict(int)
    for sid, plist in skill2pillars.items():
        for p in plist:
            if p is not None:
                pillar_counts[pillar_name_lut.get(p, "UNKNOWN")] += 1
    if pillar_counts:
        print("\nPillar distribution over skills (multi-count if skill in multiple pillars):")
        for name, cnt in sorted(pillar_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {name}: {cnt:,}")
    else:
        print("\n[!] No pillars reached from any skill. Check pillar detection and relations traversal.")

    # Depth stats (optional but informative)
    print("\nComputing group depth (min hops to a pillar)...")
    depths = []
    for gid in group_ids:
        d = min_depth_to_pillar(gid, group2parents, pillar_ids)
        depths.append(d)
    known_depths = [d for d in depths if d is not None]
    if known_depths:
        print(f"  ✓ Group→pillar min-depth: min={min(known_depths)}, median={pd.Series(known_depths).median()}, max={max(known_depths)}")
        print(f"  ✓ Groups with no path to pillar: {sum(d is None for d in depths):,}")
    else:
        print("  [!] Could not compute group depths (no paths).")

    print("\n[✓] Build complete!")

if __name__ == "__main__":
    main()
