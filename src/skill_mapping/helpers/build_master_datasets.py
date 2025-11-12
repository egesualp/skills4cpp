"""
Build comprehensive master datasets from ESCO data.

This script creates several master datasets by joining ESCO tables:
1. master_skills.csv - Skill-centric with hierarchy and occupation counts
2. master_occupation_skills.csv - Occupation-skill relationships with full context
3. master_complete_skills.csv - Complete skill dataset with all hierarchy info
4. master_graph_nodes.csv - All nodes for network analysis
5. master_graph_edges.csv - All edges for network analysis
6. master_skill_complete_hierarchy.csv - Complete hierarchy table (levels 0-3) with skills
7. master_complete_hierarchy_w_occ.csv - Occupation-skill relationships with complete hierarchy
"""

from pathlib import Path
import pandas as pd
import json
import argparse
from collections import defaultdict

# ==== CONFIG ====
PATH_ESCO = Path("data/esco_datasets")
PATH_PROCESSED = Path("data/processed")
PATH_OUT = Path("data/processed/master_datasets_2")
LANG = "en"

PILLAR_NAMES = {
    "skills",
    "knowledge",
    "language skills and knowledge",
    "transversal skills and competences"
}

def read_csv_smart(path):
    """Read CSV with low_memory=False to avoid dtype warnings."""
    return pd.read_csv(path, low_memory=False)

def load_data():
    """Load all required ESCO datasets."""
    print("Loading ESCO datasets...")
    
    data = {
        'skills': read_csv_smart(PATH_ESCO / f"skills_{LANG}.csv"),
        'occupations': read_csv_smart(PATH_ESCO / f"occupations_{LANG}.csv"),
        'skill_groups': read_csv_smart(PATH_ESCO / f"skillGroups_{LANG}.csv"),
        'broader_relations': read_csv_smart(PATH_ESCO / f"broaderRelationsSkillPillar_{LANG}.csv"),
        'occ_skill_relations': read_csv_smart(PATH_ESCO / f"occupationSkillRelations_{LANG}.csv"),
        'skill_skill_relations': read_csv_smart(PATH_ESCO / f"skillSkillRelations_{LANG}.csv"),
    }
    
    print(f"  ✓ Skills: {len(data['skills']):,}")
    print(f"  ✓ Occupations: {len(data['occupations']):,}")
    print(f"  ✓ Skill Groups: {len(data['skill_groups']):,}")
    print(f"  ✓ Broader Relations: {len(data['broader_relations']):,}")
    print(f"  ✓ Occupation-Skill Relations: {len(data['occ_skill_relations']):,}")
    print(f"  ✓ Skill-Skill Relations: {len(data['skill_skill_relations']):,}")
    
    return data

def build_master_skills(data):
    """Build skill-centric master dataset with hierarchy and occupation counts."""
    print("\n" + "="*80)
    print("Building Master Skills Dataset...")
    print("="*80)
    
    skills = data['skills']
    skill_groups = data['skill_groups']
    broader_relations = data['broader_relations']
    occ_skill_relations = data['occ_skill_relations']
    
    # Step 1: Skill to group mapping
    skill_to_group = broader_relations[
        broader_relations['conceptType'] == 'KnowledgeSkillCompetence'
    ][['conceptUri', 'broaderUri']].rename(columns={
        'conceptUri': 'skillUri',
        'broaderUri': 'groupUri'
    })
    
    # Step 2: Group labels
    group_labels = skill_groups[['conceptUri', 'preferredLabel']].rename(columns={
        'conceptUri': 'groupUri',
        'preferredLabel': 'groupLabel'
    })
    
    # Step 3: Identify pillars
    pillars = skill_groups[
        skill_groups['preferredLabel'].str.strip().isin(PILLAR_NAMES)
    ][['conceptUri', 'preferredLabel']].rename(columns={
        'conceptUri': 'pillarUri',
        'preferredLabel': 'pillarLabel'
    })
    print(f"  ✓ Identified {len(pillars)} pillars")
    
    # Step 4: Count occupations per skill
    skill_occ_counts = occ_skill_relations.groupby('skillUri').apply(
        lambda x: pd.Series({
            'total_occupations': len(x),
            'essential_occupations': (x['relationType'] == 'essential').sum(),
            'optional_occupations': (x['relationType'] == 'optional').sum()
        })
    ).reset_index()
    
    # Step 5: Build master
    master = skills[[
        'conceptUri', 'preferredLabel', 'altLabels', 'skillType', 
        'reuseLevel', 'status', 'description', 'definition'
    ]].rename(columns={
        'conceptUri': 'skillUri',
        'preferredLabel': 'skillLabel'
    })
    
    # Merge with groups
    master = master.merge(skill_to_group, on='skillUri', how='left')
    master = master.merge(group_labels, on='groupUri', how='left')
    
    # Merge with occupation counts
    master = master.merge(skill_occ_counts, on='skillUri', how='left')
    master[['total_occupations', 'essential_occupations', 'optional_occupations']] = \
        master[['total_occupations', 'essential_occupations', 'optional_occupations']].fillna(0).astype(int)
    
    print(f"  ✓ Created master skills dataset: {master.shape}")
    print(f"  ✓ Skills with groups: {master['groupUri'].notna().sum():,}")
    print(f"  ✓ Skills with occupations: {(master['total_occupations'] > 0).sum():,}")
    
    return master

def build_master_occupation_skills(data):
    """Build occupation-skill relationship master dataset."""
    print("\n" + "="*80)
    print("Building Master Occupation-Skills Dataset...")
    print("="*80)
    
    occ_skill_relations = data['occ_skill_relations']
    occupations = data['occupations']
    skills = data['skills']
    skill_groups = data['skill_groups']
    broader_relations = data['broader_relations']
    
    # Start with relationships
    master = occ_skill_relations.copy()
    
    # Add occupation details
    occ_details = occupations[[
        'conceptUri', 'preferredLabel', 'iscoGroup', 'description', 'code'
    ]].rename(columns={
        'conceptUri': 'occupationUri',
        'preferredLabel': 'occupationLabel',
        'description': 'occupationDescription',
        'code': 'occupationCode'
    })
    
    master = master.merge(occ_details, on='occupationUri', how='left')
    
    # Add skill details
    skill_details = skills[[
        'conceptUri', 'preferredLabel', 'skillType', 'reuseLevel', 'description'
    ]].rename(columns={
        'conceptUri': 'skillUri',
        'preferredLabel': 'skillLabel',
        'description': 'skillDescription'
    })
    
    master = master.merge(skill_details, on='skillUri', how='left')
    
    # Add skill groups (aggregated)
    skill_to_group = broader_relations[
        broader_relations['conceptType'] == 'KnowledgeSkillCompetence'
    ][['conceptUri', 'broaderUri']].rename(columns={
        'conceptUri': 'skillUri',
        'broaderUri': 'groupUri'
    })
    
    group_labels = skill_groups[['conceptUri', 'preferredLabel']].rename(columns={
        'conceptUri': 'groupUri',
        'preferredLabel': 'groupLabel'
    })
    
    skill_to_group = skill_to_group.merge(group_labels, on='groupUri', how='left')
    
    # Aggregate multiple groups per skill
    skill_groups_agg = skill_to_group.groupby('skillUri')['groupLabel'].apply(
        lambda x: ' | '.join(x.dropna().astype(str).unique())
    ).reset_index().rename(columns={'groupLabel': 'skillGroups'})
    
    master = master.merge(skill_groups_agg, on='skillUri', how='left')
    
    print(f"  ✓ Created master occupation-skills dataset: {master.shape}")
    print(f"  ✓ Unique occupations: {master['occupationUri'].nunique():,}")
    print(f"  ✓ Unique skills: {master['skillUri'].nunique():,}")
    print(f"  ✓ Essential relationships: {(master['relationType'] == 'essential').sum():,}")
    print(f"  ✓ Optional relationships: {(master['relationType'] == 'optional').sum():,}")
    
    return master

def build_master_complete_skills(data):
    """Build complete skills dataset with full hierarchy from build_hierarchy outputs."""
    print("\n" + "="*80)
    print("Building Complete Master Skills Dataset...")
    print("="*80)
    
    skills = data['skills']
    occ_skill_relations = data['occ_skill_relations']
    
    # Check if hierarchy files exist
    hierarchy_files = {
        'skill2group': PATH_PROCESSED / "skill2group.json",
        'skill2pillar': PATH_PROCESSED / "skill2pillar.json",
        'group2label': PATH_PROCESSED / "group2label.json"
    }
    
    missing_files = [name for name, path in hierarchy_files.items() if not path.exists()]
    
    if missing_files:
        print(f"  [!] Missing hierarchy files: {missing_files}")
        print(f"  [!] Run 'python src/skill_mapping/build_hierarchy.py' first")
        print(f"  [!] Skipping complete master dataset...")
        return None
    
    # Load hierarchy data
    with open(hierarchy_files['skill2group']) as f:
        skill2group = json.load(f)
    
    with open(hierarchy_files['skill2pillar']) as f:
        skill2pillar = json.load(f)
    
    with open(hierarchy_files['group2label']) as f:
        group2label = json.load(f)
    
    # Build master
    master = skills[[
        'conceptUri', 'preferredLabel', 'skillType', 'reuseLevel', 
        'status', 'description', 'definition', 'altLabels'
    ]].rename(columns={'conceptUri': 'skillUri', 'preferredLabel': 'skillLabel'})
    
    # Add hierarchy info
    master['groups'] = master['skillUri'].map(
        lambda x: ' | '.join([group2label.get(g, g) for g in skill2group.get(x, [])]) if x in skill2group else ''
    )
    master['groupUris'] = master['skillUri'].map(
        lambda x: ' | '.join(skill2group.get(x, [])) if x in skill2group else ''
    )
    master['pillars'] = master['skillUri'].map(
        lambda x: ' | '.join([group2label.get(p, p) for p in skill2pillar.get(x, [])]) if x in skill2pillar else ''
    )
    master['pillarUris'] = master['skillUri'].map(
        lambda x: ' | '.join(skill2pillar.get(x, [])) if x in skill2pillar else ''
    )
    
    # Add occupation counts
    skill_occ_counts = occ_skill_relations.groupby('skillUri').apply(
        lambda x: pd.Series({
            'total_occupations': len(x),
            'essential_occupations': (x['relationType'] == 'essential').sum(),
            'optional_occupations': (x['relationType'] == 'optional').sum()
        })
    ).reset_index()
    
    master = master.merge(skill_occ_counts, on='skillUri', how='left')
    master[['total_occupations', 'essential_occupations', 'optional_occupations']] = \
        master[['total_occupations', 'essential_occupations', 'optional_occupations']].fillna(0).astype(int)
    
    print(f"  ✓ Created complete master skills dataset: {master.shape}")
    print(f"  ✓ Skills with pillars: {(master['pillars'] != '').sum():,}")
    print(f"  ✓ Skills with groups: {(master['groups'] != '').sum():,}")
    
    return master

def build_graph_datasets(data):
    """Build node and edge datasets for network analysis."""
    print("\n" + "="*80)
    print("Building Graph Datasets (Nodes + Edges)...")
    print("="*80)
    
    skills = data['skills']
    occupations = data['occupations']
    occ_skill_relations = data['occ_skill_relations']
    skill_skill_relations = data['skill_skill_relations']
    
    # === NODES ===
    # Skill nodes
    skill_nodes = skills[['conceptUri', 'preferredLabel', 'skillType', 'reuseLevel']].copy()
    skill_nodes['nodeType'] = 'skill'
    skill_nodes = skill_nodes.rename(columns={
        'conceptUri': 'nodeId',
        'preferredLabel': 'label'
    })
    
    # Occupation nodes
    occ_nodes = occupations[['conceptUri', 'preferredLabel', 'iscoGroup']].copy()
    occ_nodes['nodeType'] = 'occupation'
    occ_nodes['skillType'] = None
    occ_nodes['reuseLevel'] = None
    occ_nodes = occ_nodes.rename(columns={
        'conceptUri': 'nodeId',
        'preferredLabel': 'label'
    })
    
    # Combine all nodes
    all_nodes = pd.concat([skill_nodes, occ_nodes], ignore_index=True)
    
    print(f"  ✓ Created nodes dataset: {all_nodes.shape}")
    print(f"    - Skill nodes: {(all_nodes['nodeType'] == 'skill').sum():,}")
    print(f"    - Occupation nodes: {(all_nodes['nodeType'] == 'occupation').sum():,}")
    
    # === EDGES ===
    # Occupation-Skill edges
    occ_skill_edges = occ_skill_relations.copy()
    occ_skill_edges['edgeType'] = 'requires'
    occ_skill_edges = occ_skill_edges.rename(columns={
        'occupationUri': 'source',
        'skillUri': 'target',
        'relationType': 'relationship'
    })
    occ_skill_edges['weight'] = occ_skill_edges['relationship'].map({
        'essential': 1.0,
        'optional': 0.5
    })
    
    # Skill-Skill edges
    skill_edges = skill_skill_relations[['originalSkillUri', 'relatedSkillUri', 'relationType']].copy()
    skill_edges['edgeType'] = 'relates_to'
    skill_edges = skill_edges.rename(columns={
        'originalSkillUri': 'source',
        'relatedSkillUri': 'target',
        'relationType': 'relationship'
    })
    skill_edges['weight'] = skill_edges['relationship'].map({
        'essential': 0.8,
        'optional': 0.3
    })
    
    # Combine edges
    all_edges = pd.concat([
        occ_skill_edges[['source', 'target', 'edgeType', 'relationship', 'weight']],
        skill_edges[['source', 'target', 'edgeType', 'relationship', 'weight']]
    ], ignore_index=True)
    
    print(f"  ✓ Created edges dataset: {all_edges.shape}")
    print(f"    - Occupation→Skill edges: {(all_edges['edgeType'] == 'requires').sum():,}")
    print(f"    - Skill→Skill edges: {(all_edges['edgeType'] == 'relates_to').sum():,}")
    
    return all_nodes, all_edges

def build_complete_hierarchy_table(data):
    """
    Build complete skill hierarchy table with all possible paths.
    Traces parentage through both SkillGroups and KnowledgeSkillCompetences.
    """
    print("\n" + "="*80)
    print("Building Complete Skill Hierarchy Table (All Paths)...")
    print("="*80)
    
    skills = data['skills']
    skill_groups = data['skill_groups']
    broader_relations = data['broader_relations']
    
    # Identify pillars
    pillar_ids = set(skill_groups[
        skill_groups['preferredLabel'].str.strip().isin(PILLAR_NAMES)
    ]['conceptUri'])
    
    print(f"  ✓ Identified {len(pillar_ids)} pillars")
    
    # === FIX 1: Create a Universal Parent Map ===
    # Build ALL to parent mapping (traces both skills and groups)
    print("  ✓ Building universal parent map...")
    all_concept_parents = {}
    # Use all broader relations
    for _, row in broader_relations.iterrows():
        concept_uri = row['conceptUri']
        parent_uri = row['broaderUri']
        if concept_uri not in all_concept_parents:
            all_concept_parents[concept_uri] = []
        all_concept_parents[concept_uri].append(parent_uri)
    # Modified to use all broaderRelations, not just SkillGroup

    # === FIX 2: Create a Universal Label Map ===
    # Get ALL labels (for both skills and groups)
    print("  ✓ Building universal label map...")
    all_labels = pd.concat([
        skill_groups[['conceptUri', 'preferredLabel']],
        skills[['conceptUri', 'preferredLabel']]
    ]).drop_duplicates(subset=['conceptUri'])
    all_labels_dict = dict(zip(all_labels['conceptUri'], all_labels['preferredLabel']))
    # Combines skill and group labels

    # Build skill to immediate parents mapping (now includes skill-to-skill)
    skill2parents = {}
    skill_edges = broader_relations[broader_relations['conceptType'] == 'KnowledgeSkillCompetence']
    for _, row in skill_edges.iterrows():
        sid = row['conceptUri']
        gid = row['broaderUri'] # This 'gid' can be a group OR another skill
        if sid not in skill2parents:
            skill2parents[sid] = []
        skill2parents[sid].append(gid)
    
    
    def get_paths_to_pillar(concept_id, parent_map, pillar_ids, max_depth=10):
        """
        Get all paths from a concept (skill or group) to pillars using DFS.
        Returns list of paths, where each path is [concept, parent, grandparent, ..., pillar]
        """
        if concept_id in pillar_ids:
            return [[concept_id]]
        
        all_paths = []
        
        def dfs(current, path, visited):
            if len(path) > max_depth:
                return
            
            if current in pillar_ids:
                all_paths.append(path[:])
                return
            
            # === FIX 3: Use the universal parent_map ===
            for parent in parent_map.get(current, []):
                if parent not in visited:
                    visited.add(parent)
                    path.append(parent)
                    dfs(parent, path, visited)
                    path.pop()
                    visited.remove(parent)
        
        visited = {concept_id}
        dfs(concept_id, [concept_id], visited)
        
        return all_paths
    # Modified to use generic 'parent_map'

    # Build the complete hierarchy table
    print("  ✓ Building hierarchy paths for all skills...")
    rows = []
    
    for _, skill_row in skills.iterrows():
        skill_uri = skill_row['conceptUri']
        skill_label = skill_row['preferredLabel']
        skill_type = skill_row['skillType']
        skill_alt_labels = skill_row['altLabels']
        reuse_level = skill_row['reuseLevel']
        description = skill_row.get('description', None)
        definition = skill_row.get('definition', None)
        status = skill_row.get('status', None)
        
        # Get immediate parent concepts (groups or other skills)
        parents = skill2parents.get(skill_uri, [])
        
        base_row_data = {
            'skillUri': skill_uri,
            'skillLabel': skill_label,
            'skillType': skill_type,
            'skillAltLabels': skill_alt_labels,
            'reuseLevel': reuse_level,
            'status': status,
            'description': description,
            'definition': definition,
        }
        
        if not parents:
            # Orphan skill - no hierarchy
            row = base_row_data.copy()
            row.update({
                'pillar_uri': None,
                'pillar_label': None,
                'parent_uri': None,
                'parent_label': None,
                'full_path_uris': None,
                'full_path_labels': None,
            })
            rows.append(row)
            continue
        
        # For each immediate parent, get all paths to pillars
        found_paths_for_skill = False
        for parent_concept in parents:
            # === FIX 3 (continued): Use the universal parent map ===
            paths = get_paths_to_pillar(parent_concept, all_concept_parents, pillar_ids)
            
            if not paths:
                # Parent concept has no path to pillar (e.g., row 2618 case)
                row = base_row_data.copy()
                row.update({
                    'pillar_uri': None,
                    'pillar_label': None,
                    # === FIX 4: Use universal label map ===
                    'parent_uri': parent_concept,
                    'parent_label': all_labels_dict.get(parent_concept),
                    'full_path_uris': parent_concept,
                    'full_path_labels': all_labels_dict.get(parent_concept),
                })
                rows.append(row)
                found_paths_for_skill = True
                continue
            
            found_paths_for_skill = True
            # Create a row for each path (handles multiple paths)
            for path in paths:
                # Path is [immediate_parent, parent, ..., pillar]
                # Reverse it to get [pillar, ..., parent, immediate_parent]
                reversed_path = path[::-1]
                
                # === FIX 5 & 6: Use new "lean tree" structure ===
                
                # Get all labels for the path
                path_labels_list = [all_labels_dict.get(uri) for uri in reversed_path if uri]
                uri_path_list = [uri for uri in reversed_path if uri]
                
                row = base_row_data.copy()
                row.update({
                    # 1. Save the Pillar (Top Level)
                    'pillar_uri': uri_path_list[0] if len(uri_path_list) > 0 else None,
                    'pillar_label': path_labels_list[0] if len(path_labels_list) > 0 else None,
                    
                    # 2. Save the Immediate Parent (Bottom Level)
                    'parent_uri': uri_path_list[-1] if len(uri_path_list) > 0 else None,
                    'parent_label': path_labels_list[-1] if len(path_labels_list) > 0 else None,

                    # 3. Save the FULL path as a delimited string
                    'full_path_uris': ' | '.join(uri_path_list),
                    'full_path_labels': ' | '.join(path_labels_list),

                    'level1_uri': uri_path_list[1] if len(uri_path_list) > 1 else None,
                    'level1_label': path_labels_list[1] if len(path_labels_list) > 1 else None,
                    
                    'level2_uri': uri_path_list[2] if len(uri_path_list) > 2 else None,
                    'level2_label': path_labels_list[2] if len(path_labels_list) > 2 else None,

                    'level3_uri': uri_path_list[3] if len(uri_path_list) > 3 else None,
                    'level3_label': path_labels_list[3] if len(path_labels_list) > 3 else None
                })
                
                rows.append(row)
        
        if not found_paths_for_skill:
             # This handles skills that are in skill2parents but whose parents
             # somehow failed all path finding (should be rare)
            row = base_row_data.copy()
            row.update({
                'pillar_uri': None,
                'pillar_label': None,
                'parent_uri': None,
                'parent_label': None,
                'full_path_uris': None,
                'full_path_labels': None,
            })
            rows.append(row)

    # Create DataFrame
    df_complete_hierarchy = pd.DataFrame(rows)
    
    print(f"  ✓ Created complete hierarchy table: {df_complete_hierarchy.shape}")
    print(f"  ✓ Rows with pillar: {df_complete_hierarchy['pillar_label'].notna().sum():,}")
    print(f"  ✓ Orphan rows (no pillar): {df_complete_hierarchy['pillar_label'].isna().sum():,}")
    
    # Show path completeness
    print("\n  Path completeness:")
    count_pillar = df_complete_hierarchy['pillar_label'].notna().sum()
    count_parent = df_complete_hierarchy['parent_label'].notna().sum()
    print(f"    - Rows with pillar: {count_pillar:,}")
    print(f"    - Rows with parent: {count_parent:,}")
    
    return df_complete_hierarchy

def build_complete_hierarchy_w_occ(data):
    """Build dataset merging occupation info with complete skill hierarchy."""
    print("\n" + "="*80)
    print("Building Complete Hierarchy with Occupations Dataset...")
    print("="*80)
    
    # First build the complete hierarchy
    complete_hierarchy = build_complete_hierarchy_table(data)
    
    # Get occupation-skill relationships
    occ_skill_relations = data['occ_skill_relations']
    occupations = data['occupations']
    
    # Prepare occupation details
    occ_details = occupations[[
        'conceptUri', 'preferredLabel', 'iscoGroup', 'description', 'code'
    ]].rename(columns={
        'conceptUri': 'occupationUri',
        'preferredLabel': 'occupationLabel',
        'description': 'occupationDescription',
        'code': 'occupationCode'
    })
    
    # Merge occupation-skill relationships with occupation details
    occ_skill_with_details = occ_skill_relations.merge(
        occ_details, on='occupationUri', how='left'
    )
    
    # Merge with complete hierarchy using skillUri
    occupation_hierarchy = occ_skill_with_details.merge(
        complete_hierarchy, on='skillUri', how='left'
    )
    
    print(f"  ✓ Created complete hierarchy with occupations: {occupation_hierarchy.shape}")
    print(f"  ✓ Unique occupations: {occupation_hierarchy['occupationUri'].nunique():,}")
    print(f"  ✓ Unique skills: {occupation_hierarchy['skillUri'].nunique():,}")
    print(f"  ✓ Essential relationships: {(occupation_hierarchy['relationType'] == 'essential').sum():,}")
    print(f"  ✓ Optional relationships: {(occupation_hierarchy['relationType'] == 'optional').sum():,}")
    print(f"  ✓ Records with full hierarchy: {occupation_hierarchy['pillar_label'].notna().sum():,}")
    
    return occupation_hierarchy

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Build ESCO master datasets")
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        choices=[
            'master_skills', 
            'master_occupation_skills', 
            'master_complete_skills',
            'graph',
            'complete_hierarchy',
            'complete_hierarchy_w_occ',
            'all'
        ],
        default=['all'],
        help='Which datasets to build (default: all)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("ESCO Master Datasets Builder")
    print("="*80)
    print(f"Building datasets: {', '.join(args.datasets)}")
    
    # Create output directory
    PATH_OUT.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data()
    
    # Build datasets
    datasets = {}
    
    # Determine which datasets to build
    build_all = 'all' in args.datasets
    
    if build_all or 'master_skills' in args.datasets:
        datasets['master_skills'] = build_master_skills(data)
    
    if build_all or 'master_occupation_skills' in args.datasets:
        datasets['master_occupation_skills'] = build_master_occupation_skills(data)
    
    if build_all or 'master_complete_skills' in args.datasets:
        datasets['master_complete_skills'] = build_master_complete_skills(data)
    
    if build_all or 'graph' in args.datasets:
        datasets['nodes'], datasets['edges'] = build_graph_datasets(data)
    
    if build_all or 'complete_hierarchy' in args.datasets:
        datasets['complete_hierarchy'] = build_complete_hierarchy_table(data)
    
    if build_all or 'complete_hierarchy_w_occ' in args.datasets:
        datasets['complete_hierarchy_w_occ'] = build_complete_hierarchy_w_occ(data)
    
    # Save datasets
    print("\n" + "="*80)
    print("Saving Master Datasets...")
    print("="*80)
    
    output_files = {
        'master_skills': 'master_skills.csv',
        'master_occupation_skills': 'master_occupation_skills.csv',
        'master_complete_skills': 'master_complete_skills.csv',
        'nodes': 'master_graph_nodes.csv',
        'edges': 'master_graph_edges.csv',
        'complete_hierarchy': 'master_skill_complete_hierarchy.csv',
        'complete_hierarchy_w_occ': 'master_complete_hierarchy_w_occ.csv'
    }
    
    saved_count = 0
    for key, filename in output_files.items():
        if datasets.get(key) is not None:
            filepath = PATH_OUT / filename
            datasets[key].to_csv(filepath, index=False)
            print(f"  ✓ Saved {filename} ({datasets[key].shape[0]:,} rows, {datasets[key].shape[1]} cols)")
            saved_count += 1
        else:
            print(f"  ✗ Skipped {filename} (not requested)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Output directory: {PATH_OUT}")
    print(f"Generated {saved_count} master datasets")
    print("\nNext steps:")
    print("  - Use master_skills.csv for skill-centric analysis")
    print("  - Use master_occupation_skills.csv for occupation-skill matching")
    print("  - Use master_complete_skills.csv for full hierarchy analysis")
    print("  - Use master_skill_complete_hierarchy.csv to see all levels (0-3) + skills in one table")
    print("  - Use master_complete_hierarchy_w_occ.csv for occupation-skill hierarchy analysis")
    print("  - Use master_graph_nodes.csv + master_graph_edges.csv for network analysis")
    print("\n[✓] Build complete!")

if __name__ == "__main__":
    main()

