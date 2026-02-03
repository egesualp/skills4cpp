
import sys
import os
from pathlib import Path

# Add src to sys.path to allow imports from cpp package
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from cpp.data_classes import Data
from collections import defaultdict
import re

def verify_replacements():
    print("Initializing Data(DATA_TYPE='decorte_esco')...")
    # minimal init
    data_module = Data(DATA_TYPE='decorte_esco', ONLY_TITLES=True, consider_subspans=False)
    
    # keys are original, values are replaced
    replacements_title = {
        'ICT security engineer': 'cyber incident responder',
        'ict security engineer': 'cyber incident responder',
        'care at home worker': 'care home worker',
        'residential care home worker': 'care home worker',
        'ICT security manager': 'cybersecurity risk manager',
        'ict security manager': 'cybersecurity risk manager',
        'care at hmoe worker': 'care home worker',
        'handyman': 'handyperson',
        'corporate banking manager': 'corporate banking adviser',
    }

    # Normalize keys for searching in text (lower case as utils does strip().lower())
    # But wait, utils does:
    # processed_title = original_title.strip().lower()
    # final_title = replacements_title.get(processed_title, processed_title)
    # So the replacement mapping relies on the lower-cased version of the input using the dictionary keys.
    # The dictionary keys in utils.py are mixed case: 'ICT security engineer' vs 'ict security engineer'.
    # Actually, let's look at utils.py again.
    # processed_title = original_title.strip().lower()
    # final_title = replacements_title.get(processed_title, processed_title)
    #
    # If original is 'ICT security engineer', processed is 'ict security engineer'.
    # The dict has 'ict security engineer', so it maps to 'cyber incident responder'.
    # If original is 'residential care home worker', processed is 'residential care home worker'.
    # The dict has 'residential care home worker', so it maps.
    
    # We want to check that the OUTPUT text contains the VALUES, and NOT the Keys (specifically the lower-cased keys).
    
    target_terms = set(replacements_title.values())
    avoid_terms = set(k.lower() for k in replacements_title.keys())
    
    # Remove terms from avoid_terms if they are also target terms (unlikely here, but good practice)
    avoid_terms = avoid_terms - target_terms

    print(f"Checking for presence of replacements: {target_terms}")
    print(f"Checking for absence of original terms: {avoid_terms}")
    
    # We'll check the 'train' split
    # data_module.train_pairs is a list of (doc1, doc2) tuples
    # where doc resembles: "esco role: <title> \n description: ..."
    
    found_targets = defaultdict(int)
    found_avoids = defaultdict(int)
    
    print(f"Scanning {len(data_module.train_pairs)} training pairs...")
    
    for doc1, doc2 in data_module.train_pairs:
        # Check doc1 and doc2 content
        content = (doc1 + " " + doc2).lower()
        
        for term in target_terms:
            if term in content:
                found_targets[term] += 1
                
        for term in avoid_terms:
            if term in content:
                # We need to be careful about substrings, but these follow 'esco role: '
                # Let's check strictly for "role: term " or similar if loosely checking fails
                if f"role: {term}" in content or f"role: {term}\n" in content:
                     found_avoids[term] += 1

    print("\n--- Verification Results ---")
    print("Replacements found (Counts):")
    for term, count in found_targets.items():
        print(f"  '{term}': {count}")
        
    print("\nOriginal terms found (Should be 0 or very low if replacements worked):")
    if not found_avoids:
        print("  NONE! (Success)")
    else:
        for term, count in found_avoids.items():
            print(f"  '{term}': {count}")
    
    if len(found_targets) > 0 and len(found_avoids) == 0:
        print("\nSUCCESS: Replacements appear to be applied correctly.")
    elif len(found_targets) == 0:
        print("\nWARNING: No replacement terms found. Either the dataset doesn't contain these examples or something is wrong.")
    else:
        print("\nFAILURE: Found original terms in the output.")

if __name__ == "__main__":
    verify_replacements()
