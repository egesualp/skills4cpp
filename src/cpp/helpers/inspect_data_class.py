
import sys
import os
import random

# Add project root to path (assuming script is run from project root or src/cpp)
# We try to find the 'skills4cpp' root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add the directory containing 'src' if needed, depending on how imports are structured
# Usually imports are like 'from src.cpp.data_classes import Data'
if os.path.dirname(current_dir) not in sys.path:
     sys.path.insert(0, os.path.dirname(current_dir))

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback for running directly from src/cpp without package structure
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from data_classes import Data
        from utils import SEP_TOKEN
    except ImportError as e2:
        print(f"Failed to import Data class: {e2}")
        sys.exit(1)

def print_separator(char="=", length=80):
    print(char * length)

def inspect_data_class():
    data_type = "karrierewege_100k"
    print_separator()
    print(f"Initializing Data class with DATA_TYPE='{data_type}'...")
    
    # Initialize Data
    # Note: You can change ONLY_TITLES to True to see the difference
    data = Data(DATA_TYPE=data_type, ONLY_TITLES=False)
    
    print("Data class initialized successfully.")
    print_separator()
    
    # Inspect Attributes
    print("Inspecting Attributes:")
    attributes = [
        "DATA_TYPE", "DOC_1_PROMPT", "DOC_2_PROMPT", "ONLY_TITLES",
        "labels", "train_pairs", "val_pairs", "test_pairs"
    ]
    
    for attr in attributes:
        if hasattr(data, attr):
            value = getattr(data, attr)
            if isinstance(value, list):
                print(f"  - {attr}: List with {len(value)} elements")
            else:
                print(f"  - {attr}: {value}")
        else:
            print(f"  - {attr}: Not found")
            
    print_separator()
    
    # Inspect Stages
    stages = ['embedding_finetuning', 'transformation_finetuning', 'evaluation']
    
    for stage in stages:
        print(f"\nINSPECTING STAGE: {stage}")
        print_separator("-")
        
        try:
            # We call get_data. Note: Depending on your recent changes, it might support 'include_clean_test'
            # We will try calling it with standard arguments first
            
            # Check if get_data supports include_clean_test (introspection or try/except)
            # For this inspection script, we'll stick to standard behavior unless we want to test the new feature
            
            results = data.get_data(stage=stage)
            
            # Unpack based on result length
            if len(results) == 3:
                train, val, test = results
                test_clean = None
                print("  -> get_data returned 3 splits (Train, Val, Test)")
            elif len(results) == 4:
                train, val, test, test_clean = results
                print("  -> get_data returned 4 splits (Train, Val, Test, TestClean)")
            else:
                print(f"  -> Unexpected number of return values: {len(results)}")
                continue
                
            # Print samples from Train
            print(f"\n  Samples from TRAIN split ({len(train)} pairs):")
            for i in range(min(5, len(train))):
                doc1, doc2 = train[i]
                print(f"    Sample {i+1}:")
                print(f"      DOC_1 (History): {repr(doc1)}")
                print(f"      DOC_2 (Target):  {repr(doc2)}")
                if i < 4: print("      ---")
            
            # Check if DOC_1 changed due to stage logic (e.g. minus_last)
            # We can compare with raw self.train_pairs
            raw_doc1, raw_doc2 = data.train_pairs[0]
            stage_doc1, stage_doc2 = train[0]
            
            print(f"\n  Comparison for Sample 1 (Raw vs Stage):")
            print(f"    Raw DOC_1:   {repr(raw_doc1)}")
            print(f"    Stage DOC_1: {repr(stage_doc1)}")
            
            if raw_doc1 != stage_doc1:
                print("    >> DIFFERENCE DETECTED: Stage logic modified DOC_1 (likely __minus_last)")
            else:
                print("    >> NO DIFFERENCE: DOC_1 preserved as-is")

        except Exception as e:
            print(f"Error inspecting stage {stage}: {e}")

    print_separator()
    print("Done.")

if __name__ == "__main__":
    inspect_data_class()


