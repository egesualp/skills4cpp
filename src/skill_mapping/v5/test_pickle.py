
import pickle
import sqlite3
from pathlib import Path
from fused_scorer_opt import TaskBManager
import tempfile
import os
from joblib import Parallel, delayed

def test_pickle():
    # Create a dummy json file
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
        f.write('{"job1": {"skill1": 1.0}}')
        json_path = Path(f.name)
    
    try:
        manager = TaskBManager(json_path, cache_dir=json_path.parent)
        print("Manager initialized. Connection is:", manager.conn)
        
        def func(m):
            return m.conn is not None

        # Try joblib
        try:
            print("Testing joblib Parallel...")
            results = Parallel(n_jobs=2)(delayed(func)(manager) for _ in range(2))
            print("Joblib result:", results)
            
        except Exception as e:
            print("Joblib failed:", e)
            import traceback
            traceback.print_exc()
            
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)
            
if __name__ == "__main__":
    test_pickle()
