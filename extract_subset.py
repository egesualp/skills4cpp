
import json
import os

def get_first_n_jobs(file_path, n=10):
    jobs = {}
    with open(file_path, 'r') as f:
        # Skip the opening '{'
        line = f.readline()
        if '{' not in line:
             # Maybe it's not pretty printed? 
             # Let's just try to load the first few lines and see.
             pass
        
        current_job_id = None
        current_data = ""
        in_job = False
        jobs_found = 0
        
        while jobs_found < n:
            line = f.readline()
            if not line: break
            
            stripped = line.strip()
            if not stripped: continue
            
            # Look for "ID": [
            if not in_job:
                if '": [' in stripped or '": {' in stripped:
                    current_job_id = stripped.split('"')[1]
                    current_data = stripped[stripped.find(':')+1:]
                    in_job = True
                    # Check if it's a single-line job
                    if (']' in stripped and '[' in stripped) or ('}' in stripped and '{' in stripped):
                        try:
                            jobs[current_job_id] = json.loads(current_data.strip().rstrip(','))
                            jobs_found += 1
                            in_job = False
                        except: pass
            else:
                current_data += line
                if stripped.startswith(']') or stripped.startswith('}'):
                    # might have a trailing comma
                    try:
                        clean_data = current_data.strip().rstrip(',')
                        jobs[current_job_id] = json.loads(clean_data)
                        jobs_found += 1
                        in_job = False
                    except:
                        pass
    return jobs

if __name__ == "__main__":
    task_b_path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json"
    print("Extracting 10 jobs...")
    data = get_first_n_jobs(task_b_path, 10)
    with open("subset_task_b.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Extracted {len(data)} jobs to subset_task_b.json")
    print("Job IDs found:", list(data.keys()))
