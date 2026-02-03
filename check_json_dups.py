import json
from collections import Counter

file_path = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json'

def check_duplicates(path, sample_size=1000):
    print(f"Checking for duplicates in: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Extract scores dictionary
    if 'scores' in data:
        scores_dict = data['scores']
    else:
        scores_dict = data
    
    job_ids = list(scores_dict.keys())
    print(f"Total job_ids: {len(job_ids)}")
    
    jobs_with_dups = 0
    total_dups_found = 0
    
    # Check a sample or all
    to_check = job_ids[:sample_size] if sample_size else job_ids
    
    for job_id in to_check:
        raw_skill_list = scores_dict[job_id]
        
        # Extract URIs
        uris = []
        for item in raw_skill_list:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                uris.append(item[0])
            elif isinstance(item, dict):
                uri = item.get('skill_uri') or item.get('skillUri')
                if uri:
                    uris.append(uri)
        
        # Check for duplicates
        counts = Counter(uris)
        dups = {uri: count for uri, count in counts.items() if count > 1}
        
        if dups:
            jobs_with_dups += 1
            total_dups_found += sum(dups.values()) - len(dups)
            if jobs_with_dups <= 3:
                print(f"Job ID {job_id} has duplicates: {dups}")
    
    print(f"\nResults (sampled {len(to_check)} jobs):")
    print(f"Jobs with duplicate URIs: {jobs_with_dups}")
    print(f"Percentage of jobs with dups: {100 * jobs_with_dups / len(to_check):.2f}%")
    if jobs_with_dups > 0:
        print(f"Total redundant entries found in these jobs: {total_dups_found}")

check_duplicates(file_path, sample_size=None) # Check all since it's only 128MB
