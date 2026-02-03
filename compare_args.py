import re

def get_args(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    # Find all parser.add_argument("--name" ...)
    matches = re.findall(r'parser\.add_argument\(\s*["\']--([^"\']+)["\']', content)
    return set(matches)

v2_args = get_args('/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/src/cpp/train_cpp_enhanced_v2.py')
v3_args = get_args('/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/src/cpp/train_cpp_enhanced_v3.py')

print(f"V3 only: {sorted(v3_args - v2_args)}")
print(f"V2 only: {sorted(v2_args - v3_args)}")
