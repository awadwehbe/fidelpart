import os
import re

# Directory to search
root_dir = "."

# Patterns to match and their replacements
patterns = [
    r"from bqskit\.utils\.math import",  # e.g., from bqskit.utils.math import softmax
    r"from bqskit\.utils import math\b", # e.g., from bqskit.utils import math
    r"import bqskit\.utils\.math\b"     # e.g., import bqskit.utils.math
]
replacements = [
    "from bqskit.utils.math_utils import",
    "from bqskit.utils import math_utils",
    "import bqskit.utils.math_utils"
]

# Files to skip
skip_files = {"update_imports.py"}

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    modified = False
    new_content = content
    for pattern, replacement in zip(patterns, replacements):
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, new_content)
            modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")

# Walk through all Python files
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.py') and filename not in skip_files:
            filepath = os.path.join(dirpath, filename)
            update_file(filepath)

print("Import update complete!")