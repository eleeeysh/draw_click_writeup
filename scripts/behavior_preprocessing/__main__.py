import os
import subprocess

# List of Python scripts to run sequentially
cur_folder = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "prescan.py",
    "preprocess_behavior.py",
]
scripts = [os.path.join(cur_folder, script) for script in scripts]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)  # Print the output of the script
    if result.returncode != 0:
        print(f"Error in {script}: {result.stderr}")
        break  # Stop execution if a script fails
