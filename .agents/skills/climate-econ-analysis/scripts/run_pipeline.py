import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*50}\nRunning: {script_path}\n{'='*50}")
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"Error executing {script_path}. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    # Up 4 levels from .agents/skills/climate-econ-analysis/scripts/run_pipeline.py to get to .agents
    agents_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # Up one more level to get to the project root (climate-debt-analysis)
    project_root = os.path.dirname(agents_dir)
    
    scripts_to_run = [
        # Stage 1
        os.path.join(agents_dir, 'skills', '01-individual-eda', 'scripts', 'dataset_a_rainfall.py'),
        os.path.join(agents_dir, 'skills', '01-individual-eda', 'scripts', 'dataset_b_food.py'),
        os.path.join(agents_dir, 'skills', '01-individual-eda', 'scripts', 'dataset_c_debt.py'),
        
        # Stage 2
        os.path.join(agents_dir, 'skills', '02-pairwise-alignment', 'scripts', 'a_b_rainfall_food.py'),
        os.path.join(agents_dir, 'skills', '02-pairwise-alignment', 'scripts', 'b_c_food_debt.py'),
        os.path.join(agents_dir, 'skills', '02-pairwise-alignment', 'scripts', 'a_c_rainfall_debt.py'),
        
        # Stage 3
        os.path.join(agents_dir, 'skills', '03-joint-mediation', 'scripts', 'joint_mediation.py')
    ]
    
    print("Starting full Climate-Econ Analysis Pipeline...")
    
    # Run from the project root so csv paths work
    os.chdir(project_root)
    
    for script in scripts_to_run:
        run_script(script)
        
    print("\n" + "="*50)
    print("Pipeline Execution Complete!")
    print("All datasets, models, and artifacts have been saved to the 'production_artifacts/' directory.")
    print("="*50)
