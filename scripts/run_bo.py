import argparse
import sys
import os

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bo_loop import run_bo_loop

def main():
    parser = argparse.ArgumentParser(description="Run Rosaloid BO Pipeline")
    parser.add_argument("--dms_id", type=str, default="GFP_AEQVI_Sarkisyan_2016", 
                        help="DMS ID to run (must match key in data/DMS_substitutions.csv)")
    parser.add_argument("--rounds", type=int, default=5, help="Number of BO rounds")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size per round")
    parser.add_argument("--xi", type=float, default=0.05, help="EI exploration parameter")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Results placed in results/
    output_dir = os.path.join("results")
    
    run_bo_loop(
        dms_id=args.dms_id,
        output_dir=output_dir,
        rounds=args.rounds,
        batch_size=args.batch_size,
        xi=args.xi,
        device=args.device
    )

if __name__ == "__main__":
    main()
