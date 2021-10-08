"""
PetFinder.my - Pawpularity Contest
Kaggle competition
Nick Kaparinos
2021
"""

from utilities import *
import time

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
