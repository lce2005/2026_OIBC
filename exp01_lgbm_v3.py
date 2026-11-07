"""Entry point for v0.4 lgbm; use --data to select input."""
import sys
from train_tabular_v3 import main
if __name__=='__main__':
    sys.argv[1:1] = ['--model', 'lgbm']
    main()
