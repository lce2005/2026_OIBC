"""Entry point for v0.4 xgb; use --data to select input."""
import sys
from train_tabular_v3 import main
if __name__=='__main__':
    sys.argv[1:1] = ['--model', 'xgb']
    main()
