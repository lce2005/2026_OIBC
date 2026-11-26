"""Entry point for v0.4 transformer; see README."""
import sys
from train_sequence_v3 import main
if __name__=='__main__':
    sys.argv[1:1] = ['--model','transformer']
    main()
