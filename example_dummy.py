import numpy as np
import os

from dataloaders.loader import *
from dataloaders.tbd_loader import *

def main():
	
	datasets = ['tbd','tbd3d','tbdfalling']
	evaluate_on(encoder, rendering, device, datasets[2])



if __name__ == "__main__":
    main()