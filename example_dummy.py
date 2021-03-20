import argparse
import numpy as np
import os

from benchmark.benchmark_loader import *

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tbd_path", default='/cluster/home/denysr/scratch/dataset/TbD', required=False)
	parser.add_argument("--tbd3d_path", default='/cluster/home/denysr/scratch/dataset/TbD-3D', required=False)
	parser.add_argument("--falling_path", default='/cluster/home/denysr/scratch/dataset/falling_objects', required=False)
	parser.add_argument("--verbose", default=False)
	parser.add_argument("--visualization_path", default='/cluster/home/denysr/tmp', required=False)
	parser.add_argument("--save_visualization", default=False, required=False)
	return parser.parse_args()

def main():
	args = parse_args()
	args.add_traj = False
	
	baseline_im = lambda I,B,bbox,nsplits,radius,obj_dim: (np.repeat(I[:,:,:,None], nsplits, 3), None)
	args.method_name = 'Im'
	run_benchmark(args, baseline_im)

	baseline_bgr = lambda I,B,bbox,nsplits,radius,obj_dim: (np.repeat(B[:,:,:,None], nsplits, 3), None)
	args.method_name = 'Bgr'
	run_benchmark(args, baseline_bgr)

if __name__ == "__main__":
    main()