import argparse
import numpy as np
import os
import sys

from benchmark.benchmark_loader import *
from benchmark.loaders_helpers import *

sys.path.insert(0, './deblatting_python')
from deblatting_runners import *

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tbd_path", default='/cluster/home/denysr/scratch/dataset/TbD', required=False)
	parser.add_argument("--tbd3d_path", default='/cluster/home/denysr/scratch/dataset/TbD-3D', required=False)
	parser.add_argument("--falling_path", default='/cluster/home/denysr/scratch/dataset/falling_objects', required=False)
	parser.add_argument("--verbose", default=False)
	parser.add_argument("--visualization_path", default='/cluster/home/denysr/tmp', required=False)
	parser.add_argument("--save_visualization", default=False, required=False)
	return parser.parse_args()

def deblur_tbdo(I,B,bbox,nsplits,radius,debl_dim,gt_traj):
	bbox_debl = extend_bbox_uniform(bbox.copy(),0.5*radius,I.shape)
	rgba_tbd3d_or, Hso_crop = deblatting_oracle_runner(crop_only(I,bbox_debl),crop_only(B,bbox_debl),debl_dim,gt_traj[[1,0]]-bbox_debl[:2,None])
	Hso = rev_crop_resize(Hso_crop[:,:,None,:][:,:,[-1,-1,-1],:],bbox_debl,np.zeros(I.shape))
	est_hs = np.zeros(I.shape+(nsplits,))
	for tmki in range(nsplits): 
		if np.sum(Hso[:,:,0,tmki]) > 0: 
			Hsc = Hso[:,:,0,tmki]/np.sum(Hso[:,:,0,tmki])
		else:
			Hsc = Hso[:,:,0,tmki]
		est_hs[:,:,:,tmki] = fmo_model(B,Hsc,rgba_tbd3d_or[:,:,:3,tmki],rgba_tbd3d_or[:,:,3,tmki])
	return est_hs, gt_traj

def deblur_tbd3d(I,B,bbox,nsplits,radius,debl_dim):
	bbox_debl = extend_bbox_uniform(bbox.copy(),0.5*radius,I.shape)
	est_hs_tbd_crop, est_hs_tbd3d_crop, _, _, est_traj_tbd, _ = deblatting_runner(crop_only(I,bbox_debl),crop_only(B,bbox_debl),nsplits,debl_dim)
	est_traj_tbd[0] += bbox_debl[1]
	est_traj_tbd[1] += bbox_debl[0]
	est_hs_tbd3d = rev_crop_resize(est_hs_tbd3d_crop,bbox_debl,I)
	return est_hs_tbd3d, est_traj_tbd

def deblur_tbd(I,B,bbox,nsplits,radius,debl_dim):
	bbox_debl = extend_bbox_uniform(bbox.copy(),0.5*radius,I.shape)
	est_hs_tbd_crop, est_hs_tbd3d_crop, _, _, est_traj_tbd, _ = deblatting_runner(crop_only(I,bbox_debl),crop_only(B,bbox_debl),nsplits,debl_dim)
	est_traj_tbd[0] += bbox_debl[1]
	est_traj_tbd[1] += bbox_debl[0]
	est_hs_tbd = rev_crop_resize(est_hs_tbd_crop,bbox_debl,I)
	return est_hs_tbd, est_traj_tbd

def main():
	args = parse_args()

	args.add_traj = True
	args.method_name = 'TbD-O'
	run_benchmark(args, deblur_tbdo)

	args.add_traj = False
	args.method_name = 'TbD-3D'
	run_benchmark(args, deblur_tbd3d)

	args.add_traj = False
	args.method_name = 'TbD'
	run_benchmark(args, deblur_tbd)
	

if __name__ == "__main__":
    main()