import os
import numpy as np
import time
from .loaders_helpers import *
from .reporters import *

def run_benchmark(args, method):
	files = get_falling_dataset(args.falling_path)
	evaluate_on(files, method, args)
	files = get_tbd3d_dataset(args.tbd3d_path)
	evaluate_on(files, method, args)
	files = get_tbd_dataset(args.tbd_path)
	evaluate_on(files, method, args)
	
def evaluate_on(files, method, args, callback=None):
	dataset_name = os.path.split(os.path.split(os.path.split(files[0])[0])[0])[-1]
	log_folder = os.path.join(args.visualization_path, dataset_name+'_eval/')
	medn = 50
	
	av_score_tracker = AverageScoreTracker(files.shape, args.method_name)
	
	for kkf, ff in enumerate(files):
		gtp = GroundTruthProcessor(ff,kkf,medn)
		if args.save_visualization:
			logger = SequenceLogger(log_folder, gtp, args.method_name)
		
		seq_score_tracker = SequenceScoreTracker(gtp.nfrms, args.method_name)
		for kk in range(gtp.nfrms):
			gt_traj, radius, bbox = gtp.get_trajgt(kk)
			I, B = gtp.get_img(kk)
			gt_hs = gtp.get_hs(kk)

			bbox = extend_bbox_uniform(bbox,radius,I.shape)
			bbox_tight = bbox_fmo(extend_bbox_uniform(bbox.copy(),10,I.shape),gt_hs,B)
			obj_dim = [0,0]
			for timei in range(gt_hs.shape[3]):
				bbox_temp = bbox_detect_hs(crop_only(gt_hs[:,:,:,timei],bbox_tight), crop_only(B,bbox_tight))
				if len(bbox_temp) == 0:
					bbox_temp = bbox_tight
				obj_dim_temp = bbox_temp[2:] - bbox_temp[:2]
				obj_dim[0] = max(obj_dim[0],obj_dim_temp[0])
				obj_dim[1] = max(obj_dim[1],obj_dim_temp[1])

			start = time.time()
			if args.add_traj:
				est_hs, est_traj = method(I,B,bbox_tight,gtp.nsplits,radius,obj_dim,gt_traj)
			else:
				est_hs, est_traj = method(I,B,bbox_tight,gtp.nsplits,radius,obj_dim)

			if est_hs is None:
				if args.verbose:
					print(f"method() returned None, skipping frame {kk}")
				continue

			av_score_tracker.next_time(time.time() - start)

			gt_hs_crop = crop_only(gt_hs, bbox_tight)
			est_hs_crop = crop_only(est_hs, bbox_tight)

			est_hs_crop, do_flip = sync_directions(est_hs_crop, gt_hs_crop)

			if not est_traj is None:
				iou = seq_score_tracker.next_traj(kk,gt_traj,est_traj,radius)
			seq_score_tracker.next_appearance(kk,gt_hs_crop,est_hs_crop)
			
			if args.save_visualization:
				if not est_traj is None:
					logger.write_trajest(est_traj)
				logger.write_trajgt(gt_traj)
				logger.write_superres(I,est_hs,gt_hs)
				logger.write_crops_3c(kk, est_hs_crop, gt_hs_crop, crop_only(I,bbox_tight))
					
			if args.verbose:
				seq_score_tracker.report(gtp.seqname, kk)

		means = seq_score_tracker.close()
		av_score_tracker.next(gtp.seqname, means)
		if args.save_visualization:
			logger.close()

		if callback:
			callback(kkf, means)

	return av_score_tracker.close()

