import os
import numpy as np
import time
from dataloaders.helpers import *
from dataloaders.reporters import *

def run_benchmark(args, method):
	files = get_falling_dataset(args.falling_path)
	evaluate_on(files, method, args)

def evaluate_on(files, method, args):
	dataset_name = os.path.split(os.path.split(os.path.split(files[0])[0])[0])[-1]
	log_folder = os.path.join(args.visualization_path, dataset_name+'_eval/')
	medn = 7
	
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

			start = time.time()
			est_hs, est_traj = method(I,B,bbox_tight,gtp.nsplits)
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

		av_score_tracker.next(gtp.seqname, seq_score_tracker)
		if args.save_visualization:
			logger.close()

	av_score_tracker.close()

