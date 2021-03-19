import numpy as np
import os, glob
import scipy.io
import cv2
from benchmark.loaders_helpers import *

class GroundTruthProcessor:
	def __init__(self, seqpath, kkf, medn):
		seqname = os.path.split(seqpath)[-1]
		folder = os.path.split(os.path.split(seqpath)[0])[0]

		roi_frames = []
		if os.path.exists(os.path.join(folder,'roi_frames.txt')):
			roi_frames = np.loadtxt(os.path.join(folder,'roi_frames.txt')).astype(int)
		
		if '-12' in seqname:
			self.nsplits = 12
		else:
			self.nsplits = 8

		nfrms = len(glob.glob(os.path.join(seqpath,'*.png')))
		start_ind = 0
		end_ind = nfrms
		if roi_frames != []:
			start_ind = roi_frames[kkf,0]
			end_ind = roi_frames[kkf,1]
			nfrms = end_ind - start_ind + 1
		if not os.path.exists(os.path.join(seqpath,"{:08d}.png".format(0))):
			start_ind += 1
		mednused = np.min([medn, nfrms])
		Vk = []
		for kk in range(mednused):
			Im = imread(os.path.join(seqpath,"{:08d}.png".format(start_ind+kk)))
			if Vk == []:
				Vk = np.zeros(Im.shape+(mednused,))
			Vk[:,:,:,kk] = Im
		pars = []
		w_trajgt = False
		if os.path.exists(os.path.join(seqpath,'gt.txt')):
			w_trajgt = True
			pars = np.loadtxt(os.path.join(seqpath,'gt.txt'))
		rads = []
		if os.path.exists(os.path.join(seqpath,'gtr.txt')):
			rads = np.loadtxt(os.path.join(seqpath,'gtr.txt'))
		elif os.path.exists(os.path.join(folder,'templates')):
			template_path = os.path.join(folder,'templates',seqname + '_template.mat')
			data = scipy.io.loadmat(template_path)
			template = data['template']
			rads = (template.shape[0]/2)/data['scale']
		if not w_trajgt and os.path.exists(os.path.join(folder,'gt_bbox',seqname + '.txt')):
			w_trajgt = True
			bboxes = np.loadtxt(os.path.join(folder,'gt_bbox',seqname + '.txt'))
			pars = np.reshape(bboxes[:,:2] + 0.5*bboxes[:,2:], (-1,self.nsplits,2)).transpose((0,2,1))
			pars = np.reshape(pars,(-1,self.nsplits))
			rads = np.reshape(np.max(0.5*bboxes[:,2:],1), (-1,self.nsplits))
			pars = np.r_[np.zeros((start_ind*2,self.nsplits)),pars]
			rads = np.r_[np.zeros((start_ind,self.nsplits)),rads]
		self.hspath_base = os.path.join(folder,'imgs_gt',seqname)
		self.use_hs = os.path.exists(os.path.join(self.hspath_base,"{:08d}.png".format(1)))
		self.start_zero = 1-int(os.path.exists(os.path.join(self.hspath_base,"{:08d}.png".format(0))))
		self.pars = pars
		self.rads = rads
		self.mednused = mednused
		self.start_ind = start_ind
		self.nfrms = nfrms
		self.Vk = Vk
		self.seqname = seqname
		self.seqpath = seqpath
		self.w_trajgt = w_trajgt
		print('Sequence {} has {} frames'.format(seqname, nfrms))

	def get_img_noupd(self, kk):
		path = os.path.join(self.seqpath, "{:08d}.png".format(self.start_ind+kk))
		I = imread(path)
		return I

	def get_img(self, kk):
		I = self.get_img_noupd(kk)
		B = np.median(self.Vk, 3)
		if kk >= self.mednused:
			self.Vk[:,:,:,:-1] = self.Vk[:,:,:,1:]
			self.Vk[:,:,:,-1] = I
		return I,B

	def get_hs(self, kk):
		Vos = np.zeros((self.Vk.shape[0], self.Vk.shape[1], 3, self.nsplits))
		for hsk in range(self.nsplits):
			hspath = os.path.join(self.hspath_base, "{:08d}.png".format((kk+self.start_ind)*self.nsplits + hsk + self.start_zero))
			Vo = imread(hspath)
			Vos[:,:,:,hsk] = Vo[:self.Vk.shape[0],:self.Vk.shape[1]]
		return Vos

	def get_trajgt(self, kk):
		par = self.pars[2*(kk+self.start_ind):2*(kk+self.start_ind+1),:].T
		self.nsplits = par.shape[0]
		parsum = par.sum(1)
		nans = np.isnan(parsum)
		if nans.sum() > 0:
			ind = np.nonzero(nans)[0]
			for indt in ind:
				if indt == 0:
					par[indt,:] = par[np.nonzero(~nans)[0][0],:]
				elif indt < self.nsplits-1 and not nans[indt+1]:
					par[indt,:] = (par[indt-1,:] + par[indt+1,:])/2
				else:
					par[indt,:] = par[indt-1,:]

		bbox = (par[:,1].min(), par[:,0].min(), par[:,1].max(), par[:,0].max())
		if self.rads.shape[0] > 1:
			radius = np.round(np.nanmax(self.rads[self.start_ind+kk,:])).astype(int)
		else:
			radius = np.round(self.rads[0,0]).astype(int)
		bbox = np.array(bbox).astype(int)
		return par.T, radius, bbox


#######################################################################################################################
#######################################################################################################################

class AverageScoreTracker:
	def __init__(self, nfiles, algname):
		self.av_ious = np.zeros(nfiles)
		self.av_psnr = np.zeros(nfiles)
		self.av_ssim = np.zeros(nfiles)
		self.av_times = []
		self.seqi = 0
		self.algname = algname

	def next(self, seqname, sst):
		self.av_ious[self.seqi] = np.mean(sst.all_ious)
		self.av_psnr[self.seqi] = np.mean(sst.all_psnr)
		self.av_ssim[self.seqi] = np.mean(sst.all_ssim)
		print('{}: Finished seq {}, avg. TIoU {:.3f}, PSNR {:.3f} dB, SSIM {:.3f}'.format(self.algname,seqname, self.av_ious[self.seqi], self.av_psnr[self.seqi], self.av_ssim[self.seqi]))
		self.seqi += 1

	def next_time(self, tm):
		self.av_times.append(tm)

	def close(self):
		print('AVERAGES')
		print('{}: TIoU {:.3f}, PSNR {:.3f} dB, SSIM {:.3f}'.format(self.algname, np.nanmean(self.av_ious), np.nanmean(self.av_psnr), np.nanmean(self.av_ssim)))
		print('{}: time {:.3f} seconds'.format(self.algname, np.nanmean(np.array(self.av_times))))

#######################################################################################################################
#######################################################################################################################

class SequenceScoreTracker:
	def __init__(self, nfrms, algname):
		self.all_ious = np.zeros(nfrms)
		self.all_psnr = np.zeros(nfrms)
		self.all_ssim = np.zeros(nfrms)
		self.algname = algname

	def next_traj(self,kk,gt_traj,est_traj,minor_axis_length):
		ious = calciou(gt_traj, est_traj, minor_axis_length)
		ious2 = calciou(gt_traj, est_traj[:,-1::-1], minor_axis_length)
		iou = np.max([np.mean(ious), np.mean(ious2)])
		self.all_ious[kk] = iou
		return iou

	def next_appearance(self,kk,gt_hs,est_hs):
		self.all_psnr[kk] = calculate_psnr(gt_hs, est_hs)
		self.all_ssim[kk] = calculate_ssim(gt_hs, est_hs)

	def report(self, seqname, kk):
		print('{}: Seq {}, frm {}, TIoU {:.3f}, PSNR {:.3f} dB, SSIM {:.3f}'.format(self.algname, seqname, kk, self.all_ious[kk], self.all_psnr[kk], self.all_ssim[kk]))

#######################################################################################################################
#######################################################################################################################

class SequenceLogger:
	def __init__(self, log_folder, gtp, algname=''):
		self.writepath = os.path.join(log_folder,gtp.seqname)
		if not os.path.exists(self.writepath):
			os.makedirs(self.writepath)
		self.ImGT = gtp.Vk[:,:,:,0].copy()
		self.ImEst = gtp.Vk[:,:,:,0].copy()
		self.nsplits = gtp.nsplits
		self.save_superres = True
		self.algname = algname
		if self.save_superres:
			self.srwriter = SRWriter(self.ImGT, os.path.join(self.writepath, self.algname+'sr.avi'), gtp.use_hs)

	def write_trajgt(self, gt_traj):
		write_trajectory(self.ImGT, gt_traj)
		imwrite(self.ImGT, os.path.join(self.writepath, 'imgt.png'))

	def write_trajest(self, est_traj):
		write_trajectory(self.ImEst, est_traj)
		imwrite(self.ImEst,os.path.join(self.writepath, self.algname + 'imest.png'))

	def write_crops(self,kk,renders_rgb, est_hs_crop, gt_hs_crop, im_crop):
		videoname = '{:04d}video{}.avi'.format(kk,self.algname)
		shp = (im_crop.shape[0]*2, im_crop.shape[1]*2, 3)
		imw = np.zeros(shp)
		imw[im_crop.shape[0]:,im_crop.shape[1]:] = im_crop
		out = cv2.VideoWriter(os.path.join(self.writepath, videoname),cv2.VideoWriter_fourcc(*"MJPG"), 6, (shp[1], shp[0]), True)
		for ki in range(est_hs_crop.shape[3]):
			imw[:im_crop.shape[0],:im_crop.shape[1]] = est_hs_crop[:,:,:,ki]
			imw[im_crop.shape[0]:,:im_crop.shape[1]] = renders_rgb[:,:,:,ki]
			if gt_hs_crop is not None:
				imw[:im_crop.shape[0],im_crop.shape[1]:] = gt_hs_crop[:,:,:,ki]
			imw[imw>1]=1
			imw[imw<0]=0
			out.write( (imw[:,:,[2,1,0]] * 255).astype(np.uint8) )
		out.release()

	def write_crops_3c(self,kk, est_hs_crop, gt_hs_crop, im_crop):
		videoname = '{:04d}video{}_3c.avi'.format(kk,self.algname)
		if gt_hs_crop is None:
			fctr = 2
		else:
			fctr = 3
		shp = (im_crop.shape[0], im_crop.shape[1]*fctr, 3)
		imw = np.zeros(shp)
		imw[:,:im_crop.shape[1]] = im_crop
		out = cv2.VideoWriter(os.path.join(self.writepath, videoname),cv2.VideoWriter_fourcc(*"MJPG"), 6, (shp[1], shp[0]), True)
		for ki in range(est_hs_crop.shape[3]):
			imw[:,im_crop.shape[1]:2*im_crop.shape[1]] = est_hs_crop[:,:,:,ki]
			if not gt_hs_crop is None:
				imw[:,2*im_crop.shape[1]:3*im_crop.shape[1]] = gt_hs_crop[:,:,:,ki]
			imw[imw>1]=1
			imw[imw<0]=0
			out.write( (imw[:,:,[2,1,0]] * 255).astype(np.uint8) )
		out.release()

	def write_superres(self, I, est_hs, Vos):
		if self.save_superres:
			self.srwriter.update_ls(I)
			for hsk in range(self.nsplits):
				vosframe = None
				if Vos is not None:
					vosframe = Vos[:,:,:,hsk]
				if est_hs is None:
					self.srwriter.write_next(vosframe,I) 
				else:
					self.srwriter.write_next(vosframe,est_hs[:,:,:,hsk]) 

	def close(self):
		if self.save_superres:
			self.srwriter.close()


class SRWriter:
	def __init__(self, imtemp, path, available_gt=True):
		self.available_gt = available_gt
		if self.available_gt:
			fctr = 3
		else:
			fctr = 2
		if imtemp.shape[0] > imtemp.shape[1]:
			self.width = True
			shp = (imtemp.shape[0], imtemp.shape[1]*fctr, 3)
			self.value = imtemp.shape[1]
		else:
			self.width = False
			shp = (imtemp.shape[0]*fctr, imtemp.shape[1], 3)
			self.value = imtemp.shape[0]
		self.video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"MJPG"), 12, (shp[1], shp[0]), True)
		self.img = np.zeros(shp)

	def update_ls(self,lsf):
		if self.width:
			self.img[:,:self.value] = lsf
		else:
			self.img[:self.value,:] = lsf

	def write_next(self,hs,est):
		if hs is not None:
			if self.width:
				self.img[:,2*self.value:] = hs
			else:
				self.img[2*self.value:,:] = hs
		if est is not None:
			if self.width:
				self.img[:,self.value:2*self.value] = est
			else:
				self.img[self.value:2*self.value,:] = est
		self.img[self.img>1]=1
		self.img[self.img<0]=0
		self.video.write( (self.img.copy() * 255)[:,:,[2,1,0]].astype(np.uint8) )

	def close(self):
		self.video.release()