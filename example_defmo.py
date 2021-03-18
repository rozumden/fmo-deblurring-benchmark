import os
import torch

from benchmark.benchmark_loader import *
from benchmark.loaders_helpers import *
import argparse

import sys
sys.path.insert(0, './DeFMO')
from models.encoder import *
from models.rendering import *
from dataloaders.loader import get_transform
from helpers.torch_helpers import renders2traj

g_saved_models_folder = './DeFMO/saved_models/'

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
	g_resolution_x = int(640/2)
	g_resolution_y = int(480/2)
	multi_f = 5 ## simulate small motion blur

	gpu_id = 0
	device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
	print(device)
	torch.backends.cudnn.benchmark = True
	encoder = EncoderCNN()
	rendering = RenderingCNN()

	if torch.cuda.is_available():
		encoder.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'encoder_best.pt')))
		rendering.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'rendering_best.pt')))
	else:
		encoder.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'encoder_best.pt'),map_location=torch.device('cpu')))
		rendering.load_state_dict(torch.load(os.path.join(g_saved_models_folder, 'rendering_best.pt'),map_location=torch.device('cpu')))
		
	encoder = encoder.to(device)
	rendering = rendering.to(device)

	encoder.train(False)
	rendering.train(False)

	def deblur_defmo(I,B,bbox_tight,nsplits,radius):
		bbox = extend_bbox(bbox_tight.copy(),4*np.max(radius),g_resolution_y/g_resolution_x,I.shape)
		im_crop = crop_resize(I, bbox, (g_resolution_x, g_resolution_y))
		bgr_crop = crop_resize(B, bbox, (g_resolution_x, g_resolution_y))
		preprocess = get_transform()
		input_batch = torch.cat((preprocess(im_crop), preprocess(bgr_crop)), 0).to(device).unsqueeze(0).float()
		with torch.no_grad():
			latent = encoder(input_batch)
			times = torch.linspace(0,1,nsplits*multi_f+1).to(device)
			renders = rendering(latent,times[None])
			renders = renders[:,:-1].reshape(1, nsplits, multi_f, 4, g_resolution_y, g_resolution_x).mean(2) # add small motion blur 

		renders_rgba = renders[0].data.cpu().detach().numpy().transpose(2,3,1,0)
		est_hs_crop = rgba2hs(renders_rgba, bgr_crop)
		est_hs = rev_crop_resize(est_hs_crop,bbox,I)
		est_traj = renders2traj(renders,device)[0].T.cpu()
		est_traj = rev_crop_resize_traj(est_traj, bbox, (g_resolution_x, g_resolution_y))
		return est_hs, est_traj

	args.add_traj = False	
	args.method_name = 'DeFMO'
	run_benchmark(args, deblur_defmo)


if __name__ == "__main__":
    main()