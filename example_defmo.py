import numpy as np
import pdb
import os
import torch
from torchvision.utils import save_image

import torch

from dataloaders.loader import *
from dataloaders.tbd_loader import *
from helpers.torch_helpers import *

g_saved_models_folder = './saved_models/'

def main():
	print(torch.__version__)

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

	encoder_params = sum(p.numel() for p in encoder.parameters())
	rendering_params = sum(p.numel() for p in rendering.parameters())
	print('Encoder params {:2f}M, rendering params {:2f}M'.format(encoder_params/1e6,rendering_params/1e6))

	datasets = ['tbd','tbd3d','tbdfalling']
	evaluate_on(encoder, rendering, device, datasets[2])



if __name__ == "__main__":
    main()