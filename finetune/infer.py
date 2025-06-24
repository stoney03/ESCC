import os
os.environ["OMP_NUM_THREADS"]="10"
import timm
import torch
import torch.nn as nn
from math import floor
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
import openphi
from torchvision import transforms
from train import MultiHeadAttentionMILAggregator, TrainablePositionalEncoding, load_model


def load_model_and_weights(args, model, position_encoder, mil_aggregator, checkpoint_path):
    checkpoint = torch.load(checkpoint_path,map_location=f"cuda:{args.gpus}")
    model.load_state_dict(checkpoint['model_state_dict'])
    position_encoder.load_state_dict(checkpoint['position_encoder_state_dict'])
    mil_aggregator.load_state_dict(checkpoint['mil_aggregator_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model, position_encoder, mil_aggregator

def split_into_patches(wsi, coord, bigpatch_size, patch_size):
    image = wsi.read_region(coord, 0, (bigpatch_size, bigpatch_size)).convert('RGB')
    img_width, img_height = image.size
    patches = []

    for i in range(0, img_width, patch_size):
        for j in range(0, img_height, patch_size):

            patch = image.crop((i, j, min(i + patch_size, img_width), min(j + patch_size, img_height)))

            to_tensor = transforms.ToTensor()
            patch_tensor = to_tensor(patch)

            patch_tensor = patch_tensor.unsqueeze(0)
            patches.append(patch_tensor)

    patches_tensor = torch.cat(patches, dim=0)
    return patches_tensor


def compute_w_loader(args, file_path, output_path, wsi, position_encoder, mil_aggregator, device, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 20, 'pin_memory': True}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
                     
			batch_aggregated_features = []

			for i in range(batch.size(0)):
				large_patch = batch[i]
				patch_coords = coords[i]
                
				small_patches = split_into_patches(wsi, patch_coords, bigpatch_size=args.input_size, patch_size=224)
				small_patches = small_patches.to(device)
				small_patch_features = model(small_patches)
				small_patch_features = small_patch_features.to(device)
				small_patch_features = position_encoder(small_patch_features)
				small_patch_features = small_patch_features.to(device)
				small_patch_aggregated_feature = mil_aggregator(small_patch_features)
				batch_aggregated_features.append(small_patch_aggregated_feature)
                        
			batch_aggregated_features = torch.cat(batch_aggregated_features, dim=0)
			batch_aggregated_features = batch_aggregated_features.cpu().numpy()

			asset_dict = {'features': batch_aggregated_features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--slide_ext', type=str, default= '.svs', help="Data suffix.")
parser.add_argument('--data_h5_dir', type=str, default="", help="Preprocessed data from CLAM.")
parser.add_argument('--data_slide_dir', type=str, default="", help="WSI directory.")
parser.add_argument('--csv_path', type=str, default="./process_list_autogen.csv", help="Preprocessed data from CLAM.")
parser.add_argument('--input_size', type=int, default=1120, help="Patch size.")
parser.add_argument('--feat_dir', type=str, default="", help="Directory to save feat.")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--gpus', type=int, default=0, help="GPU indices, comma separated, e.g. '0,1' ")
parser.add_argument('--checkpoint_path', type=str, default="", help="Fine-tune model path.")
parser.add_argument('--model', default='vit_large_uni', help="Fine-tune model name.")

if __name__ == '__main__':
      
	args = parser.parse_args()
	print('initializing dataset')
	bags_dataset = Dataset_All_Bags(args.csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	device = torch.device(args.gpus)

	num_patches = (args.input_size // 224) ** 2
	model = timm.create_model("vit_large_patch16_224", img_size=224, init_values=1e-5, num_classes=0, dynamic_img_size=True)
	

	position_encoder = TrainablePositionalEncoding(num_patches=num_patches, embed_dim=1024, input_size=args.input_size, patch_size=224)  # 用适当的参数初始化
	mil_aggregator = MultiHeadAttentionMILAggregator(num_patches=num_patches, embed_dim=1024)


	model, position_encoder, mil_aggregator = load_model_and_weights(args, model, position_encoder, mil_aggregator, args.checkpoint_path)	

	model = model.to(device)

	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
  
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openphi.OpenPhi(slide_file_path) if slide_file_path.endswith(".isyntax") else openslide.open_slide(slide_file_path) 
		output_file_path = compute_w_loader(args, h5_file_path, output_path, wsi, position_encoder, mil_aggregator, device,
		                   model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		                   custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
