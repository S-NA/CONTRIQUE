import torch
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
from torchvision import transforms
import numpy as np
import os
import argparse
from PIL import Image
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def compute_similarities(feat1, feat2):
    # Compute Euclidean distance
    euclidean_dist = euclidean(feat1, feat2)
    
    # Compute Cosine similarity
    cos_sim = cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1))[0][0]
    
    # Compute distance-based similarity
    distance_based_sim = 1 / (1 + euclidean_dist)
    
    return euclidean_dist, cos_sim, distance_based_sim

def main(args):
    # Load images
    image1 = Image.open(args.im_path1)
    image2 = Image.open(args.im_path2)
    
    # Downscale images by 2
    sz1 = image1.size
    sz2 = image2.size
    image1_2 = image1.resize((sz1[0] // 2, sz1[1] // 2))
    image2_2 = image2.resize((sz2[0] // 2, sz2[1] // 2))
    
    # Transform to tensor
    image1 = transforms.ToTensor()(image1).unsqueeze(0).cuda()
    image1_2 = transforms.ToTensor()(image1_2).unsqueeze(0).cuda()
    image2 = transforms.ToTensor()(image2).unsqueeze(0).cuda()
    image2_2 = transforms.ToTensor()(image2_2).unsqueeze(0).cuda()
    
    # Load CONTRIQUE Model
    encoder = get_network('resnet50', pretrained=False)
    model = CONTRIQUE_model(args, encoder, 2048)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device.type))
    model = model.to(args.device)
    
    # Extract features
    model.eval()
    with torch.no_grad():
        _, _, _, _, model_feat1, model_feat1_2, _, _ = model(image1, image1_2)
        _, _, _, _, model_feat2, model_feat2_2, _, _ = model(image2, image2_2)
    
    # Combine features
    feat1 = np.hstack((model_feat1.detach().cpu().numpy(), model_feat1_2.detach().cpu().numpy()))
    feat2 = np.hstack((model_feat2.detach().cpu().numpy(), model_feat2_2.detach().cpu().numpy()))
    
    # Compute similarities
    euclidean_dist, cos_sim, distance_based_sim = compute_similarities(feat1, feat2)
    
    # Print results
    print(f'Euclidean Distance: {euclidean_dist}')
    print(f'Cosine Similarity: {cos_sim}')
    print(f'Distance-based Similarity: {distance_based_sim}')

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--im_path1', type=str, 
                        default='sample_images/33.bmp', 
                        help='Path to first image', metavar='')
    parser.add_argument('--im_path2', type=str, 
                        default='sample_images/34.bmp', 
                        help='Path to second image', metavar='')
    parser.add_argument('--model_path', type=str, 
                        default='models/CONTRIQUE_checkpoint25.tar', 
                        help='Path to trained CONTRIQUE model', metavar='')
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

