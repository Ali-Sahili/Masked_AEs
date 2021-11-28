import os
import cv2
import torch
import argparse
import numpy as np 
import imageio as io
from PIL import Image

from torchvision.transforms import transforms
from torch.cuda.amp import autocast as autocast

from losses.mae_loss import build_mask
from models.mae import MAEVisionTransformers as MAE


parser = argparse.ArgumentParser()
parser.add_argument('--test_image', type=str)
parser.add_argument('--ckpt', default='checkpoints', type=str)
parser.add_argument('--use-gpu', action='store_true')


def test_mae(args):
    img = io.imread(args.test_image)
    raw_image = cv2.resize(img, (224, 224))
    cv2.imwrite("output/src_image.jpg", raw_image)
    image  = Image.open("output/src_image.jpg")
    raw_tensor  = torch.from_numpy(np.array(image))
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = transforms.Compose([transforms.Resize((224, 224)),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, 
                                                     std=std)]
                              )(image)
    image_tensor = image.unsqueeze(0)

    model = MAE( img_size = 224, patch_size = 16,  
                 encoder_dim = 192, encoder_depth = 12, encoder_heads = 3,
                 decoder_dim = 512, decoder_depth = 8, decoder_heads = 16, 
                 mask_ratio = 0.75
               )
    # print(model)

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")['state_dict']
        model.load_state_dict(ckpt, strict=True)

    if args.use_gpu and torch.cuda.is_available():
        model.cuda()
        image_tensor = image_tensor.cuda()

    model.eval()

    with torch.no_grad():
        with autocast():
            output, mask_index = model(image_tensor)
            print(output.shape)
        
    output_image = output.squeeze(0)
    output_image = output_image.permute(1,2,0).cpu().numpy()
    output_image = output_image * std + mean
    output_image = output_image * 255
 
    output_image = output_image[:,:,::-1]
    cv2.imwrite("output/output_image.jpg", output_image)

    mask_map = build_mask(mask_index, patch_size=16, img_size=224)

    non_mask = 1 - mask_map 
    non_mask = non_mask.unsqueeze(-1)

    non_mask_image = non_mask * raw_tensor

    mask_map = mask_map * 127
    mask_map = mask_map.unsqueeze(-1)

    print(torch.min(mask_map))

    non_mask_image  += mask_map 

    # print(non_mask_image)
    non_mask_image = non_mask_image.cpu().numpy()
    print(non_mask_image.shape)
    
    cv2.imwrite("output/mask_image.jpg", non_mask_image[:,:,::-1])
    print(output_image.shape)

if __name__ == "__main__":
    args = parser.parse_args()
    test_mae(args)
