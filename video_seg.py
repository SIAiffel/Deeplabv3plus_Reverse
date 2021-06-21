import network
import utils
import os
import random
import argparse
import numpy as np

#import Satellites 추가
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torch.onnx
import sys
#import torchvision.transforms as T
import cv2
import torchvision
import time
import torchvision.transforms as transforms



def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/SIA/buildings',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='satellites',
                        choices=['voc', 'cityscapes', 'satellites'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=3,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")                    
    parser.add_argument("--ckpt", default='./checkpoints/best_deeplabv3plus_mobilenet_satellites_os16.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    parser.add_argument('-i', '--input', help='path to input video')

    parser.add_argument('--resolution_x', help='input_video_x', default=800, type=int)
    parser.add_argument('--resolution_y', help='input_video_y', default=600, type=int)

    return parser


def main():
    opts = get_argparser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Set up model
    model_map = {
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)        

    model.eval()

    #segment(model, './images/example_05.png')

    cap = cv2.VideoCapture(opts.input)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    #save_name = f"{opts.input.split('/')[-1].split('.')[0]}"
    # define codec and create VideoWriter object 
    #out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
    #                    cv2.VideoWriter_fourcc(*'mp4v'), 30, 
    #                    (frame_width, frame_height))

    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8))
    ax2.set_title("Segmentation")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    im1 = ax1.imshow(grab_frame(cap))
    im2 = ax2.imshow(grab_frame(cap))


    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret == True:
            # get the start time
            start_time = time.time()
            with torch.no_grad():
                # get predictions for the current frame
                frame = cv2.resize(frame,(opts.resolution_x,opts.resolution_y),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                outputs = get_segment_labels(frame, model, device)
            
            # draw boxes and show current frame on screen
            segmented_image = draw_segmentation_map(outputs[0])
            final_image = image_overlay(frame, segmented_image)
            # get the end time
            end_time = time.time()
            # get the fps
            fps = 1 / (end_time - start_time)
            # add fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            # press `q` to exit
            wait_time = max(1, int(fps/4))

            #cv2.imshow('image', final_image)
            #out.write(final_image)
            avg_fps = total_fps / frame_count
            str = "FPS : %0.1f" % fps
            #print(f"Average FPS: {avg_fps:.3f}")
            cv2.putText(final_image, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 250, 255), 2)
            
            im1.set_data(frame)
            im2.set_data(final_image)
            plt.pause(0.01)

            #if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            #    break
        else:
            break

    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


def get_segment_labels(image, model, device):
    # transform the image to tensor and load into computation device
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image)
    # uncomment the following lines for more info
    # print(type(outputs))
    # print(outputs['out'].shape)
    # print(outputs)
    return outputs

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    
    label_colors = np.array([(0, 0, 0),(254, 94, 0), (128, 64, 128)])  # 0=background 1=buliding, 2=road

    
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, 3):
        index = labels == label_num
        red_map[index] = label_colors[label_num, 0]
        green_map[index] = label_colors[label_num, 1]
        blue_map[index] = label_colors[label_num, 2]
        
    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image


def image_overlay(image, segmented_image):
    alpha = 0.6 # how much transparency to apply
    beta = 1 - alpha # alpha + beta should equal 1
    gamma = 0 # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image

def grab_frame(cap):
  _, frame = cap.read()
  return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    main()
