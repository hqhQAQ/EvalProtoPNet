import os
import cv2
import model
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm

from util.datasets import Cub2011Eval
from util.preprocess import mean, std
from util.local_parts import id_to_path, id_to_part_loc, id_to_bbox, part_num, in_bbox


all_colors = [(83, 172, 252), (212, 183, 156), (48, 89, 182), (78, 223, 244), (182, 114, 1),
                (72, 57, 55), (151, 149, 148), (204, 225, 240), (138, 181, 224), (82, 138, 155),
                (169, 219, 161), (126, 137, 235), (112, 160, 0), (166, 106, 146), (108, 57, 209)]


def draw_point(img, point, bbox_size=10, color=(0, 0, 255)):
    img[point[1] - bbox_size // 2: point[1] + bbox_size // 2, point[0] - bbox_size // 2: point[0] + bbox_size // 2] = color
    return img


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                    bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)


@torch.no_grad()
def visualize_corresponding_regions(ppnet, args, half_size=36):
    ppnet.eval()
    ppnet_without_ddp = ppnet.module if hasattr(ppnet, 'module') else ppnet
    img_size = ppnet_without_ddp.img_size
    proto_per_class = ppnet_without_ddp.num_prototypes_per_class

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = Cub2011Eval(args.data_path, train=False, transform=transform)    # CUB test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=10, pin_memory=True, drop_last=False, shuffle=False)

    # Infer on the whole test dataset
    all_proto_acts, all_targets, all_img_ids = [], [], []
    for _, (data, targets, img_ids) in tqdm(enumerate(test_loader)):
        data = data.cuda()
        targets = targets.cuda()

        _, proto_acts = ppnet_without_ddp.push_forward(data)
        # Select the prototypes belonging to the ground-truth class of each image
        fea_size = proto_acts.shape[-1]
        proto_indices = (targets * proto_per_class).unsqueeze(dim=-1).repeat(1, proto_per_class)
        proto_indices += torch.arange(proto_per_class).cuda()   # The indexes of prototypes belonging to the ground-truth class of each image
        proto_indices = proto_indices[:, :, None, None].repeat(1, 1, fea_size, fea_size)
        proto_acts = torch.gather(proto_acts, 1, proto_indices) # (B, proto_per_class, fea_size, fea_size)

        all_proto_acts.append(proto_acts.cpu().detach())
        all_targets.append(targets.cpu())
        all_img_ids.append(img_ids)
    all_proto_acts = torch.cat(all_proto_acts, dim=0).numpy()   # The activation maps of all test images
    all_targets = torch.cat(all_targets, dim=0).numpy() # The categories of all test images
    all_img_ids = torch.cat(all_img_ids, dim=0).numpy() # The image ids of all test images

    # Enumerate the visualized classes
    for test_image_label in args.vis_classes:
        arr_ids = np.nonzero(all_targets == test_image_label)[0]
        class_proto_acts = all_proto_acts[arr_ids] # Get the activation maps of all the images of current class
        img_ids = all_img_ids[arr_ids]  # Get the image ids of all the images of current class

        # Get part annotations on all the images of current class
        class_part_labels, class_original_images = [], []
        for _, img_id in enumerate(img_ids):
            test_image_path = os.path.join(args.data_path, 'test_cropped', id_to_path[img_id][0], id_to_path[img_id][1])
            # Read the image
            original_img = cv2.imread(test_image_path)
            original_img = cv2.resize(original_img, (img_size, img_size))
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Get part annotations
            part_labels, part_mask = [], np.zeros(part_num,)
            bbox = id_to_bbox[img_id]
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            part_locs = id_to_part_loc[img_id]
            for part_loc in part_locs:
                part_id = part_loc[0] - 1   # The id of current object part (begin from 0)
                part_mask[part_id] = 1  # The current object part exists in current image
                loc_x, loc_y = part_loc[1] - bbox_x1, part_loc[2] - bbox_y1
                ratio_x, ratio_y = loc_x / (bbox_x2 - bbox_x1), loc_y / (bbox_y2 - bbox_y1) # Fit the bounding boxes' coordinates to the cropped images
                re_loc_x, re_loc_y = int(img_size * ratio_x), int(img_size * ratio_y)
                part_labels.append([part_id, re_loc_x, re_loc_y])
            class_part_labels.append(part_labels)
            class_original_images.append(original_img)

        img_num = len(img_ids)
        # Init the saving directories
        for img_idx in range(img_num):
            img_dir = os.path.join(args.output_path, 'class_{}'.format(test_image_label), 'img_{}'.format(img_idx))
            if os.path.exists(img_dir) is False:
                os.makedirs(img_dir, exist_ok=True)
            original_img = class_original_images[img_idx]
            
            # Visualize the annotated objects
            '''
            part_labels = class_part_labels[img_idx]
            for part_label in part_labels:
                color = all_colors[part_label[0]]
                original_img = draw_point(original_img, [part_label[1], part_label[2]], color=color)
            '''
            plt.imsave(os.path.join(img_dir, '0_img_original.jpg'), original_img)

        # Enumerate the prototypes of current class
        for proto_idx in range(proto_per_class):
            img_num = len(img_ids)
            for img_idx in range(img_num):
                original_img = class_original_images[img_idx]
                normalize_img = np.float32(original_img) / 255
                img_dir = os.path.join(args.output_path, 'class_{}'.format(test_image_label), 'img_{}'.format(img_idx))

                part_labels = class_part_labels[img_idx]
                activation_map = class_proto_acts[img_idx, proto_idx]
                upsampled_activation_map = cv2.resize(activation_map, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)

                max_indice = np.where(upsampled_activation_map==upsampled_activation_map.max())
                max_indice = (max_indice[0][0], max_indice[1][0])
                region_pred = (max(0, max_indice[0] - half_size), min(img_size, max_indice[0] + half_size), max(0, max_indice[1] - half_size), min(img_size, max_indice[1] + half_size)) # Get the corresponding region of current prototype, (y1, y2, x1, x2)

                # Visualize the activation map
                upsampled_activation_map = upsampled_activation_map - np.amin(upsampled_activation_map)
                upsampled_activation_map = upsampled_activation_map / np.amax(upsampled_activation_map)
                heatmap = cv2.applyColorMap(np.uint8(255 * upsampled_activation_map), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[..., ::-1]
                overlayed_img = 0.7 * normalize_img + 0.3 * heatmap
                plt.imsave(os.path.join(img_dir, '%d_prototype_act.jpg' % (proto_idx)), overlayed_img)

                # Visualize the corresponding region on the original image
                imsave_with_bbox(fname=os.path.join(img_dir, '{}_prototype_bbox.jpg'.format(proto_idx)),
                                    img_rgb=normalize_img,
                                    bbox_height_start=region_pred[0],
                                    bbox_height_end=region_pred[1],
                                    bbox_width_start=region_pred[2],
                                    bbox_width_end=region_pred[3], color=(0, 255, 255))

                # Visualize the corresponding region individually
                '''
                region_pred_img = original_img[region_pred[0]:region_pred[1], region_pred[2]:region_pred[3], :]
                plt.imsave(os.path.join(img_dir, '{}_prototype_patch.jpg'.format(proto_idx)), region_pred_img)
                '''


parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--data_set', default='CUB2011', type=str)
parser.add_argument('--data_path', type=str, default='datasets/cub200_cropped/')
parser.add_argument('--nb_classes', type=int, default=200)
parser.add_argument('--test_batch_size', type=int, default=30)
parser.add_argument('--vis_classes', nargs='+', type=int)
parser.add_argument('--output_path', type=str, default='output_view/')

# Model
parser.add_argument('--base_architecture', type=str, default='vgg16')
parser.add_argument('--input_size', default=224, type=int, help='images input size')
parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
parser.add_argument('--prototype_activation_function', type=str, default='log')
parser.add_argument('--add_on_layers_type', type=str, default='regular')

parser.add_argument('--resume', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
img_size = args.input_size
device = torch.device('cuda')

# Load the model
ppnet = model.construct_OursNet(base_architecture=args.base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=args.prototype_shape,
                              num_classes=args.nb_classes,
                              prototype_activation_function=args.prototype_activation_function,
                              add_on_layers_type=args.add_on_layers_type)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
ppnet.load_state_dict(checkpoint['model'])

args.output_path = os.path.join(args.output_path, args.base_architecture)
visualize_corresponding_regions(ppnet, args)