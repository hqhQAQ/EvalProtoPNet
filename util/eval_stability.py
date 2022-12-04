import os
import cv2
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
from util.datasets import Cub2011Eval
from util.preprocess import mean, std
from util.local_parts import id_to_path, id_to_part_loc, id_to_bbox, in_bbox


def recover_img(norm_img, mean, std):
    img = torch.mul(norm_img, std) + mean
    img *= 255
    return img


def perturb_img(norm_img, std=1.):
    noise = torch.zeros(norm_img.shape).normal_(mean=0, std=std).cuda()
    perturb_img = norm_img + noise
    return perturb_img


def infer_on_dataset(ppnet, num_classes, test_loader, args, use_perturb=False, half_size=36):
    img_size = args.input_size
    # Infer on the Test Dataset
    proto_per_class = 10
    part_num = 15
    all_proto_acts, all_targets, all_img_ids = [], [], []
    for idx, (data, targets, img_ids) in tqdm(enumerate(test_loader)):
        data = data.cuda()
        targets = targets.cuda()

        if use_perturb:
            data = perturb_img(data, std=0.2)
        conv_output, proto_acts = ppnet.push_forward(data)
        # select the prototypes belonging to its category
        fea_size = proto_acts.shape[-1]
        proto_indices = (targets * proto_per_class).unsqueeze(dim=-1).repeat(1, proto_per_class)
        proto_indices += torch.arange(proto_per_class).cuda()
        proto_indices = proto_indices[:, :, None, None].repeat(1, 1, fea_size, fea_size)
        proto_acts = torch.gather(proto_acts, 1, proto_indices)

        all_proto_acts.append(proto_acts.cpu().detach())
        all_targets.append(targets.cpu())
        all_img_ids.append(img_ids)
    all_proto_acts = torch.cat(all_proto_acts, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_img_ids = torch.cat(all_img_ids, dim=0).numpy()

    class_proto_to_part = []
    for test_image_label in tqdm(range(num_classes)):
        arr_ids = np.nonzero(all_targets == test_image_label)[0]
        cur_proto_acts = all_proto_acts[arr_ids, :]
        img_ids = all_img_ids[arr_ids]
        all_proto_to_part = []

        for index, img_id in enumerate(img_ids):
            test_image_path = os.path.join(args.data_path, 'images', id_to_path[img_id][0], id_to_path[img_id][1])
            original_img = cv2.imread(test_image_path)
            original_img = cv2.resize(original_img, (img_size, img_size))
            prototype_activation_patterns = cur_proto_acts[index : index + 1]

            ##### Get Part Labels
            part_labels = []
            bbox = id_to_bbox[img_id]
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            part_locs = id_to_part_loc[img_id]
            for part_loc in part_locs:
                part_id = part_loc[0] - 1   # begin from 0
                loc_x, loc_y = part_loc[1] - bbox_x1, part_loc[2] - bbox_y1
                ratio_x, ratio_y = loc_x / (bbox_x2 - bbox_x1), loc_y / (bbox_y2 - bbox_y1)
                re_loc_x, re_loc_y = int(img_size * ratio_x), int(img_size * ratio_y)
                part_labels.append([part_id, re_loc_x, re_loc_y])

            ##### Prototypes From The True Class
            proto_to_part = np.zeros((proto_per_class, part_num))

            for prototype_index in range(proto_per_class):
                activation_pattern = prototype_activation_patterns[0, prototype_index]
                upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                        interpolation=cv2.INTER_CUBIC)

                ##### Show the most highly activated patch of the image by this prototype
                max_indice = np.where(upsampled_activation_pattern==upsampled_activation_pattern.max())
                max_indice = (max_indice[0][0], max_indice[1][0])
                high_act_patch_indices = (max(0, max_indice[0] - half_size), min(img_size, max_indice[0] + half_size), max(0, max_indice[1] - half_size), min(img_size, max_indice[1] + half_size))
                
                ##### Get the related parts of prototype j
                for part_label in part_labels:
                    part_id, loc_x, loc_y = part_label[0], part_label[1], part_label[2]
                    if in_bbox((loc_y, loc_x), high_act_patch_indices):
                        proto_to_part[prototype_index, part_id] = 1



            all_proto_to_part.append(proto_to_part)
        all_proto_to_part = np.stack(all_proto_to_part, axis=0)
        all_proto_to_part = np.transpose(all_proto_to_part, (1, 0, 2))
        class_proto_to_part.append(all_proto_to_part)
    return class_proto_to_part


def evaluate_stability(ppnet, args, half_size = 36):
    ppnet.eval()
    ppnet_without_ddp = ppnet.module if hasattr(ppnet, 'module') else ppnet
    img_size = ppnet_without_ddp.img_size

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = Cub2011Eval(args.data_path, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=False, shuffle=False)
    num_classes = args.nb_classes

    class_proto_to_part = infer_on_dataset(ppnet, num_classes, test_loader, args, use_perturb=False, half_size=half_size)
    class_proto_to_part_perturb = infer_on_dataset(ppnet, num_classes, test_loader, args, use_perturb=True, half_size=half_size)

    all_stability_score = []
    for i in range(len(class_proto_to_part)):
        proto_to_part = class_proto_to_part[i]
        proto_to_part_perturb = class_proto_to_part_perturb[i]
        is_equal = ((proto_to_part - proto_to_part_perturb).sum(axis=-1) == 0).astype(np.float32)
        stability_score = is_equal.mean()
        all_stability_score.append(stability_score)
    all_stability_score = np.array(all_stability_score)
    all_stability_score = all_stability_score.mean()

    return all_stability_score * 100