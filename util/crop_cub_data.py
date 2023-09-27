import os
import cv2
import shutil
from tqdm import tqdm

data_root = 'datasets/CUB_200_2011/'
out_dir = 'datasets/cub200_cropped/'

img_txt = os.path.join(data_root, 'images.txt')
bbox_txt = os.path.join(data_root, 'bounding_boxes.txt')
train_txt = os.path.join(data_root, 'train_test_split.txt')

# Get the image path of each image id
id_to_path = {}
with open(img_txt, 'r') as f:
    img_lines = f.readlines()
for img_line in img_lines:
    img_id, img_path = int(img_line.split(' ')[0]), img_line.split(' ')[1][:-1]
    img_folder, img_name = img_path.split('/')[0], img_path.split('/')[1]
    id_to_path[img_id] = (img_folder, img_name)

# Get the bounding box of each image id
id_to_bbox = {}
with open(bbox_txt, 'r') as f:
    bbox_lines = f.readlines()
for bbox_line in bbox_lines:
    cts = bbox_line.split(' ')
    img_id, bbox_x, bbox_y, bbox_width, bbox_height = int(cts[0]), int(cts[1].split('.')[0]), int(cts[2].split('.')[0]), int(cts[3].split('.')[0]), int(cts[4].split('.')[0])
    bbox_x2, bbox_y2 = bbox_x + bbox_width, bbox_y + bbox_height
    id_to_bbox[img_id] = (bbox_x, bbox_y, bbox_x2, bbox_y2)

# Get the train/test (1/0) label of each image id
id_to_train = {}
with open(train_txt, 'r') as f:
    train_lines = f.readlines()
for train_line in train_lines:
    img_id, is_train = int(train_line.split(' ')[0]), int(train_line.split(' ')[1][:-1])
    id_to_train[img_id] = is_train

for img_id in tqdm(id_to_path.keys()):
    root_path = id_to_path[img_id]
    pre_path = os.path.join(data_root, 'images', root_path[0], root_path[1])
    is_train = id_to_train[img_id]
    folder_name = 'train_cropped' if is_train == 1 else 'test_cropped'  # Save training images to 'train_cropped', save test images to 'test_cropped'
    save_dir = os.path.join(out_dir, folder_name, root_path[0])
    out_path = os.path.join(save_dir, root_path[1])
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir, exist_ok=True)
    pre_img = cv2.imread(pre_path)
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = id_to_bbox[img_id]
    out_img = pre_img[bbox_y1:bbox_y2, bbox_x1:bbox_x2] # Crop the bird part
    cv2.imwrite(out_path, out_img)  # Save the cropped image

# Copy the required annotation files
required_files = ['images.txt', 'image_class_labels.txt', 'train_test_split.txt']
for file_name in required_files:
    shutil.copy(os.path.join(data_root, file_name), os.path.join(out_dir, file_name))