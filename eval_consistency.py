import os
import model
import torch
import argparse
from util.eval_interpretability import evaluate_consistency
    

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--data_set', default='CUB2011', type=str)
parser.add_argument('--data_path', type=str, default='datasets/cub200_cropped/')
parser.add_argument('--nb_classes', type=int, default=200)
parser.add_argument('--test_batch_size', type=int, default=30)

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

consistency_score = evaluate_consistency(ppnet, args)
print('Consistency Score : {:.2f}%'.format(consistency_score))