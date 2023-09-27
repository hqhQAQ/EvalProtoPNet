import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_inat_features, resnet101_features, resnet152_features
from models.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features, vgg19_features, vgg19_bn_features

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_inat_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class OursNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(OursNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        self.prototype_activation_function = prototype_activation_function

        assert(self.num_prototypes % self.num_classes == 0)
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG'):
            self.shallow_layer_idx = 17
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('RES'):
            self.shallow_layer_idx = 0
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            self.shallow_layer_idx = 4
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.activation_weight = nn.Parameter(torch.ones(self.num_prototypes),
                                            requires_grad=True)

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        x, all_feas = self.features.forward_all(x)
        x = self.add_on_layers(x)

        return x, all_feas

    def _cosine_convolution(self, x):
        x = F.normalize(x,p=2,dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors,p=2,dim=1)
        activations = F.conv2d(input=x, weight=now_prototype_vectors)
        distances = -activations

        return distances

    def _project2basis(self,x):
        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)
        activations = F.conv2d(input=x, weight=now_prototype_vectors)
        return activations

    def prototype_distances(self, x):
        conv_features, all_feas = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)
        project_activations = self._project2basis(conv_features)

        return project_activations, cosine_distances, all_feas

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            raise Exception('other activation function NOT implemented')

    def global_min_pooling(self, distances):
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)

        return min_distances

    def global_max_pooling(self, distances):
        max_distances = F.max_pool2d(distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        max_distances = max_distances.view(-1, self.num_prototypes)

        return max_distances

    def get_clst_loss(self, min_distances, label):
        max_dist = (self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3])
        prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, label]).cuda()
        inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
        cluster_cost = torch.mean(max_dist - inverted_distances)

        return cluster_cost
    
    def get_sep_loss(self, min_distances, label):
        max_dist = (self.prototype_shape[1] * self.prototype_shape[2] * self.prototype_shape[3])
        prototypes_of_correct_class = torch.t(self.prototype_class_identity[:, label]).cuda()
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = \
            torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

        return separation_cost

    def get_ortho_loss(self,):
        cur_basis_matrix = torch.squeeze(self.prototype_vectors)
        subspace_basis_matrix = cur_basis_matrix.reshape(self.num_classes, self.num_prototypes_per_class, self.prototype_shape[1])
        subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix,1,2)
        orth_operator = torch.matmul(subspace_basis_matrix, subspace_basis_matrix_T)
        I_operator = torch.eye(subspace_basis_matrix.size(1), subspace_basis_matrix.size(1)).cuda()
        difference_value = orth_operator - I_operator
        ortho_cost = torch.sum(torch.relu(torch.norm(difference_value,p=1,dim=[1,2]) - 0))

        return ortho_cost

    def get_SDFA_loss(self, proto_acts, shallow_feas, deep_feas, label, consis_thresh=None):
        shallow_feas = shallow_feas.detach()
        shallow_feas = shallow_feas.permute(0, 2, 1)
        deep_feas = deep_feas.permute(0, 2, 1)
        
        _, fea_len = shallow_feas.shape[0], deep_feas.shape[1]
        # Select the prototypes corresponding to the label
        proto_per_class = self.num_prototypes_per_class
        proto_indices = (label * proto_per_class).unsqueeze(dim=-1).repeat(1, proto_per_class)
        proto_indices += torch.arange(proto_per_class).cuda()   # (B, 10), get 10 indices of activation maps of each sample
        max_positions = proto_acts.argmax(dim=-1)   # (B, 2000)
        max_positions = torch.gather(max_positions, 1, proto_indices)   # (B, 10)
        
        shallow_norm = shallow_feas / shallow_feas.norm(dim=-1, p=2).unsqueeze(dim=-1)
        shallow_simi = shallow_norm.bmm(shallow_norm.permute(0, 2, 1))   # (B, 49, 49)
        deep_norm = deep_feas / deep_feas.norm(dim=-1, p=2).unsqueeze(dim=-1)
        deep_simi = deep_norm.bmm(deep_norm.permute(0, 2, 1))    # (B, 49, 49)  

        max_positions = max_positions[:, :, None].repeat(1, 1, fea_len)
        shallow_simi = torch.gather(shallow_simi, 1, max_positions)
        deep_simi = torch.gather(deep_simi, 1, max_positions)

        consis_cost = F.relu(torch.abs(deep_simi - shallow_simi) - consis_thresh).mean()

        return consis_cost

    def forward(self, x):
        project_activations, cosine_distances, all_feas = self.prototype_distances(x)
        cosine_min_distances = self.global_min_pooling(cosine_distances)

        prototype_activations = self.global_max_pooling(project_activations)
        logits = self.forward_SA_module(prototype_activations)

        fea_size = project_activations.shape[-1]
        project_activations = project_activations.flatten(start_dim=2)
        shallow_feas = all_feas[self.shallow_layer_idx]
        batch_size, dim, shallow_size = shallow_feas.shape[0], shallow_feas.shape[1], shallow_feas.shape[-1]
        shallow_feas = shallow_feas.reshape(batch_size, dim, fea_size, shallow_size // fea_size, fea_size, shallow_size // fea_size)
        shallow_feas = shallow_feas.permute(0, 1, 3, 5, 2, 4)   # (B, dim, 8, 8, 7, 7)
        shallow_feas = shallow_feas.reshape(batch_size, -1, fea_size, fea_size)
        shallow_feas = shallow_feas.flatten(start_dim=2)
        deep_feas = all_feas[-1].flatten(start_dim=2)

        return logits, (cosine_min_distances, project_activations, shallow_feas, deep_feas)

    def forward_SA_module(self, prototype_activations):
        activation_weight = self.activation_weight.reshape(self.num_classes, -1)
        activation_weight = F.softmax(activation_weight, dim=-1) * self.num_prototypes_per_class
        activation_weight = activation_weight.flatten(start_dim=0)
        prototype_activations = prototype_activations * activation_weight
        logits = prototype_activations.reshape(-1, self.num_classes, self.num_prototypes_per_class).sum(dim=-1) # (B, 200)

        return logits

    def push_forward(self, x):
        conv_output, _ = self.conv_features(x)
        proto_acts = self._project2basis(conv_output)

        return conv_output, proto_acts

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def construct_OursNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 128, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)

    return OursNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)