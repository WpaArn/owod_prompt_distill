
import torch
import numpy as np
from torch import nn
from core.pascal_voc import VOC_COCO_CLASS_NAMES

# voc="M-OWODB"
voc="S-OWODB"
# voc="IOD"

def get_semantic_vectors(class_names, norm=True):
    # text = clip.tokenize(class_names + ['background']).to('cpu')
    with torch.no_grad():
        text_features = []
        for class_name in class_names:
            text_features.append(np.loadtxt('datasets/clip/'+ voc + '/' + class_name + '.txt'))
        # text_features.append(np.loadtxt('datasets/clip/' + 'unknown.txt'))
        text_features = torch.tensor(np.array(text_features))
        if norm:
            x_norm = torch.norm(text_features, p=2, dim=1).unsqueeze(1).expand_as(text_features)
            text_features = text_features.div(x_norm + 1e-5)
    return text_features.to('cuda').type(torch.float)


class AuxFastRCNNOutputLayers(nn.Module):

    def __init__(self, cfg, input_size, num_classes):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
        """
        super().__init__()
        semantic_dim = 512
        self.project = nn.Linear(input_size, semantic_dim)
        nn.init.normal_(self.project.weight, std=0.001)
        nn.init.constant_(self.project.bias, 0)
        classes = VOC_COCO_CLASS_NAMES[voc] 
        self.semantic_vectors = get_semantic_vectors(classes, norm=True)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        semantic_x = x

        semantic_x = self.project(semantic_x)
        aux_cls_score = torch.matmul(self.semantic_vectors[None, :, :], semantic_x[:, :, None]).squeeze(2)

        return aux_cls_score
