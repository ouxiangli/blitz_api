# %%
from app.stgcn_plus_plus.data_loader import *
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import random
import torch
import numpy as np
from torch import nn
from app.stgcn_plus_plus.stgcn import STGCN
import os
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# %%
class RecognizerGCN(nn.Module):
    def __init__(self):
        super().__init__()
        # record the source of the backbone
        graph_cfg = {'layout': 'mediapipe', 'mode': 'spatial'}
        kwargs = {
            'gcn_adaptive': 'init',
            'gcn_with_res': True,
            'tcn_type': 'unit_tcn' # unit_tcn, mstcn
        }

        self.backbone = STGCN(
            graph_cfg,
            in_channels=4,
            base_channels=64,
            data_bn_type='VC',
            ch_ratio=2,
            num_person=2,  # * Only used when data_bn_type == 'MVC'
            num_stages=10,
            inflate_stages=[5, 8],
            down_stages=[5, 8],
            pretrained=None,
            **kwargs
        )

        # self.loss_cls = nn.CrossEntropyLoss()
        # fc_cls = nn.Linear(in_c, num_classes)
        self.fc_cls = nn.Linear(256, 3)

    def forward(self, x):
        x = self.backbone(x)

        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)

        x = pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)

        cls_score = self.fc_cls(x)
        return cls_score


# %%
class ActionRecognitioner():
    def __init__(self):
        super().__init__()
        self.model = RecognizerGCN()
        self.model.load_state_dict(torch.load('app/models/stgcnpp.pth')['model'])
        self.model.eval()
        self.actions = ['write', 'read', 'eat']

    def inference(self, image):
        results = pose.process(image)
        if results.pose_landmarks == None:
            return False, self.actions[0]
        skeleton = []
        for i in range(23):
            landmark = results.pose_landmarks.landmark[i]
            skeleton.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        skeleton = np.array(skeleton)
        x = np.expand_dims(skeleton, (0, 1, 2))
        pred_y = self.model(Tensor(x))
        idx = pred_y.argmax().item()

        return True, self.actions[idx]

# %%
# action_recognitioner = ActionRecognitioner()
# image = cv2.imread("1.png")
# print(action_recognitioner.inference(image))





