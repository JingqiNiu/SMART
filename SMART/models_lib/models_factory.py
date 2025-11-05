import yaml
import timm
import torch.nn as nn
from .efficientnet_pytorch import EfficientNet
from .utils_model import load_pretrain_model
from  torchvision import models as tv_models
import torch

with open('models_lib/fundus_pretrained.yaml', 'r') as f:
    pretrain_dict = yaml.load(f, Loader=yaml.FullLoader)

class Model_Add_Infor(nn.Module):
    def __init__(self,name = 'efficientnet-b3', fc_num=[7,64,32,32,128,],num_classes=2, add_info = False, add_channel_num = 0):
        super(Model_Add_Infor, self).__init__()
        # 加载预训练的EfficientNet-B0模型，并去掉最后一层分类器
        if 'efficientnet' in name:
            self.backbone = EfficientNet.from_pretrained(name)
        else:
            self.backbone = timm.create_model('tinynet_c', pretrained=True, features_only=True)
        # self.backbone = load_pretrain_model(self.backbone, '/mnt/nas_ssd_data/ywchen/release/pretrained_ck/b3_dr.ckpt', 'state_dict')

        # 获取EfficientNet最后一层特征的维度
        num_features = self.backbone._fc.in_features
        print('the model last featuree dim is',num_features)
        self.add_info = add_info
        print('@@@@@@@@@@@@@@@@@ Here the ADD Infomation is',add_info)
        if add_info == True:
            self.backbone._fc = nn.Sequential(nn.Linear( num_features, 16), nn.ReLU())
            self.info_fc = nn.Sequential(nn.Linear( add_channel_num , add_channel_num ), nn.ReLU()
                                    )
            self.fc = nn.Sequential(nn.Linear( add_channel_num + 16, 64 ), nn.ReLU(),
                                    nn.Linear( 64, 32 ),nn.ReLU(),
                                    nn.Linear( 32, num_classes )
                                    )
            self.img_batchnorm = nn.BatchNorm1d(16)
        else:
            self.fc = nn.Sequential(nn.Linear( 1000, 64 ), nn.ReLU(),
                                    nn.Linear( 64, 32 ),nn.ReLU(),
                                    nn.Linear( 32, num_classes )
                                    )
            self.img_batchnorm = nn.BatchNorm1d(1000)
        self.info_batchnorm = nn.BatchNorm1d(add_channel_num)
        
    def forward(self, image, patient_info = None):
        # 通过EfficientNet提取图像特征
        # print(f'The image max {image.max()}, min {image.min()}, mean {image.mean()}, std {image.std()}')
        image_features = self.backbone(image)
        image_features = self.img_batchnorm(image_features)
        # print(f'The image_features shape is',image_features.shape)
        # print(f'The image_features max {image_features.max()}, min {image_features.min()}, mean {image_features.mean()}, std {image_features.std()}')
        # patient_info = self.fc_infor(patient_info)
        # print(f'The patient_info max {patient_info.max()}, min {patient_info.min()}, mean {patient_info.mean()}, std {patient_info.std()}')
        # 将图像特征和病人信息拼接起来
        if self.add_info == True:
            info_feature = self.info_fc(patient_info)
            info_feature = self.info_batchnorm(info_feature)
            features = torch.cat([image_features, info_feature], dim=1)
        else:
            features = image_features
        # 通过全连接层输出类别概率
        logits = self.fc(features)
        # 返回logits
        return logits


def build_model(model_name, num_classes, pretrain_checkpoint=None, key_checkpoint='state_dict', add_info = False,  config = None, **kwargs):
    """
    The available models are viewed through  "timm.list_models()" efficientnet-b0 b7
    Args:
        model_name (str): the name of model.  
        num_classes (int): the number of classes
        pretrain_checkpoint (str, optional):  The path of the checkpoint. Defaults to None.
        key_checkpoint (str, optional): the key in checkpoint. Defaults to 'state_dict'.

    Returns:
        [nn.Module]: model
    """
    more_col = config['data']['more_col']
    if model_name.startswith('efficientnet-b'):
        # EfficientNet.from_pretrained(model_name, advprop=False, num_classes=1000, in_channels=3):
        model = Model_Add_Infor(model_name, num_classes=num_classes, add_info = add_info ,add_channel_num = len(more_col))
    elif model_name.startswith('shufflenet') :
        model =  tv_models.__dict__[model_name](pretrained = True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith('mobilenet_v2'):
        model =  tv_models.__dict__[model_name](pretrained = True)
        model.classifier = nn.Sequential(nn.Dropout(0.2),nn.Linear(1280, num_classes))
    else:
        model = timm.create_model( model_name, num_classes=num_classes, pretrained=True, **kwargs)
    resume = config['resume']
    if resume != None:
        checkpoint = torch.load(resume)
        # print('checkpoint',checkpoint.keys())
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_state_dict[key[6:]] = value  # 去除 "model." 前缀
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
    if pretrain_checkpoint:  # load pretrain checkpoint
        if pretrain_checkpoint in pretrain_dict:
            pretrain_checkpoint = pretrain_dict[pretrain_checkpoint]['path']
        model = load_pretrain_model(model, pretrain_checkpoint, key_checkpoint)
    return model
