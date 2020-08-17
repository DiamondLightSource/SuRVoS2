import sys 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import warnings

from PIL import Image
from skimage import img_as_ubyte

warnings.simplefilter("ignore", category=UserWarning)

from loguru import logger
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


class CNNFeatures():
    """
    Simple class for extracting features from pretrained network
    """
    def __init__(self, cuda=False, layer='default', model='resnet-18',  layer_output_size=512):

        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model     
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.vec_len = 512

    def extract_feature(self, img, tensor=False):

        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)
 
        embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return embedding
        else:
            return embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        if model_name == 'resnet-18':
            self.vec_len = 512
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer
        
        if model_name == 'resnet-50':

            model = models.resnet50(pretrained=True)
            
            self.vec_len = 2048

            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer
        

        else:
            raise KeyError('Model {} not available.'.format(model_name))
            
            

