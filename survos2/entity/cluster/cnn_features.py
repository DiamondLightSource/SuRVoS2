import warnings
import numpy as np
import PIL
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from skimage import img_as_ubyte

warnings.simplefilter("ignore", category=UserWarning)


class CNNFeatures:
    """
    Extract features from pretrained network
    """

    def __init__(
        self,
        cuda=False,
        gpu_id=0,
        layer="default",
        model="resnet-18",
        layer_output_size=512,
    ):
        # self.device = torch.device("cuda" if cuda else "cpu")
        self.device = torch.device(gpu_id)
        self.layer_output_size = layer_output_size
        self.model_name = model
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()
        self.vec_len = 512

    def extract_feature(self, img, tensor=False):
        image = (
            self.normalize(self.to_tensor(self.scaler(img)))
            .unsqueeze(0)
            .to(self.device)
        )

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
        if model_name == "resnet-18":
            self.vec_len = 512
            model = models.resnet18(pretrained=True)
            if layer == "default":
                layer = model._modules.get("avgpool")
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        if model_name == "resnet-50":
            model = models.resnet50(pretrained=True)

            self.vec_len = 2048

            if layer == "default":
                layer = model._modules.get("avgpool")
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        else:
            raise KeyError("Model {} not available.".format(model_name))


def grab_features(img_3channel, num_fv=None):
    if num_fv is None:
        num_fv = len(img_3channel)

    deepfeat = CNNFeatures(cuda=True, model="resnet-50")
    vec_mat = np.zeros((num_fv, 2048))

    for j in range(0, num_fv):
        if j % 500 == 0:
            print("\nGenerated: {} features".format(j))

        vec = deepfeat.extract_feature(
            PIL.Image.fromarray(img_as_ubyte(img_3channel[j]))
        )

        vec_mat[j, :] = vec

    return vec_mat


# This is just to explore the distribution of the pixel data and check the normalization is working


def calc_patch_stats(selected_images):
    im_w, im_h = (
        selected_images[0].shape[0],
        selected_images[0].shape[1],
    )  # this is width and height of each patch 64,64
    print(im_w, im_h)
    selected_images[0].shape  # 64 , 64
    img_data = selected_images[0].copy()
    mean = img_data.mean()

    print(img_data.shape)
    print("Min: %.3f, Max: %.3f" % (img_data.min(), img_data.max()))
    print("Mean: %.3f" % mean)
    img_data = img_data - mean

    print(img_data)

    img_data = img_data - img_data.min()

    print("Min: %.3f, Max: %.3f" % (img_data.min(), img_data.max()))

    print(img_data)
    img_data /= np.max(img_data)

    print("Min: %.3f, Max: %.3f" % (img_data.min(), img_data.max()))


# The pre-trained ResNet is trained on 3 channel RGB.
# Duplicating grayscale channels seems to work ok


def stdize(image):

    mean, std = np.mean(image), np.std(image)
    image = image - mean
    image = image / std
    return image


def simple_norm(img_data):
    img_data = img_data - img_data.mean()
    img_data = img_data - img_data.min()
    # image = image / std
    img_data /= np.max(img_data)

    return img_data


def prepare_3channel(selected_images, patch_size=(28, 28)):
    print(f"Generating list of {len(selected_images)} patches of size {patch_size}")
    selected_3channel = []

    for i in range(len(selected_images)):
        img_out = np.zeros((patch_size[0], patch_size[1], 3))

        if i % (len(selected_images) // 10) == 0:
            print(i, selected_images[i].shape)

        try:
            img_data = selected_images[i]
            img_data = simple_norm(img_data)

            img_out[:, :, 0] = img_data
            img_out[:, :, 1] = img_data
            img_out[:, :, 2] = img_data

            selected_3channel.append(img_out)

        except ValueError as e:
            print(e)

    return selected_3channel
