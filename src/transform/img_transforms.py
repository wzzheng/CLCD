import torchvision.transforms.functional as F
import torch, torchvision.transforms as transforms
from PIL import Image, ImageFilter
import random

class RandomColorJitter(object):
    """
    Apply ColorJitter with a given probability.
    """
    def __init__(self, p, brightness, contrast, saturation, hue):
        self.T = transforms.RandomApply([
            transforms.ColorJitter(brightness, contrast, saturation, hue)
        ], p)

    def __call__(self, img):
        img = self.T(img)
        return img

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class RandomGaussianBlur(object):
    def __init__(self, p, sigma=[0.1, 2.0]):
        self.p = p
        self.sigma = sigma
        self.T = transforms.RandomApply([GaussianBlur(sigma)], p=p)
    
    def __call__(self, img):
        img = self.T(img)
        return img
    
    def __repr__(self):
        return "{}(probability={}, sigma={})".format(self.__class__.__name__, self.p, self.sigma)
