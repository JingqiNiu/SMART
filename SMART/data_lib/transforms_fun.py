import albumentations
import numpy as np
import cv2


class RandomCenterCut(albumentations.core.transforms_interface.ImageOnlyTransform):
    """ Center circle cut """

    def __init__(self, scale=0, always_apply=False, p=0.5):
        '''
        scale:  Percentage of cut
        '''
        super(RandomCenterCut, self).__init__(always_apply, p)
        self.cut_scale = scale
        
    def get_params(self):
        return { "keep_scale" :  np.random.uniform(1 - self.cut_scale, 1)}

    def apply(self, img, keep_scale, **params):
        h, w, d = img.shape
        x, y = w//2, h//2
        r = min(x, y) * keep_scale

        mask = np.zeros((h, w), np.uint8)
        mask = cv2.circle(mask, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=mask)
        return img.astype(np.uint8)
