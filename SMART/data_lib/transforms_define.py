import cv2
import albumentations
import albumentations.augmentations.transforms as transforms
import albumentations.core.composition as composition

from .transforms_fun import RandomCenterCut


def default_train(target_size, prob=0.5, aug_m=2):
    high_p = prob
    low_p = high_p / 2.0
    M = aug_m
    first_size = [int(x/0.7) for x in target_size]

    return composition.Compose([

        transforms.Resize(first_size[0], first_size[1], interpolation=3),
        transforms.Flip(p=0.5),
        composition.OneOf([
            RandomCenterCut(scale=0.1 * M),
            transforms.ShiftScaleRotate(shift_limit=0.05*M, scale_limit=0.1*M, rotate_limit=180,
                                        border_mode=cv2.BORDER_CONSTANT, value=0),
            albumentations.imgaug.transforms.IAAAffine(
                shear=(-10*M, 10*M), mode='constant')
        ], p=high_p),

        transforms.RandomBrightnessContrast(
            brightness_limit=0.1*M, contrast_limit=0.03*M, p=high_p),
        transforms.HueSaturationValue(
            hue_shift_limit=5*M, sat_shift_limit=15*M, val_shift_limit=10*M, p=high_p),
        transforms.OpticalDistortion(
            distort_limit=0.03 * M, shift_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=low_p),

        composition.OneOf([
            transforms.Blur(blur_limit=7),
            albumentations.imgaug.transforms.IAASharpen(),
            transforms.GaussNoise(var_limit=(2.0, 10.0), mean=0),
            transforms.ISONoise()
        ], p=low_p),

        transforms.Resize(target_size[0], target_size[1], interpolation=3),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.5, 0.5, 0.5), max_pixel_value=255.0)
    ], p=1)


def default_val(target_size):
    return composition.Compose([
        transforms.Resize(target_size[0], target_size[1], interpolation=3),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.5, 0.5, 0.5), max_pixel_value=255.0)
    ], p=1)

def resize_crop_train(target_size, prob=0.5, aug_m=2):
    """  
    采用了更随机的切边操作"RandomResizedCrop",代替了随机中心裁剪"RandomCenterCut".
    """
    high_p = prob
    low_p = high_p / 2.0
    M = aug_m
    return composition.Compose([
        transforms.RandomResizedCrop(target_size[0], target_size[1],scale=(0.9, 1.0),ratio=(0.95, 1.05), interpolation=3),
        transforms.Flip(p=0.5),
        composition.OneOf([
            transforms.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=30,
                                        border_mode=cv2.BORDER_CONSTANT, value=0),
            albumentations.imgaug.transforms.IAAAffine(
                shear=(-5*M, 5*M), mode='constant')
        ], p=high_p),

        transforms.RandomBrightnessContrast(
            brightness_limit=0.05*M, contrast_limit=0.1*M, p=high_p),


        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.5, 0.5, 0.5), max_pixel_value=255.0)
    ], p=1)

def your_own_train(target_size, prob=0.5, aug_m=2):
    return
