import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def find_label_map_name(img_filenames, labelExtension=".png"):
    img_filenames = img_filenames.replace('_sat.jpg', '_mask')
    return img_filenames + labelExtension


def RGB_mapping_to_class(label):
    l, w = label.shape[0], label.shape[1]
    classmap = np.zeros(shape=(l, w))
    #backgroundbackground[0,0,0], eyes[0,255,0],underwear[255,0,255],hair[0,0,255]
    indices = np.where(np.all(label == (0,0,0), axis=-1))
    if len(indices[0])!=0:
        classmap[indices[0].tolist(), indices[1].tolist()] = 0
    indices = np.where(np.all(label == (0,255,0), axis=-1))
    if len(indices[0]) != 0:
        classmap[indices[0].tolist(), indices[1].tolist()] = 0
    indices = np.where(np.all(label == (255,0,255), axis=-1))
    if len(indices[0]) != 0:
        classmap[indices[0].tolist(), indices[1].tolist()] = 0
    indices = np.where(np.all(label == (0,0,255), axis=-1))
    if len(indices[0]) != 0:
        classmap[indices[0].tolist(), indices[1].tolist()] = 0
    #vitiligo [255,255,0]
    indices = np.where(np.all(label == (255,255,0), axis=-1))
    if len(indices[0]) != 0:
        classmap[indices[0].tolist(), indices[1].tolist()] = 1
    # normal skin normal skin[255,0,0],mouth [0,255,255]
    indices = np.where(np.all(label == (255,0,0), axis=-1))
    if len(indices[0]) != 0:
        classmap[indices[0].tolist(), indices[1].tolist()] = 2
    indices = np.where(np.all(label == (0,255,255), axis=-1))
    if len(indices[0]) != 0:
        classmap[indices[0].tolist(), indices[1].tolist()] = 2
    #     plt.imshow(colmap)
    #     plt.show()
    return classmap


def classToRGB(label):
    l, w = label.shape[0], label.shape[1]
    colmap = np.zeros(shape=(l, w, 3)).astype(np.float32)
    indices = np.where(label == 0)
    if len(indices[0]) != 0:
        colmap[indices[0].tolist(), indices[1].tolist(), :] = [0, 0, 0]
    indices = np.where(label == 1)
    if len(indices[0]) != 0:
        colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 255, 255]
    indices = np.where(label == 2)
    if len(indices[0]) != 0:
        colmap[indices[0].tolist(), indices[1].tolist(), :] = [255, 0, 0]
    transform = ToTensor();
    #     plt.imshow(colmap)
    #     plt.show()
    return transform(colmap)


def class_to_target(inputs, numClass):
    batchSize, l, w = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    target = np.zeros(shape=(batchSize, l, w, numClass), dtype=np.float32)
    for index in range(7):
        indices = np.where(inputs == index)
        temp = np.zeros(shape=7, dtype=np.float32)
        temp[index] = 1
        target[indices[0].tolist(), indices[1].tolist(), indices[2].tolist(), :] = temp
    return target.transpose(0, 3, 1, 2)


def label_bluring(inputs):
    batchSize, numClass, height, width = inputs.shape
    outputs = np.ones((batchSize, numClass, height, width), dtype=np.float)
    for batchCnt in range(batchSize):
        for index in range(numClass):
            outputs[batchCnt, index, ...] = cv2.GaussianBlur(inputs[batchCnt, index, ...].astype(np.float), (7, 7), 0)
    return outputs


class Viti(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, ids, label=False, transform=False):
        super(Viti, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        self.classdict = {0: "background", 1: "vitiligo", 2: "normalskin"}
        
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04)
        self.resizer = transforms.Resize((2448, 2448))

    def __getitem__(self, index):
        sample = {}
        sample['id'] = self.ids[index][:-4]
        image = Image.open(os.path.join(self.root, "JPEGImages/" + self.ids[index])) # w, h
        image = image.convert("RGB")
        sample['image'] = image
        # sample['image'] = transforms.functional.adjust_contrast(image, 1.4)
        if self.label:
            # label = scipy.io.loadmat(join(self.root, 'Notification/' + self.ids[index].replace('_sat.jpg', '_mask.mat')))["label"]
            # label = Image.fromarray(label)
            label = Image.open(os.path.join(self.root, 'Annotations/' + self.ids[index].replace('.jpg', '.png')))
            N_mask = Image.open(os.path.join(self.root, 'N_mask/' + self.ids[index].replace('.jpg', '.png')))
            V_mask = Image.open(os.path.join(self.root, 'V_mask/' + self.ids[index].replace('.jpg', '.png')))
            sample['N_mask'] = N_mask
            sample['V_mask'] = V_mask
            sample['label'] = label
        if self.transform and self.label:
            image, label,N_mask, V_mask = self._transform(image, label, N_mask, V_mask)
            sample['image'] = image
            sample['label'] = label
            sample['N_mask'] = N_mask
            sample['V_mask'] = V_mask
        # return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}
        return sample

    def _transform(self, image, label, N_mask, V_mask):
        # if np.random.random() > 0.5:
        #     image = self.color_jitter(image)

        # if np.random.random() > 0.5:
        #     image = transforms.functional.vflip(image)
        #     label = transforms.functional.vflip(label)

        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)
            N_mask = transforms.functional.hflip(N_mask)
            V_mask = transforms.functional.hflip(V_mask)

        if np.random.random() > 0.5:
            degree = random.choice([90, 180, 270])
            image = transforms.functional.rotate(image, degree)
            label = transforms.functional.rotate(label, degree)
            N_mask = transforms.functional.rotate(N_mask, degree)
            V_mask = transforms.functional.rotate(V_mask, degree)

        # if np.random.random() > 0.5:
        #     degree = 60 * np.random.random() - 30
        #     image = transforms.functional.rotate(image, degree)
        #     label = transforms.functional.rotate(label, degree)

        # if np.random.random() > 0.5:
        #     ratio = np.random.random()
        #     h = int(2448 * (ratio + 2) / 3.)
        #     w = int(2448 * (ratio + 2) / 3.)
        #     i = int(np.floor(np.random.random() * (2448 - h)))
        #     j = int(np.floor(np.random.random() * (2448 - w)))
        #     image = self.resizer(transforms.functional.crop(image, i, j, h, w))
        #     label = self.resizer(transforms.functional.crop(label, i, j, h, w))
        
        return image, label, N_mask, V_mask


    def __len__(self):
        return len(self.ids)