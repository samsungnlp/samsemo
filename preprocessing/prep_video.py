# This source code is licensed under CC BY 4.0 license.
# Copyright 2024 Samsung Electronics Co., Ltd.
import glob
import os

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms as T


class VideoProcessing:
    def __init__(self, frames_path: str, img_interval: int = 500):
        self.frames_path = frames_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.normalize = T.Normalize(mean=[159, 111, 102], std=[37, 33, 32])
        self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device='cpu')
        self.img_interval = img_interval

    @staticmethod
    def crop_img_center(img: torch.tensor, target_size=48):
        '''
        Some images have un-detectable faces,
        to make the training goes normally,
        for those images, we crop the center part,
        which highly likely contains the face or part of the face.

        @img - (channel, height, width)
        '''
        current_size = img.size(1)
        off = (current_size - target_size) // 2  # offset
        cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
        return cropped

    def sample_imgs_by_interval(self, folder: str, fps=2):

        files = glob.glob(f'{folder}/*')
        files = [f for f in files if "image" in f]
        nums = (len(files) - 5) // 5 if "Ses" in folder else len(files)
        step = 1

        sampled = [os.path.join(folder, f'image_{i}.jpg') for i in list(range(0, nums, step))]
        if len(sampled) == 0:
            step = int(self.img_interval / 1000 * fps) // 4
            step = step if step != 0 else 1
            sampled = [os.path.join(folder, f'image_{i}.jpg') for i in list(range(0, nums, step))]
        return sampled

    def frames_processing_filename(self, video_filename: str):
        images_folder = f'{self.frames_path}{video_filename}'
        return self.frames_processing(images_folder)

    def frames_processing(self, images_folder: str):
        sampled_imgs = []
        for imgPath in self.sample_imgs_by_interval(images_folder):
            this_img = Image.open(imgPath)
            H = np.float32(this_img).shape[0]
            W = np.float32(this_img).shape[1]
            if H > 360:
                resize = T.Resize([270, 480])
                this_img = resize(this_img)

            sampled_imgs.append(np.float32(this_img))

        sampled_imgs = np.array(sampled_imgs, dtype=np.float32)
        faces = self.mtcnn(sampled_imgs)

        for i, face in enumerate(faces):
            if face is None:
                center = self.crop_img_center(torch.tensor(sampled_imgs[i]).squeeze().permute(2, 0, 1))
                faces[i] = center
        faces = [self.normalize(face) for face in faces]

        faces = torch.stack(faces, dim=0)
        #

        # version with uniform dims
        length = len(faces)
        if length > 16:
            faces = faces[:16]
        else:
            shapes = faces.shape
            oriLen, dim = shapes[0], (shapes[1], shapes[2], shapes[3])
            faces = torch.cat((faces, torch.zeros(16 - oriLen, *dim).to(faces.device)), dim=0)

        return faces.unsqueeze(0), length  # to stack for the batch, 1st dim should be 1
