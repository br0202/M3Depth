import os
from PIL import Image

from torch.utils.data import Dataset

file_dir = os.path.dirname(__file__)  # the directory that main.py resides in

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, filenames, transform=None):
        self.filenames = filenames
        self.loader = pil_loader
        self.img_ext = '.jpg'
        self.root_dir = root_dir
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        line = self.filenames[idx].split()
        folder = line[0]
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        left_image = self.get_color(folder, frame_index, 'l')
        if self.mode == 'train':
            right_image = self.get_color(folder, frame_index, 'r')
            sample = {'left_image': left_image, 'right_image': right_image}
            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            right_image = self.get_color(folder, frame_index, 'r')
            sample = {'left_image': left_image, 'right_image': right_image}
            if self.transform:
                if self.transform:
                    sample = self.transform(sample)
                    return sample
                else:
                    return sample

    def get_color(self, folder, frame_index, side):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        return color

    def get_image_path(self, folder, frame_index, side):
        ''' For my Endovis split'''
        # root = os.path.join(file_dir, "Data")
        # f_str = "{:06d}{}".format(frame_index, self.img_ext)
        # image_path = os.path.join(
        #     root+folder[61:], "image_0{}".format(self.side_map[side]), f_str)
        '''For the Endovis origin split'''
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            folder, "image_0{}".format(self.side_map[side]), f_str)
        return image_path

