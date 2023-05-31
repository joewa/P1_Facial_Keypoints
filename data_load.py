import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
from torchvision import transforms


# Use only keypoints that have a high probability to be associated with an edge
keypoint_list = []
keypoint_sublist = [19, 22, 23, 26, 28, 31, 37, 40, 43, 46, 49, 52, 55, 59]
for id in keypoint_sublist:
    keypoint_list.append((id-1)*2)
    keypoint_list.append((id-1)*2+1)


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.key_pts_frame = self.key_pts_frame[['Unnamed: 0']+[str(i) for i in keypoint_list]]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


norm_means = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]#


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h)/2)
        left = int((w - new_w)/2)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}


class ToTensorRGB(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # Convert BGR image to RGB image
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        #print(image.shape)
        normalizer = transforms.Normalize(norm_means, norm_std)
        image_torch = normalizer.forward(torch.from_numpy(image))

        key_pts_copy = np.copy(key_pts)        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0
        
        return {'image': image_torch,
                'keypoints': torch.from_numpy(key_pts_copy)}


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


class FaceCrop(object):
    """Crop a face in the image in a sample using the keypoints.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(
            self, output_size,
            random_rotate_deg=None,
            random_pan_percentage=None,
            random_scale=0.0,  # Not implemented
        ):
        #assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            #assert len(output_size) == 2
            self.output_size = output_size
        self.random_rotate_deg = random_rotate_deg
        self.random_pan_percentage = random_pan_percentage
        # TODO self.colorjitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        image_np, key_pts = sample['image'], sample['keypoints']
        h, w = image_np.shape[:2]
        # Random rotation
        if self.random_rotate_deg is not None:
            angle = np.random.uniform(-self.random_rotate_deg, self.random_rotate_deg)
            # angle = 20.0
            # Rotation of the keypoints
            keypoints_rotation_matrix = np.array([
                    [+np.cos(np.radians(angle)), -np.sin(np.radians(angle))], 
                    [+np.sin(np.radians(angle)), +np.cos(np.radians(angle))]
                ])
            image_rot_np = rotate_image(image_np, angle)
            h_rot, w_rot = image_rot_np.shape[:2]
            center = (w/2, h/2)  # get the center coordinates of the image to create the 2D rotation matrix
            key_pts = key_pts - center
            key_pts = np.matmul(key_pts, keypoints_rotation_matrix)
            key_pts = key_pts + center + ((w_rot-w)/2, (h_rot-h)/2)
            h, w = h_rot, w_rot  # image.shape[:2]
        else:
            image_rot_np = image_np
        # Face cropping using the keypoints
        weighted_center_0, weighted_center_1 = (key_pts[:, 0]).mean(), (key_pts[:, 1]).mean()
        new_h = (key_pts[:, 1].max() - key_pts[:, 1].min()) * self.output_size # * rand_scale
        new_w = (key_pts[:, 0].max() - key_pts[:, 0].min()) * self.output_size # * rand_scale
        new_wh_max = int(min(
            max(new_h, new_w),
            w, h
        ))
        top = max(0, int(min(weighted_center_1 - new_wh_max/2, key_pts[:, 1].min())))
        left = max(0, int(min(weighted_center_0 - new_wh_max/2, key_pts[:, 0].min())))
        new_wh_max = min(new_wh_max, h - top - 1, w - left - 1)
        bottom = top + new_wh_max
        right = left + new_wh_max
        # Random pan
        if self.random_pan_percentage is not None:
            key_pts_crop = key_pts - [left, top]  # key_pts in the coordinate system square new_wh_max, where (left, top is the new (0,0))
            pan_x_to_right_lim = max(right - w, -key_pts_crop[:, 0].min())
            pan_x_to_left_lim = min(left, new_wh_max - key_pts_crop[:, 0].max())
            pan_y_to_bottom_lim = max(bottom - h, -key_pts_crop[:, 1].min())
            pan_y_to_top_lim = min(top, new_wh_max - key_pts_crop[:, 1].max())
            pan_x = int(np.random.uniform(pan_x_to_right_lim, pan_x_to_left_lim) * self.random_pan_percentage)
            pan_y = int(np.random.uniform(pan_y_to_bottom_lim, pan_y_to_top_lim) * self.random_pan_percentage)
            # print(
            #     'x: ', pan_x_to_right_lim, pan_x, pan_x_to_left_lim,
            #     '; w:', w, ' left:', left, ' right:', right, ' kp_min:', key_pts_crop[:, 0].min(), ' kp_max:',  key_pts_crop[:, 0].max()
            # )
            # print(
            #     'y: ', pan_y_to_bottom_lim, pan_y, pan_y_to_top_lim,
            #     '; h:', h, ' top:', top, ' bottom:', bottom, ' kp_min:', key_pts_crop[:, 1].min(), ' kp_max:',  key_pts_crop[:, 1].max()
            # )
            left, right = left - pan_x, right - pan_x
            top, bottom = top - pan_y, bottom - pan_y
            #key_pts = key_pts - (pan_x, pan_y)

        key_pts = key_pts - [left, top]
        image_rot_np = image_rot_np[top:bottom, left:right]

        return {'image': image_rot_np, 'keypoints': key_pts}
