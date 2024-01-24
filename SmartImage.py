import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
import re

class SmartImage:
    def __init__(self, image, coordinates, rotation_number = 0):
        self.name = None
        self.label = None
        self.rotation_number = rotation_number
        self.img = image
        self.coord = coordinates
    
    def contains(self, point):
        assert len(point) == 2
        x, y = point[0], point[1]

        xs = self.coord[:,1]
        x_high, x_low = xs.max(), xs.min()
        ys = self.coord[:,0]
        y_high, y_low = ys.max(), ys.min()
        
        if x_low <= x <= x_high and y_low <= y <= y_high:
            return True
        else:
            return False

    def setLabel(self, label):
        self.label = label

    def setName(self, name, wire):
        file = re.split("\.|\/", str(name))
        if wire < 10: #TODO and wire.len()<2 ?
            self.name = f"{file[-2]}_0{wire}.tiff"
        else:
            self.name = f"{file[-2]}_{wire}.tiff"

    def increase_rot_num(self):
        self.rotation_number +=1
        if self.rotation_number % 4 == 0:
            self.rotation_number = 0

    def assess_coords(self):
        if self.rotation_number == 0 or self.rotation_number % 4 == 0:
            oc = self.coord
            if oc[0][0] > oc[1][0]: #y axis
                self.coord = np.array([[oc[1][0], oc[0][1]], 
                                       [oc[0][0], oc[1][1]]])
            if oc[0][1] > oc[1][1]: #x axis
                self.coord = np.array([[oc[0][0], oc[1][1]], 
                                       [oc[1][0], oc[0][1]]])

    def fliplr(self):
        assert isinstance(self.img, np.ndarray), "Image is not an numpy.ndarray instance."
        self.img = np.fliplr(self.img)
        self.coord[0][1], self.coord[1][1] = self.coord[1][1], self.coord[0][1] 

    def rot90(self, number = 1):
        assert isinstance(self.img, np.ndarray), "Image is not an numpy.ndarray instance."
        assert number == 1 or number == 3, 'Please provide an integer equal to 1 or 3 for "number".'

        for i in range(1, number + 1):
            self.img = np.rot90(self.img)
            self.increase_rot_num()
            if self.rotation_number % 2 == 0:
                self.coord[0][1], self.coord[1][1] = self.coord[1][1], self.coord[0][1] 
            else:
                self.coord[0][0], self.coord[1][0] = self.coord[1][0], self.coord[0][0]
    
    @staticmethod
    def min_pool(img, sz = (5,2), stride = (5,1)):
        im = copy.deepcopy(img)
        im_size = img.shape

        for i in range(0, im_size[0], stride[0]):
            for j in range(0, im_size[1], stride[1]):
                im[i : i + sz[0], j : j + sz[1]] = np.min(im[i : i + sz[0],j : j + sz[1]])
        return im

    @staticmethod
    def max_pool(img, sz = (5,2), stride = (5,1)):
        im = copy.deepcopy(img)
        im_size = img.shape

        for i in range(0, im_size[0], stride[0]):
            for j in range(0, im_size[1], stride[1]):
                im[i : i + sz[0], j : j + sz[1]] = np.max(im[i : i + sz[0],j : j + sz[1]])
        return im

    def cut(self, side):
        A = cv2.fastNlMeansDenoising(copy.deepcopy(self.img))
        oc = copy.deepcopy(self.coord) # original coordinates

        min_pl = SmartImage.min_pool(A)[A.shape[0]//3:2*A.shape[0]//3]
        max_pl = SmartImage.max_pool(A)[A.shape[0]//3:2*A.shape[0]//3]

        avg_min = np.min(min_pl, axis=0)
        avg_max = np.max(max_pl, axis=0)
    
        grad_min = np.abs(np.gradient(avg_min))
        grad_max = np.abs(np.gradient(avg_max))

        cut_point_1 = np.argmax(grad_min) + 1
        cut_point_2 = np.argmax(grad_max) + 1
        cut_point = cut_point_1 if cut_point_1>cut_point_2 else cut_point_2

        self.img = self.img[:, cut_point :]
        if side % 2 == 0:
            offset = -1 * cut_point
        else:
            offset = cut_point
        if side == 1 or side == 2:
            new_coord = np.array([[oc[0][0], oc[0][1] + offset], 
                                  oc[1]])
        elif side == 3 or side == 4:
            new_coord = np.array([oc[0], 
                                  [oc[1][0] - offset, oc[1][1]]])
        else:
            raise AssertionError('"side" must be and int with a value of 1, 2, 3, or 4.')
        
        assert new_coord[0][1] >= 0, f"Coordinate lower than 0 after processing side {side}."
        assert new_coord[1][0] >= 0, f"Coordinate lower than 0 after processing side {side}."
        self.coord = new_coord

    def print(self):
        plt.imshow(self.img, cmap='Greys_r')
        plt.annotate(self.label, xy = (0, 0))
        plt.savefig("test.png" )
    
