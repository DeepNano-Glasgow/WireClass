import numpy as np
from SmartImage import SmartImage

def cut_side(img1 , sides):
    for side in sides:
        if side == 1: # cut left
            img1.cut(side)
        if side == 2: # cut right
            img1.fliplr()
            img1.cut(side)
            img1.fliplr()
        if side == 3: # cut bottom
            img1.rot90(number = 3)
            img1.cut(side)
            img1.rot90()
        if side == 4: # cut top
            img1.rot90()
            img1.cut(side)
            img1.rot90(3)
    return img1

def split(img1, num = 3):
    arrays = np.array_split(img1.img, num, axis=1)
    x_sizes = []
    for array in arrays:
        x_sizes.append(array.shape[1])

    smart_image_list = []
    oc = img1.coord

    if num == 11:
        starting_x = oc[1][0]
    else:
        starting_x = oc[0][1]
    for i in range(len(arrays)):
        new_start = starting_x
        starting_x += x_sizes[i]

        if num == 11:
            coords_i = np.array([[starting_x, oc[0][1]], 
                                 [new_start, oc[1][1]]])
        else:
            coords_i = np.array([[oc[0][0], new_start], 
                                 [oc[1][0], starting_x]])

        smart_image_list.append(SmartImage(arrays[i], coords_i, img1.rotation_number))
    
    if num == 2:
        smart_image_list[1].fliplr()
        smart_image_list[1].assess_coords()
    return smart_image_list

