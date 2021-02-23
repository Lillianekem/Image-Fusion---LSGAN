import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import urllib

def download_url(url_file): 
    with open(url_file) as url_file: 
        url_data = url_file.readlines() 
    url_data = [g.replace('\n', '') for g in url_data]

def load_and_crop(image_paths, normalized = True):
    image_stack = []
    for i in range(len(image_paths)):
        image = Image.open(urllib.request.urlopen(image_paths[i])).resize([358,358])
        if normalized:
            return np.array(image).astype(np.float32)/255.0
        else: 
            return np.array(image).astype(np.float32)
        image_stack.append(image)
    return image_stack


def convert(image_stack, color, channel):
    image_stack_c = []
    for i in range(len(image_stack):
        if color == 'True':
            img_yuv = cv2.cvtColor(image_stack[i], cv2.COLOR_BGR2YUV)
            y, u, v = np.float32(cv2.split(img_yuv))
            if channel == 'y': 
                image_stack_c.append(y)
            if channel == 'u':
                image_stack_c.append(u)  
            if channel == 'v': 
                image_stack_c.append(v)
        if color == 'False': 
            gray = cv2.cvtColor(image_stack[i], cv2.COLOR_BGR2GRAY)
            image_stack_c.append(gray)
    return np.array(image_stack_c).reshape(148, 358, 358, 1)


def extract_patches(image_stack):
    image_patches = []
    for i in range(len(image_stack)):
        image = np.expand_dims(image_stack[i], axis=0)
        patches = tf.image.extract_patches(images=image, 
                sizes=[1, 112, 112, 1],
                strides = [1, 14, 14, 1],
                rates = [1, 1, 1, 1],
                padding = 'VALID')
        patches = tf.reshape(patches, [324, 112, 112])
        # 441 = final number of patches/number of images 
        image_patches.append(patches)
    image_patch_stack = tf.reshape(tf.stack(image_patches), [47952, 112, 112])
    image_patch_stack = np.expand_dims(image_patch_stack, axis = 3)
    return image_patch_stack
    

def crop_center(image_stack, cropx, cropy):
    image_stack_crop = []
    for i in range(len(image_stack)): 
        image = image_stack[i]
        x, y, z = image.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        image_stack_crop.append(image[starty:starty+cropy, startx:startx+cropx, z-1])
    return np.expand_dims(image_stack_crop, axis = 3)



