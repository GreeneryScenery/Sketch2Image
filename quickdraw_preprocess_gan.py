# Generate QuickDraw dataset for training

from quickdraw_class import *
import tensorflow as tf
from scipy.ndimage import rotate, zoom
from skimage.transform import resize
import math

global max_drawings
max_drawings = 500
global resize_size
resize_size = (64, 64)
global qd
qd = QuickDrawData(max_drawings = max_drawings)

def preprocess_image(image):
    image = draw_image(image)
    image = resize_image(image)
    image = np.array(image)
    image = normalize_image(image)

    # Randomly rotate image between -10 and 10 degrees
    angle = random.uniform(-10, 10)
    image = rotate(image, angle, reshape=False, cval = 1)

    # Randomly scale image between 0.5 and 0.9
    scale = random.uniform(0.95, 1.0)
    output_shape = np.array(image.shape) * scale
    zoom_factor = output_shape / np.array(image.shape)
    image = zoom(image, zoom_factor, cval = 1)

    image = np.clip(image, 0, 1)

    if random.randint(0, 1) == 1:
        image = np.fliplr(image)
    
    image += 1
    image[np.round(image.copy(), 2) == 2.0] = 0
    image = crop3(image)
    image[image == 0.0] = 1
    image -= 1
    image[image == 0.0] = 1
    image = np.pad(image, pad_width=((math.ceil(((resize_size[0]) - image.shape[0])/2.0),math.floor(((resize_size[0]) - image.shape[0])/2.0)), (math.ceil(((resize_size[1]) - image.shape[1])/2.0),math.floor(((resize_size[1]) - image.shape[1])/2.0))), mode='constant', constant_values=1)

    image = np.clip(image, 0, 1)

    return image

def _fill_gap(a):
    a[slice(*a.nonzero()[0].take([0,-1]))] = True
    return a

def crop3(d, clip=True):
    dat = np.array(d)
   # if clip: np.clip(dat, 0, 1, out=dat)
    dat = np.compress(_fill_gap(dat.any(axis=0)), dat, axis=1)
    dat = np.compress(_fill_gap(dat.any(axis=1)), dat, axis=0)
    return dat

def subprocess_image(image):
    image = np.array(image)
    image = resize(image, resize_size, anti_aliasing=True)

    return image

def resize_image(image):
    resized_image = ImageOps.fit(image, resize_size, Image.LANCZOS)
    return resized_image

def normalize_image(image):
    image = image / 255.0
    return image

def unprocess_array(array):
    array = array.copy()
    array *= 255.0
    image = array_to_image(array)
    return image

def array_to_image(array):
    array = np.rint(array).astype(np.uint8)
    image = Image.fromarray(array, 'L')
    return image

def image_dict_to_arrays(d):
    examples = []
    labels = []
    for name in d:
        for image in d[name]:
            examples.append(np.array(draw_image(image)))
            labels.append(name)
    examples = np.array(examples)
    labels = np.array(labels)
    return examples, labels

def preprocess_image_dict_to_arrays(d):
    examples = []
    labels = []
    for name in d:
        for image in d[name]:
            examples.append(preprocess_image(image))
            labels.append(name)
    examples = np.array(examples)
    labels = np.array(labels)
    return examples, labels