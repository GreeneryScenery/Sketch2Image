# Generate QuickDraw dataset for training

from quickdraw_class import *
import tensorflow as tf
from scipy.ndimage import rotate, zoom
from skimage.transform import resize

global max_drawings
max_drawings = 200
global resize_size
resize_size = (128, 128)
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

    # Randomly scale image between 0.7 and 1.1
    scale = random.uniform(0.7, 1.1)
    output_shape = np.array(image.shape) * scale
    zoom_factor = output_shape / np.array(image.shape)
    image = zoom(image, zoom_factor, cval = 1)

    image = np.clip(image, 0, 1)
    image = resize(image, resize_size, anti_aliasing=True)

    if random.randint(0, 1) == 1:
        image = np.fliplr(image)

    image = np.stack((image,)*3, axis=-1)
    
    return image

def subprocess_image(image):
    image = np.array(image)
    image = resize(image, resize_size, anti_aliasing=True)

    return image

def resize_image(image):
    resized_image = ImageOps.fit(image, resize_size, Image.ANTIALIAS)
    return resized_image

def normalize_image(image):
    image = image / 255.0
    return image

def unprocess_array(array):
    array *= 255.0
    image = array_to_image(array)
    return image

def array_to_image(array):
    array = np.rint(array).astype(np.uint8)
    image = Image.fromarray(array, 'RGB')
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