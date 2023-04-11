from quickdraw import *
import random as random
from PIL import Image, ImageDraw, ImageOps
import ipyplot
import numpy as np

global max_drawings
max_drawings = 500
global qd
qd = QuickDrawData(max_drawings = max_drawings)

def random_image(seed = None):
    random.seed(seed)
    random_image = qd.get_drawing(name = qd.drawing_names[random.randint(0, len(qd.drawing_names))], index = random.randint(0, qd._max_drawings))
    return random_image

def random_names(n, seed = None):
    random.seed(seed)
    name_list = []
    name_list = np.random.choice(qd.drawing_names, n, replace = False).tolist()
    # for i in range(n):
    #     name_list.append(qd.drawing_names[random.randint(0, len(qd.drawing_names))])
    return name_list

def image_list(name):
    image_list = []
    qg = qd.get_drawing_group(name)
    for i in range(qd._max_drawings):
        image_list.append(qg.get_drawing(index = i))
    return image_list

def draw_image_list(l):
    image_list = []
    for i in l:
        image_list.append(draw_image(i))
    return image_list

def plot_image_list(l, w = 2):
    ipyplot.plot_images(images = [draw_image(i, w = w) for i in l])

def draw_images_name(name):
    return draw_image_list(image_list(name))

def plot_images_name(name, max_images = 9, img_width = 256):
    ipyplot.plot_images(images = draw_image_list(image_list(name)), max_images = max_images, img_width = img_width)

def plot_image_names_tabs(names, max_imgs_per_tab = 9, img_width = 256):
    images = []
    labels = []
    for name in names:
        images += draw_images(image_list(name))
        labels += [name for i in range(qd._max_drawings)]
    ipyplot.plot_class_tabs(images = images, labels = labels, max_imgs_per_tab = max_imgs_per_tab, img_width = img_width)

def plot_random_image_names_tabs(n, seed = None):
    name_list = random_names(n, seed = seed)
    plot_tabs(l = name_list)

def plot_image_names_representations(names, img_width = 256):
    images = []
    labels = []
    for name in names:
        images += draw_image_list(image_list(name))
        labels += [name for i in range(qd._max_drawings)]
    ipyplot.plot_class_representations(images = images, labels = labels, img_width = img_width)

def image_dict_names(names):
    image_dict = dict()
    for name in names:
        image_dict[name] = image_list(name)
    return image_dict

def plot_image_dict_tabs(d, max_imgs_per_tab = 9, img_width = 256):
    images = []
    labels = []
    for name in d:
        images += draw_image_list(d[name])
        labels += [name for i in range(len(d[name]))]
    ipyplot.plot_class_tabs(images = images, labels = labels, max_imgs_per_tab = max_imgs_per_tab, img_width = img_width)

def print_image_strokes(i):
    for stroke in i.strokes:
        for x, y in stroke:
            print("x={} y={}".format(x, y)) 

def draw_image(i, w = 2):
    image = Image.new('L', (256, 256), color = 'white')
    draw = ImageDraw.Draw(image)

    for stroke in i.strokes:
        x0 = -1
        y0 = -1
        for x, y in stroke:
            if (x0 == -1) & (y0 == -1):
                pass
            else:
                draw.line((x0, y0, x, y), 0, width = w)
            x0 = x
            y0 = y

    return image

def plot_image(i, w = 2):
    ipyplot.plot_images(images = [draw_image(i, w = w)])

def draw_image_stroke(i, s, w = 2):
    image = Image.new('L', (256, 256), color = 'white')
    draw = ImageDraw.Draw(image)

    if s not in range(i.no_of_strokes):
        raise ValueError("Stroke index out of range.")
        return

    x0 = -1
    y0 = -1
    for x, y in i.strokes[s]:
        if (x0 == -1) & (y0 == -1):
            pass
        else:
            draw.line((x0, y0, x, y), 0, width = w)
        x0 = x
        y0 = y

    return image

def plot_image_stroke(i, s, w = 2):
    ipyplot.plot_images(images = [draw_image_stroke(i, s, w = w)])

def draw_image_strokes(i, s, w = 2):
    image = Image.new('L', (256, 256), color = 'white')
    draw = ImageDraw.Draw(image)

    for stroke in s:
        if stroke not in range(i.no_of_strokes):
            raise ValueError("Stroke index out of range.")
            return

    for stroke in [i.strokes[index] for index in s]:
        x0 = -1
        y0 = -1
        for x, y in stroke:
            if (x0 == -1) & (y0 == -1):
                pass
            else:
                draw.line((x0, y0, x, y), 0, width = w)
            x0 = x
            y0 = y

    return image

def plot_image_strokes(i, s, w = 2):
    ipyplot.plot_images(images = [draw_image_strokes(i, s, w = w)])