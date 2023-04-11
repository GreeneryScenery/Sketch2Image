#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from PIL import Image, ImageDraw, ImageTk
from tensorflow.keras.models import load_model
from quickdraw_preprocess import *
from sklearn.preprocessing import LabelBinarizer
import io
import torch
import os
import replicate
import requests


# In[2]:


number_of_names = 30
number_of_drawings = max_drawings


# In[3]:


# random_names = random_names(number_of_names, seed = 2)
random_names = ['basket',
 'cruise ship',
 'line',
 'lighthouse',
 'horse',
 'calendar',
 'lion',
 'eyeglasses',
 'eye',
 'leg',
 'streetlight',
 'police car',
 'tornado',
 'sheep',
 'beard',
 'peanut',
 'grass',
 'lantern',
 'circle',
 'dresser',
 'hat',
 'bathtub',
 'cannon',
 'megaphone',
 'rifle',
 'fence',
 'fan',
 'stereo',
 'bat',
 'canoe']
random_names


# In[4]:


encoder = LabelBinarizer()
encoder.fit_transform(random_names)


# In[5]:


def decode(yhat):
    decoded_predictions = []
    for i in range(len(yhat[0])):
        array = np.zeros(len(yhat[0]))
        array[i] = 1
        label = encoder.inverse_transform(np.expand_dims(array, 0))[0]
        decoded_predictions.append((label, yhat[0][i] * 100))

    text = ''
    for pred in decoded_predictions:
        text += '{}: {:.2f}%'.format(pred[0], pred[1]) + '\n'
    return text


# In[6]:

# TODO: Add your API token here
os.environ["REPLICATE_API_TOKEN"] = ""

model_stable_diffusion_img2img = replicate.models.get("stability-ai/stable-diffusion-img2img")
version_stable_diffusion_img2img = model_stable_diffusion_img2img.versions.get("15a3689ee13b0d2616e98820eca31d4c3abcd36672df6afce5cb6feb1d66087d")


# In[7]:


def diffuser(prompt, image, negative_prompt = 'Sketch, Drawing, Ugly, Bad, Badly drawn, Badly drawn art, Badly drawn cartoon, Badly drawn comic, Badly drawn manga, Badly drawn sketch, Badly drawn drawing', prompt_strength = 0.8, num_outputs = 1, num_inference_steps = 25, guidance_scale = 7.5, scheduler = 'DPMSolverMultistep', seed = 1):
    inputs = {
        # Input prompt
        'prompt': prompt,

        # The prompt NOT to guide the image generation. Ignored when not using
        # guidance
        # 'negative_prompt': negative_prompt,

        # Inital image to generate variations of.
        'image': image,

        # Prompt strength when providing the image. 1.0 corresponds to full
        # destruction of information in init image
        'prompt_strength': prompt_strength,

        # Number of images to output. Higher number of outputs may OOM.
        # Range: 1 to 8
        'num_outputs': num_outputs,

        # Number of denoising steps
        # Range: 1 to 500
        'num_inference_steps': num_inference_steps,

        # Scale for classifier-free guidance
        # Range: 1 to 20
        'guidance_scale': guidance_scale,

        # Choose a scheduler.
        'scheduler': scheduler,

        # Random seed. Leave blank to randomize the seed
        'seed': seed,
    }

    # https://replicate.com/stability-ai/stable-diffusion-img2img/versions/15a3689ee13b0d2616e98820eca31d4c3abcd36672df6afce5cb6feb1d66087d#output-schema
    output = version_stable_diffusion_img2img.predict(**inputs)
    return output


# In[8]:


output_dir = 'outputs'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

input_dir = 'inputs'

if not os.path.exists(input_dir):
    os.mkdir(input_dir)

edge_dir = 'edges'

if not os.path.exists(edge_dir):
    os.mkdir(edge_dir)


# In[9]:


def save_output(response, name):
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '', name)
    name = name[:100]
    filename = f'{output_dir}/{name}_output.png'
    if os.path.exists(filename):
        i = 1
        while True:
            new_filename = f'{output_dir}/{name}{i}_output.png'
            if not os.path.exists(new_filename):
                filename = new_filename
                break
            i += 1

    with open(filename, "wb") as f:
        f.write(response.content)

def save_input(image, name):
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '', name)
    name = name[:100]
    filename = f'{input_dir}/{name}_input.png'
    if os.path.exists(filename):
        i = 1
        while True:
            new_filename = f'{input_dir}/{name}{i}_input.png'
            if not os.path.exists(new_filename):
                filename = new_filename
                break
            i += 1

    image.save(filename)
    return filename

def save_edge(image, name):
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '', name)
    name = name[:100]
    filename = f'{edge_dir}/{name}_edge.png'
    if os.path.exists(filename):
        i = 1
        while True:
            new_filename = f'{edge_dir}/{name}{i}_edge.png'
            if not os.path.exists(new_filename):
                filename = new_filename
                break
            i += 1

    image.save(filename)
    return filename


# In[10]:


import argparse
import torch
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn
from typing import List
import math
from tqdm import tqdm
import os

class latent_guidance_predictor(nn.Module):
    def __init__(self, output_dim, input_dim, num_encodings):
        super(latent_guidance_predictor, self).__init__()
        self.num_encodings = num_encodings
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),         
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),     
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),      
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, t):
        # Concatenate input pixels with noise level t and positional encodings
        pos_encoding = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_encodings)]
        pos_encoding = torch.cat(pos_encoding, dim=-1)
        x = torch.cat((x, t, pos_encoding), dim=-1)
        x = x.flatten(start_dim=0, end_dim=2)
        
        return self.layers(x)

@torch.no_grad()
def to_latents(img:Image, vae, device = 'cpu'):
    np_img = (np.array(img).astype(np.float32) / 255.0) * 2.0 - 1.0
    np_img = np_img[None].transpose(0, 3, 1, 2)
    torch_img = torch.from_numpy(np_img)
    generator = torch.Generator(device).manual_seed(0)
    latents = vae.encode(torch_img.to(vae.dtype).to(device)).latent_dist.sample(generator=generator)
    latents = latents * 0.18215
    return latents

@torch.no_grad()
def to_img(latents, vae, device = 'cpu'):
    torch_img = vae.decode(latents.to(vae.dtype).to(device)).sample
    torch_img = (torch_img / 2 + 0.5).clamp(0, 1)
    np_img = torch_img.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
    np_img = (np_img * 255.0).astype(np.uint8)
    img = Image.fromarray(np_img)
    return img

def noisy_latent(image, noise_scheduler, timesteps, device = 'cpu'):
    noise = torch.randn(image.shape).to(device)
    noisy_image = noise_scheduler.add_noise(image, noise, timesteps)
    sqrt_alpha_prod = noise_scheduler.alphas_cumprod[timesteps].to(device) ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(image.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    noise_level = noisy_image - (sqrt_alpha_prod * image)
    return noisy_image, noise_level

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() for f in features if f is not None and isinstance(f, torch.Tensor)]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())

def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out

def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out

def extract_features(latent_image, blocks, unet, timesteps, text_embeddings):
    latent_model_input = torch.cat([latent_image] * 2)
    activations = []
    save_hook = save_out_hook
    feature_blocks = []
    for idx, block in enumerate(unet.down_blocks):
        if idx in blocks:
            block.register_forward_hook(save_hook)
            feature_blocks.append(block) 
            
    for idx, block in enumerate(unet.up_blocks):
        if idx in blocks:
            block.register_forward_hook(save_hook)
            feature_blocks.append(block)  
    with torch.no_grad():
        noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings).sample

    # Extract activations
    for block in feature_blocks:
        activations.append(block.activations)
        block.activations = None
        
    activations = [activations[0][0], activations[1][0], activations[2][0], activations[3][0], activations[4], activations[5], activations[6], activations[7]]
    
    return activations

def resize_and_concatenate(activations: List[torch.Tensor], reference):
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = reference.shape[2:]
    resized_activations = []
    for acts in activations:
        acts = nn.functional.interpolate(
            acts, size=size, mode="bilinear"
        )
        acts = acts[:1]
        acts = acts.transpose(1,3)
        resized_activations.append(acts)
    
    return torch.cat(resized_activations, dim=3)


# In[11]:


def edge_map(caption, vae, device, unet, lgp_path, strength, img_path):
    # parser = argparse.ArgumentParser(description= "Encode images")
    # parser.add_argument("--caption", type=str, help="image caption")
    # parser.add_argument("--vae", type=str, help="folder vae is located", default="runwayml/stable-diffusion-v1-5")
    # parser.add_argument("--device", type=str, help="Device to use", default="cuda", required=False)
    # parser.add_argument("--unet", type=str, help="folder unet subfolder is located", default="runwayml/stable-diffusion-v1-5")
    # parser.add_argument("--LGP_path", type=str, help="folder pre-trained LGP is located")
    # parser.add_argument("--noise_strength", type=float, help="denoising strength")
    # parser.add_argument("--image_path", type=str, help="folder skecth is located")

    # args = parser.parse_args()   
    # device = args.device
    # lgp_path = args.LGP_path
    # img_path = args.image_path

    blocks = [0,1,2,3]
    # caption = args.caption
    num_inference_steps = 50
    batch_size = 1
    guidance_scale = 8
    # strength = args.noise_strength
    eta = 0.0
        
    model = latent_guidance_predictor(output_dim=4, input_dim=7080, num_encodings=9).to(device)
    checkpoint = torch.load(lgp_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    vae = AutoencoderKL.from_pretrained(vae, subfolder= "vae", use_auth_token=False).to(device)
    unet = UNet2DConditionModel.from_pretrained(unet, subfolder="unet", use_auth_token=False).to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    offset = noise_scheduler.config.get("steps_offset", 0)

    noise_scheduler.set_timesteps(num_inference_steps)

    # get the original timestep using init_timestep
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)

    if isinstance(noise_scheduler, LMSDiscreteScheduler):
        timesteps = torch.tensor([num_inference_steps - init_timestep] * batch_size, dtype=torch.long, device=device)
    else:
        timesteps = noise_scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=device)
        
    text_input = tokenizer([caption], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]   
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    img = Image.open(img_path)
    img = img.resize((512,512))
    imagelatent = to_latents(img, vae)

    noisy_image, noise_level = noisy_latent(imagelatent, noise_scheduler, timesteps)
    noise_level = noise_level.transpose(1,3)

    file_name = os.path.basename(img_path)
    img_name = os.path.splitext(file_name)[0]

    features = extract_features(noisy_image, blocks, unet, timesteps, text_embeddings)
    features = resize_and_concatenate(features, imagelatent)

    pred_edge_map = model(features, noise_level).unflatten(0, (1, 64, 64)).transpose(3, 1)
    pred_edge_map = to_img(pred_edge_map, vae)
    save_edge(pred_edge_map, img_name)
    return pred_edge_map
    #pred_edge_map.save(img_name + '-edge_map.jpg')


# In[12]:


model_controlnet_scribble = replicate.models.get("jagilley/controlnet-scribble")
version_controlnet_scribble = model_controlnet_scribble.versions.get("435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117")


# In[13]:


def controlnet(image, prompt, num_samples = '1', image_resolution = '512', ddim_steps = 20, scale = 9, seed = 1, eta = 0, a_prompt = 'best quality, extremely detailed', n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'):
    # https://replicate.com/jagilley/controlnet-scribble/versions/435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117#input
    inputs = {
        # Input image
        'image': image,

        # Prompt for the model
        'prompt': prompt,

        # Number of samples (higher values may OOM)
        'num_samples': num_samples,

        # Image resolution to be generated
        'image_resolution': image_resolution,

        # Steps
        'ddim_steps': ddim_steps,

        # Guidance Scale
        # Range: 0.1 to 30
        'scale': scale,

        # Seed
        'seed': seed,

        # eta (DDIM)
        'eta': eta,

        # Added Prompt
        'a_prompt': a_prompt,

        # Negative Prompt
        'n_prompt': n_prompt,
    }

    # https://replicate.com/jagilley/controlnet-scribble/versions/435061a1b5a4c1e26740464bf786efdfa9cb3a3ac488595a2de23e143fdb0117#output-schema
    output = version_controlnet_scribble.predict(**inputs)
    return output


# In[28]:


model_sheepscontrol = replicate.models.get('greeneryscenery/sheeps-control-v3')
version_sheepscontrol = model_sheepscontrol.versions.get('2c9cdfc4e64142451ad71932a6e513bcc725012598e75d374867f71e2b53ff51')


# In[29]:


def sheepscontrol(image, prompt, seed = 0):
    inputs ={
        'seed': seed,
        'text': prompt,
        'image': image,
    }

    output = version_sheepscontrol.predict(**inputs)
    return output


# In[15]:


def separate_images(image):
    img = image.copy()
    _, binary_img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append((x, y, w, h))
    for rectangle in rectangles:
        x, y, w, h = rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #ipyplot.plot_images([img])

    cropped_images = []
    for rectangle in rectangles:
        x, y, w, h = rectangle
        cropped_img = image[y:y+h, x:x+w]
        cropped_images.append(cropped_img)
    
    if len(cropped_images) == 0:
        return None

    min_size = 10

    i = 0
    while i < len(cropped_images):
        x1, y1, w1, h1 = rectangles[i]
        if w1 < min_size or h1 < min_size:
            del cropped_images[i]
            del rectangles[i]
            continue
        j = i + 1
        while j < len(cropped_images):
            x2, y2, w2, h2 = rectangles[j]
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            if overlap_area > 0:
                area1 = w1 * h1
                area2 = w2 * h2
                if area1 >= area2:
                    del cropped_images[j]
                    del rectangles[j]
                else:
                    del cropped_images[i]
                    del rectangles[i]
                    i -= 1
                    break
            j += 1
        i += 1
    #ipyplot.plot_images(cropped_images)
    return cropped_images


# In[16]:


from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer_magic_prompt = AutoTokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
model_magic_prompt = AutoModelForCausalLM.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")


# In[17]:


regex_magic_prompt = "[\[\]']+"


# In[18]:


def magic_prompt_generator(string):
    input_magic_prompt = tokenizer_magic_prompt.encode(string, return_tensors='pt')
    output_magic_prompt = model_magic_prompt.generate(input_magic_prompt, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.5, num_return_sequences=1)
    text_magic_prompt = tokenizer_magic_prompt.decode(output_magic_prompt[0], skip_special_tokens=True)
    return text_magic_prompt


# In[19]:


format_open_assistant = '<|prompter|>Please form a sentence describing an image using these words: |INPUT|. This sentence will be used for image generation in ControlNet and Stable Diffusion so be as descriptive and creative as possible, while not adding too much extra stuff.<|endoftext|><|assistant|>'


# In[20]:


API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-1-pythia-12b"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


# In[21]:


def open_assistant_generator(string):
    text_open_assistant = format_open_assistant.replace('|INPUT|', string)
    # input_open_assistant = tokenizer_open_assistant.encode(text_open_assistant, return_tensors='pt')
    # output_open_assistant = model_open_assistant.generate(input_open_assistant, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1)
    # text_open_assistant = tokenizer_open_assistant.decode(output_open_assistant[0], skip_special_tokens=True)
    output = query({
        "inputs": text_open_assistant,
    })
    output = output[0]['generated_text']
    return output.replace(text_open_assistant, '')


# In[22]:


def create_prompt(string):
    prompt = open_assistant_generator(string)
    string = re.sub(rf'{regex_magic_prompt}', '', string)
    string = f'{string},'
    magic_prompt = magic_prompt_generator(string)
    prompt = prompt + ' ' + magic_prompt
    return prompt


# In[23]:


import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

def gpt_2_generator(string):
    prompt = gpt2.generate(sess, length=50, temperature=0.7, prefix=string, return_as_list=True)[0]
    prompt = prompt.split('\n')[0]
    return prompt


# In[24]:


cnn_model = load_model('models/cnn_model_4.h5')


# In[32]:


import tkinter
import customtkinter
from tkinter import colorchooser, simpledialog, filedialog, ttk, Frame, PhotoImage
import cv2
import re

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

auto_text = True
pen_color = 'black'
pen_width = 5

canvas_width = 800
canvas_height = 800
canvas = None
img = Image.new('RGB', (canvas_width, canvas_height), color='white')
draw = ImageDraw.Draw(img)

def update_canvas():
    canvas.image = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor='nw', image=canvas.image)

class LineDrawer:
    def __init__(self, image, canvas, app):
        self.draw = ImageDraw.Draw(image)
        self.prev_x = None
        self.prev_y = None
        self.strokes = []
        self.buffer = []
        self.canvas = canvas
        self.image = image
        self.app = app
        self.current_stroke = dict()

    def start_line(self, event):
        self.current_stroke = dict()
        self.buffer = []
        self.prev_x = event.x
        self.prev_y = event.y
        self.current_stroke['color'] = pen_color
        self.current_stroke['width'] = pen_width
        self.current_stroke['points'] = [(self.prev_x, self.prev_y)]
        update_canvas()

    def continue_line(self, event):
        if self.prev_x is not None and self.prev_y is not None:
            x, y = event.x, event.y
            self.draw.line([(self.prev_x, self.prev_y), (x, y)],
                        width = pen_width, fill = pen_color)
            self.prev_x, self.prev_y = x, y
            self.current_stroke['points'].append((self.prev_x, self.prev_y))
        update_canvas()

    def end_line(self, event):
        self.prev_x, self.prev_y = None, None
        update_canvas()
        self.app.classify_image(event)
        self.strokes.append(self.current_stroke)
    
    def start_erase(self, event):
        self.current_stroke = dict()
        self.buffer = []
        self.prev_x = event.x
        self.prev_y = event.y
        self.current_stroke['color'] = 'white'
        self.current_stroke['width'] = pen_width
        self.current_stroke['points'] = [(self.prev_x, self.prev_y)]
        update_canvas()

    def continue_erase(self, event):
        if self.prev_x is not None and self.prev_y is not None:
            x, y = event.x, event.y
            self.draw.line([(self.prev_x, self.prev_y), (x, y)],
                        width = pen_width, fill = 'white')
            self.prev_x, self.prev_y = x, y
            self.current_stroke['points'].append((self.prev_x, self.prev_y))
        update_canvas()

    def end_erase(self, event):
        self.prev_x, self.prev_y = None, None
        update_canvas()
        self.app.classify_image(event)
        self.strokes.append(self.current_stroke)

    def get_strokes(self):
        return self.strokes
    
    def undo(self, event):
        if self.strokes != []:
            self.draw.rectangle([(0, 0), (canvas_width, canvas_height)], fill='white')
            for stroke in self.strokes[:-1]:
                self.draw.line(stroke['points'], width = stroke['width'], fill = stroke['color'])
            update_canvas()
            self.buffer.append(self.strokes[-1])
            self.strokes = self.strokes[:-1]
            self.app.classify_image(event)
    
    def redo(self, event):
        if self.buffer != []:
            self.draw.rectangle([(0, 0), (canvas_width, canvas_height)], fill='white')
            for stroke in self.strokes:
                self.draw.line(stroke['points'], width = stroke['width'], fill = stroke['color'])
            self.draw.line(self.buffer[-1]['points'], width = self.buffer[-1]['width'], fill = self.buffer[-1]['color'])
            self.strokes.append(self.buffer[-1])
            self.buffer = self.buffer[:-1]
            update_canvas()
            self.app.classify_image(event)
    
    def clear(self, event):
        self.draw.rectangle([(0, 0), (canvas_width, canvas_height)], fill='white')
        self.strokes = []
        self.buffer = []
        update_canvas()
        self.app.classify_image(event)

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        self.crop = IntVar()
        self.separate = IntVar()
        self.magic = customtkinter.StringVar(value="on")
        self.open = customtkinter.StringVar(value="on")

        # configure window
        self.title("Sketcher")
        self.geometry(f"{1920}x{1080}")
        self.iconbitmap('assets/Sketcher.ico')

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure((0, 1), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Sketcher", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.theme_label = customtkinter.CTkLabel(self.sidebar_frame, text="Theme:", anchor="w")
        self.theme_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.theme_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["blue", "dark-blue", "green"],
                                                                       command=self.change_theme_event)
        self.theme_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        self.prompt_entry = customtkinter.CTkEntry(self, placeholder_text="Prompt goes here...")
        self.prompt_entry.grid(row=2, column=1, columnspan=2, padx=(20, 20), pady=(0, 20), sticky="nsew")
        self.prompt_entry.bind("<KeyRelease>", self.prompt_changed)

        # create canvas
        global canvas
        canvas = Canvas(master=self, width=canvas_width, height=canvas_height, bg='white', cursor = "@Posys_Cursor_Strokeless/Posy_pen.cur")
        canvas.grid(row=0, column=1, padx=(20, 0), pady=(40, 0), sticky='n')

        self.bind('<Control-z>', self.undo)
        self.bind('<Control-Z>', self.redo)

        self.line_drawer = LineDrawer(img, canvas, self)
        canvas.bind('<Button-1>', self.line_drawer.start_line)
        canvas.bind('<B1-Motion>', self.line_drawer.continue_line)
        canvas.bind('<ButtonRelease-1>', self.line_drawer.end_line)
        canvas.bind("<Button-3>", self.line_drawer.start_erase)
        canvas.bind("<B3-Motion>", self.line_drawer.continue_erase)
        canvas.bind("<ButtonRelease-3>", self.line_drawer.end_erase)

        # create tabview
        self.tabview = customtkinter.CTkTabview(self, width=400)
        self.tabview.grid(row=0, column=2, padx=(20, 20), pady=(20, 10), sticky="nsew")
        self.tabview.add("Predictions")
        self.tabview.add("Generations")
        self.tabview.add("Settings")
        self.tabview.tab("Predictions").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Generations").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Settings").grid_columnconfigure(0, weight=1)
        
        self.predictions_frame = customtkinter.CTkScrollableFrame(self.tabview.tab("Predictions"), width=350, height=450)
        self.predictions_frame.grid(row=0, column=0, padx=20, pady=20)
        self.predictions_frame.grid_columnconfigure(0, weight=1)

        self.best_label = customtkinter.CTkLabel(self.predictions_frame, text = 'Prediction:', font = ('Helvetica', 24, 'bold'), ) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless.cur')
        self.best_label.grid(row = 0, column = 0, padx = 5, pady = 5)
        self.best_label.configure(wraplength = 300, text_color = 'green')

        self.prediction_label = customtkinter.CTkLabel(self.predictions_frame, text = decode(np.expand_dims(np.zeros(number_of_names), 0)), ) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless.cur')
        self.prediction_label.grid(row=1, column=0, padx=5, pady=5)
        self.prediction_label.configure(wraplength = 200, justify = RIGHT)

        self.generation_frame = customtkinter.CTkScrollableFrame(self.tabview.tab("Generations"), width=350, height=450)
        self.generation_frame.grid(row=0, column=0, padx=20, pady=20)
        self.generation_frame.grid_columnconfigure(0, weight=1)

        self.image_generation_frame = customtkinter.CTkFrame(self.generation_frame)
        self.image_generation_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.image_generation_frame.grid_columnconfigure(0, weight=1)

        self.random_button = customtkinter.CTkButton(self.image_generation_frame, text='Random (💻)', command = self.empty) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless_link.cur')
        self.random_button.grid(row = 0, column = 0, padx = 10, pady = (20, 5))
        self.random_button.bind("<ButtonRelease-1>", self.random_image)

        self.random_label = customtkinter.CTkLabel(self.image_generation_frame, text = '', ) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless.cur')
        self.random_label.grid(row = 1, column = 0, padx = 10, pady = 5)

        self.diffuse_button = customtkinter.CTkButton(self.image_generation_frame, text='Diffuse (₿)', command = self.empty) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless_link.cur')
        self.diffuse_button.grid(row = 2, column = 0, padx = 10, pady = 5)
        self.diffuse_button.bind("<ButtonRelease-1>", self.diffuse)

        self.edge_button = customtkinter.CTkButton(self.image_generation_frame, text='Edge (💻⏳)', command = self.empty) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless_link.cur')
        self.edge_button.grid(row = 3, column = 0, padx = 10, pady = 5)
        self.edge_button.bind("<ButtonRelease-1>", self.edge)

        self.control_button = customtkinter.CTkButton(self.image_generation_frame, text='Control (₿)', command = self.empty) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless_link.cur')
        self.control_button.grid(row = 4, column = 0, padx = 10, pady = 5)
        self.control_button.bind("<ButtonRelease-1>", self.control)

        self.sheeps_button = customtkinter.CTkButton(self.image_generation_frame, text='Sheeps (₿)', command = self.empty)
        self.sheeps_button.grid(row = 5, column = 0, padx = 10, pady = (5, 20))
        self.sheeps_button.bind("<ButtonRelease-1>", self.sheeps)

        self.text_generation_frame = customtkinter.CTkFrame(self.generation_frame)
        self.text_generation_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.text_generation_frame.grid_columnconfigure(0, weight=1)

        self.switch_magic = customtkinter.CTkSwitch(self.text_generation_frame, text = 'Magic Prompt (💻)', variable = self.magic, onvalue = 'on', offvalue = 'off')
        self.switch_magic.grid(row = 0, column = 0, padx = 5, pady = (20, 5))

        self.switch_open = customtkinter.CTkSwitch(self.text_generation_frame, text = 'Open Assistant (🤗)', variable = self.open, onvalue = 'on', offvalue = 'off')
        self.switch_open.grid(row = 1, column = 0, padx = 10, pady = 5)

        self.label_open = customtkinter.CTkLabel(self.text_generation_frame, text = 'Format for Open Assistant (|INPUT| will be replaced by prompt.) :')
        self.label_open.grid(row = 2, column = 0, padx = 10, pady = (10, 0), sticky = 'w')
        self.label_open.configure(wraplength = 250, justify = 'left')

        self.textbox_open = customtkinter.CTkTextbox(self.text_generation_frame, width = 300, height = 200)
        self.textbox_open.grid(row = 3, column = 0, padx = 10, pady = 5)
        self.textbox_open.insert("0.0", format_open_assistant)

        self.generate_prompt_button = customtkinter.CTkButton(self.text_generation_frame, text='Generate Prompt', command=self.generate_prompt)
        self.generate_prompt_button.grid(row = 4, column = 0, padx = 10, pady = 5)

        self.gpt_2_generate_prompt_button = customtkinter.CTkButton(self.text_generation_frame, text='Generate Prompt with GPT-2 (💻)', command=self.gpt_2_generate_prompt)
        self.gpt_2_generate_prompt_button.grid(row = 5, column = 0, padx = 10, pady = (5, 20))

        self.settings_frame = customtkinter.CTkScrollableFrame(self.tabview.tab("Settings"), width=350, height=450)
        self.settings_frame.grid(row=0, column=0, padx=20, pady=20)
        self.settings_frame.grid_columnconfigure(0, weight=1)

        self.color_button = customtkinter.CTkButton(self.settings_frame, text="Change Pen Color", command=self.change_pen_color, ) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless_link.cur')
        self.color_button.grid(row = 0, column = 0, padx = 5, pady = (20, 5))

        self.width_button = customtkinter.CTkButton(self.settings_frame, text="Change Pen Width", command=self.change_pen_width, ) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless_link.cur')
        self.width_button.grid(row = 1, column = 0, padx = 5, pady = 5)

        self.width_slider = customtkinter.CTkSlider(self.settings_frame, from_ = 0, to = 30, command = self.empty) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless_link.cur')
        self.width_slider.set(pen_width)
        self.width_slider.bind("<ButtonRelease-1>", self.change_width_slider)
        self.width_slider.grid(row = 2, column = 0, padx = 5, pady = 5)

        clear = customtkinter.CTkImage(light_image = Image.open('assets/clear_light.png'), dark_image=Image.open('assets/clear_dark.png'), size=(20, 20))
        self.clear_button = customtkinter.CTkButton(self.settings_frame, text='Clear', image = clear, compound = 'left', command = self.empty)
        self.clear_button.grid(row = 3, column = 0, padx = 5, pady = 5)
        self.clear_button.bind("<ButtonRelease-1>", self.clear)

        self.crop_button = customtkinter.CTkCheckBox(self.settings_frame, text = 'Crop', command = self.empty) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless_link.cur')
        self.crop_button.bind("<ButtonRelease-1>", self.crop_switch)
        self.crop_button.grid(row = 4, column = 0, padx = 5, pady = 5)

        self.separate_button = customtkinter.CTkCheckBox(self.settings_frame, text = 'Separate', command = self.empty) # cursor = '@Posys_Cursor_Strokeless/Posy_Strokeless_link.cur')
        self.separate_button.bind("<ButtonRelease-1>", self.separate_switch)
        self.separate_button.grid(row = 5, column = 0, padx = 5, pady = 5)

        reset = customtkinter.CTkImage(light_image = Image.open('assets/reset_light.png'), dark_image=Image.open('assets/reset_dark.png'), size=(20, 20))
        self.reset_button = customtkinter.CTkButton(self.settings_frame, text='Reset', image =  reset, compound = 'left', command = self.empty)
        self.reset_button.grid(row = 6, column = 0, padx = 5, pady = 5)
        self.reset_button.bind("<ButtonRelease-1>", self.reset)

        lock = customtkinter.CTkImage(light_image = Image.open('assets/lock_light.png'), dark_image=Image.open('assets/lock_dark.png'), size=(20, 20))
        self.lock_button = customtkinter.CTkButton(self.settings_frame, text='Lock', image = lock, compound = 'left', command = self.empty)
        self.lock_button.grid(row = 7, column = 0, padx = 5, pady = 5)
        self.lock_button.bind("<ButtonRelease-1>", self.lock)

        upload = customtkinter.CTkImage(light_image = Image.open('assets/upload_light.png'), dark_image=Image.open('assets/upload_dark.png'), size=(20, 20))
        self.upload_button = customtkinter.CTkButton(self.settings_frame, text = 'Upload', command = self.upload, image = upload, compound = 'left')
        self.upload_button.grid(row = 8, column = 0, padx = 5, pady = 5)

        download = customtkinter.CTkImage(light_image = Image.open('assets/download_light.png'), dark_image=Image.open('assets/download_dark.png'), size=(20, 20))
        self.download_button = customtkinter.CTkButton(self.settings_frame, text = 'Download', command = self.download, image = download, compound = 'left')
        self.download_button.grid(row = 9, column = 0, padx = 5, pady = 5)

        undo = customtkinter.CTkImage(light_image = Image.open('assets/undo_light.png'), dark_image=Image.open('assets/undo_dark.png'), size=(20, 20))
        self.undo_button = customtkinter.CTkButton(self.settings_frame, text = 'Undo', image = undo, compound = 'left', command = self.empty)
        self.undo_button.grid(row = 10, column = 0, padx = 5, pady = 5)
        self.undo_button.bind("<ButtonRelease-1>", self.undo)

        redo = customtkinter.CTkImage(light_image = Image.open('assets/redo_light.png'), dark_image=Image.open('assets/redo_dark.png'), size=(20, 20))
        self.redo_button = customtkinter.CTkButton(self.settings_frame, text = 'Redo', image = redo, compound = 'left', command = self.empty)
        self.redo_button.grid(row = 11, column = 0, padx = 5, pady = (5,20))
        self.redo_button.bind("<ButtonRelease-1>", self.redo)

        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
    
    def empty(self):
        pass

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
    
    def change_theme_event(self, new_theme: str):
        customtkinter.set_default_color_theme(new_theme)
        self.destroy()
        app = App()
        app.mainloop()

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
    
    def change_pen_color(self):
        global pen_color
        color = colorchooser.askcolor()[1]
        if color:
            pen_color = color
    
    def change_pen_width(self):
        global pen_width
        width = simpledialog.askinteger("Pen Width", "Enter the new pen width:", initialvalue=pen_width)
        if width:
            pen_width = int(width)
    
    def change_width_slider(self, event):
        global pen_width
        pen_width = int(self.width_slider.get())

    def classify_image(self, event):
        global auto_text
        # Convert the drawn image to a numpy array

        if self.separate.get() != 1:
            image = subprocess_image(np.array(img.convert('L')))
            if self.crop.get() == 1:
                image += 1
                image[np.round(image.copy(), 1) == 2.0] = 0
                image = crop3(image)
                image[image == 0.0] = 1
                image -= 1
                image[image == 0.0] = 1

                width, height = image.shape

                if (width != 0) and (height != 0):
                    if width > height:
                        new_width = 90
                        new_height = int((90/width) * height)
                    else:
                        new_height = 90
                        new_width = int((90/height) * width)

                    image = resize(image, (new_width, new_height), anti_aliasing = True)

                image = np.pad(image, pad_width=((math.ceil(((resize_size[0]) - image.shape[0])/2.0),math.floor(((resize_size[0]) - image.shape[0])/2.0)), (math.ceil(((resize_size[1]) - image.shape[1])/2.0),math.floor(((resize_size[1]) - image.shape[1])/2.0))), mode='constant', constant_values=1)

                image = np.clip(image, 0, 1)
            
            #ipyplot.plot_images([image])

            # Make a prediction with your CNN model
            yhat = cnn_model.predict(np.expand_dims(np.expand_dims(image, 2), 0))
            # Display the prediction
            self.prediction_label.configure(text = decode(yhat))
            self.best_label.configure(text = 'Prediction: ' + encoder.inverse_transform(yhat)[0]) #+ ' (' + str(round(np.max(yhat)*100, 2)) + '%)')
            
            
            if auto_text:
                self.prompt_entry.delete(0, END)
                self.prompt_entry.insert(0, encoder.inverse_transform(yhat)[0])
        else:
            image = np.array(img.convert('L'))
            images = separate_images(image)
            if images == None:
                self.best_label.configure(text = 'Predictions: ')
                if auto_text:
                    self.prompt_entry.delete(0, END)
                    self.prompt_entry.insert(0, '')
                return
            predictions = []
            for image in images:
                width, height = image.shape

                if (width != 0) and (height != 0):
                    if width > height:
                        new_width = 90
                        new_height = int((90/width) * height)
                    else:
                        new_height = 90
                        new_width = int((90/height) * width)

                    image = resize(image, (new_width, new_height), anti_aliasing = True)
                
                image = np.pad(image, pad_width=((math.ceil(((resize_size[0]) - image.shape[0])/2.0),math.floor(((resize_size[0]) - image.shape[0])/2.0)), (math.ceil(((resize_size[1]) - image.shape[1])/2.0),math.floor(((resize_size[1]) - image.shape[1])/2.0))), mode='constant', constant_values=1)
                image = np.clip(image, 0, 1)
                #ipyplot.plot_images([image])
                # image = resize(image, resize_size, anti_aliasing=True)
                yhat = cnn_model.predict(np.expand_dims(np.expand_dims(image, 2), 0))
                predictions.append(encoder.inverse_transform(yhat)[0])
            # prediction_label.config(text = decode(yhat))
            self.best_label.configure(text = 'Predictions: ' + ', '.join(predictions))

            if auto_text:
                self.prompt_entry.delete(0, END)
                self.prompt_entry.insert(0, ', '.join(predictions))
    
    def clear(self, event):
        self.line_drawer.clear(event)    
    
    def random_image(self, event):
        # Get a random name
        random_name = random.choice(random_names)

        # Get a random drawing
        random_image = qd.get_drawing(random_name)

        random_image = unprocess_array(preprocess_image(random_image))

        random_image = ImageOps.fit(random_image, (canvas_width, canvas_height), Image.LANCZOS)

        # canvas.image = ImageTk.PhotoImage(random_image)
        # canvas.create_image(0, 0, anchor='nw', image=canvas.image)
        img.paste(random_image)
        update_canvas()
        self.classify_image(event)
        self.random_label.configure(text = random_name)
    
    def crop_switch(self, event):
        if self.crop.get() == 1:
            self.crop.set(0)
        else:
            self.crop.set(1)
        self.classify_image(event)
    
    def separate_switch(self, event):
        if self.separate.get() == 1:
            self.separate.set(0)
        else:
            self.separate.set(1)
        self.classify_image(event)
    
    def diffuse(self, event):
        global img
        image = subprocess_image(np.array(img.convert('L')))
        yhat = cnn_model.predict(np.expand_dims(np.expand_dims(image, 2), 0))
        prediction = encoder.inverse_transform(yhat)[0]

        resized_image = ImageOps.fit(img, (800, 800), Image.LANCZOS)

        print(self.prompt_entry.get())
        output = diffuser(prompt = self.prompt_entry.get(), image = open(save_input(resized_image, self.prompt_entry.get()), 'rb'))
        url = output[0]
        print(url)

        response = requests.get(url)
        save_output(response, self.prompt_entry.get())
        output_image = Image.open(io.BytesIO(response.content))
        output_image = ImageOps.fit(output_image, (canvas_width, canvas_height), Image.LANCZOS)

        # Draw the image on the canvas
        # canvas.image = ImageTk.PhotoImage(output_image)
        # canvas.create_image(0, 0, anchor='nw', image=canvas.image)

        # set output to img
        img.paste(output_image)

        update_canvas()
    
    def edge(self, event):
        global img
        image = ImageOps.fit(img, (800, 800), Image.LANCZOS)
        edge = edge_map(caption = self.prompt_entry.get(), vae = 'runwayml/stable-diffusion-v1-5', device = 'cpu', unet = 'runwayml/stable-diffusion-v1-5', lgp_path = 'SDv1.5-trained_LGP.pt', strength = 0.3, img_path = save_input(image, self.prompt_entry.get()))

        edge= ImageOps.fit(edge, (canvas_width, canvas_height), Image.LANCZOS)
        # canvas.image = ImageTk.PhotoImage(edge)
        # canvas.create_image(0, 0, anchor='nw', image=canvas.image)
        img.paste(edge)
        update_canvas()
    
    def control(self, event):
        global img
        image = subprocess_image(np.array(img.convert('L')))
        yhat = cnn_model.predict(np.expand_dims(np.expand_dims(image, 2), 0))
        prediction = encoder.inverse_transform(yhat)[0]

        resized_image = ImageOps.fit(img, (800, 800), Image.LANCZOS)

        print(self.prompt_entry.get())
        output = controlnet(image = open(save_input(resized_image, self.prompt_entry.get()), 'rb'), prompt = self.prompt_entry.get())
        url = output[1]
        print(url)

        response = requests.get(url)
        save_output(response, self.prompt_entry.get())
        output_image = Image.open(io.BytesIO(response.content))
        output_image = ImageOps.fit(output_image, (canvas_width, canvas_height), Image.LANCZOS)

        # Draw the image on the canvas
        # canvas.image = ImageTk.PhotoImage(output_image)
        # canvas.create_image(0, 0, anchor='nw', image=canvas.image)

        # set output to img
        img.paste(output_image)
        update_canvas()
    
    def sheeps(self, event):
        global img
        image = subprocess_image(np.array(img.convert('L')))
        yhat = cnn_model.predict(np.expand_dims(np.expand_dims(image, 2), 0))
        prediction = encoder.inverse_transform(yhat)[0]

        resized_image = ImageOps.fit(img, (800, 800), Image.LANCZOS)

        print(self.prompt_entry.get())
        output = sheepscontrol(image = open(save_input(resized_image, self.prompt_entry.get()), 'rb'), prompt = self.prompt_entry.get())
        url = output
        print(url)

        response = requests.get(url)
        save_output(response, self.prompt_entry.get())
        output_image = Image.open(io.BytesIO(response.content))
        output_image = ImageOps.fit(output_image, (canvas_width, canvas_height), Image.LANCZOS)

        # Draw the image on the canvas
        # canvas.image = ImageTk.PhotoImage(output_image)
        # canvas.create_image(0, 0, anchor='nw', image=canvas.image)

        # set output to img
        img.paste(output_image)
        update_canvas()

    def generate_prompt(self):
        if self.prompt_entry.get() != '':
            if (self.magic.get() == 'on') & (self.open.get() == 'on'):
                string = self.prompt_entry.get()
                self.prompt_entry.delete(0, END)
                self.prompt_entry.insert(0, create_prompt(string))
            elif (self.magic.get() == 'on') & (self.open.get() == 'off'):
                string = self.prompt_entry.get()
                string = re.sub(rf'{regex_magic_prompt}', '', string)
                string = f'{string},'
                self.prompt_entry.delete(0, END)
                self.prompt_entry.insert(0, magic_prompt_generator(string))
            elif (self.magic.get() == 'off') & (self.open.get() == 'on'):
                string = self.prompt_entry.get()
                self.prompt_entry.delete(0, END)
                self.prompt_entry.insert(0, open_assistant_generator(string))
    
    def gpt_2_generate_prompt(self):
        string = self.prompt_entry.get()
        string = re.sub(rf'{regex_magic_prompt}', '', string)
        string = f'{string},'
        self.prompt_entry.delete(0, END)
        self.prompt_entry.insert(0, gpt_2_generator(string))

    def prompt_changed(self, event):
        global auto_text
        auto_text = False
        if self.prompt_entry.get() == '':
            auto_text = True
    
    def reset(self, event):
        global auto_text
        auto_text = True
        self.classify_image(event)
    
    def lock(self, event):
        global auto_text
        auto_text = False
    
    def upload(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = ImageOps.fit(image, (canvas_width, canvas_height), Image.LANCZOS)
            # canvas.image = ImageTk.PhotoImage(image)
            # canvas.create_image(0, 0, anchor='nw', image=canvas.image)
            img.paste(image)
            update_canvas()
    
    def download(self):
        file_path = filedialog.asksaveasfilename(defaultextension = [('PNG', '*.png'), ('JPG', '*.jpg')], filetypes = [('PNG', '*.png'), ('JPG', '*.jpg')])
        if file_path:
            img.save(file_path)
    
    def undo(self, event):
        self.line_drawer.undo(event)

    def redo(self, event):
        self.line_drawer.redo(event)
if __name__ == "__main__":
    app = App()
    app.mainloop()

