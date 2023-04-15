# Sketch2Image

## Overview:

https://user-images.githubusercontent.com/89194387/232005474-7d55a293-f1cd-4817-a338-37eeed4dc0e0.mp4

https://user-images.githubusercontent.com/89194387/232189846-e96139be-f660-4d10-b718-e40b8366076d.mp4

(From [user_interface_google_colab](https://github.com/GreeneryScenery/Sketch2Image/blob/main/user_interface_google_colab.ipynb))

## Instructions:
1. Open user_interface.ipynb.
2. Enter your Replicate API token.
3. The models should be available online or automatically download.
4. If the models do not successfully download, download them [here](https://huggingface.co/GreeneryScenery/Sketch2ImageModels) and place them in the same directory as the other folders and files.
5. Run user_interface.ipynb.
6. The user interface should open. It may take a while downloading the files. Enjoy!

## Objective:
Sketch to image in one click.

## Applications / Usages:
- Allow more control over the composition of the generated image
- Enable everyone to generate beautiful images
- As not everyone is able to draw well, a sketch makes the tool more accessible

## Methodology:
### Primary:
- Sketch -> CNN (Classify image) -> GPT-2 (Generate prompt) -> ControlNet (Generate image)
### Others:
- OpenAssistant (Generate prompt)
- MagicPrompt-Stable-Diffusion (Generate prompt)
- Stable-Diffusion-img2img (Generate image)
- ControlNet-Scribble (Generate image)
- Sketch-Guided-Stable-Diffusion (Generate edge map)

## Models:
### Trained:
- CNN
- Fine-tuned GPT-2 (https://github.com/minimaxir/gpt-2-simple)
- Fine-tuned Stable Diffusion (ControlNet) (https://huggingface.co/GreeneryScenery/SheepsControlV4)
### Borrowed:
- OpenAssistant (Hugging Face) (https://huggingface.co/OpenAssistant/oasst-sft-1-pythia-12b)
- MagicPrompt-Stable-Diffusion (Hugging face) (https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion)
- Stable-Diffusion-img2img (Replicate) (https://replicate.com/stability-ai/stable-diffusion-img2img)
- ControlNet-Scribble (Replicate) (https://replicate.com/jagilley/controlnet-scribble)
- Sketch-Guided-Stable-Diffusion (GitHub) (https://github.com/ogkalu2/Sketch-Guided-Stable-Diffusion)

## Datasets:
- QuickDraw-Dataset (GitHub) (https://github.com/googlecreativelab/quickdraw-dataset) -> CNN
- SketchyCOCO (GitHub) (https://github.com/sysu-imsl/SketchyCOCO) -> ControlNet
- Stable-Diffusion-Prompts (Hugging Face) (https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts) -> GPT-2

## Link to models and datasets:
### Models:
- [Sketch2ImageModels](https://huggingface.co/GreeneryScenery/Sketch2ImageModels)
- [SheepsControlV1](https://huggingface.co/GreeneryScenery/SheepsControlV1)
- [SheepsControlV2](https://huggingface.co/GreeneryScenery/SheepsControlV2)
- [SheepsControlV3](https://huggingface.co/GreeneryScenery/SheepsControlV3) [SheepsControlV3 on Replicate](https://replicate.com/greeneryscenery/sheeps-control-v3)
- [SheepsControlV4](https://huggingface.co/GreeneryScenery/SheepsControlV4) [SheepsControlV4 on Replicate](https://replicate.com/greeneryscenery/sheeps-control-v4)
- [SheepsControlV5](https://huggingface.co/GreeneryScenery/SheepsControlV5) [SheepsControlV5 on Replicate](https://replicate.com/greeneryscenery/sheeps-controlnet-sketch-2-image)
### Datasets:
- [SheepsNet](https://huggingface.co/datasets/GreeneryScenery/SheepsNet)
- [SheepsNetV2](https://huggingface.co/datasets/GreeneryScenery/SheepsNetV2)
- [SheepsCanny](https://huggingface.co/datasets/GreeneryScenery/SheepsCanny)

## Examples:
### ControlNet-Scribble:
- Conditioning image:
- Prompt: Turtle, happy, highly detailed, artgerm, artstation, concept art, matte, sharp, focus, art by WLOP and James Jean and Victo Ngai
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/TurtlehappyhighlydetailedartgermartstationconceptartmattesharpfocusartbyWLOPandJamesJeanandVictoNgai2_input.png' style = 'width: 256px'>

- Image:
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/TurtlehappyhighlydetailedartgermartstationconceptartmattesharpfocusartbyWLOPandJamesJeanandVictoNgai1_output.png' style = 'width: 256px'>

### SheepsControlV3:
- Prompt: Sheep, trending, on artstation, artstation, HD, artstationHQ, patreon, 4k, 8k
- Conditioning image: Anything :(
- Image:
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/SheeptrendingonartstationartstationHDartstationHQpatreon4k8k_output.png' style = 'width: 256px'>

### SheepsControlV4
<img src = 'https://huggingface.co/GreeneryScenery/SheepsControlV4/resolve/main/overview.png'>

- Prompt: Cute turtle

- Conditioning image:
<img src = 'https://huggingface.co/GreeneryScenery/SheepsControlV4/resolve/main/turtle.png' style = 'width: 256px'>

- Image:
<img src = 'https://huggingface.co/GreeneryScenery/SheepsControlV4/resolve/main/Cute turtle.png' style = 'width: 256px'>

### SheepsControlV5
[Validation report](https://wandb.ai/cs-project/controlnet/reports/SheepsControlV5--Vmlldzo0MDcyNTU5?accessToken=2u04hzf2ud525fkqyji3rrpaatolgab3vfwa2pow3lcgbouhbz4jghoiblc8uvyj)

<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/9.png'>

### UI:
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/1.png' style = 'width: 512px'>
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/2.png' style = 'width: 512px'>
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/3.png' style = 'width: 512px'>
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/4.png' style = 'width: 512px'>
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/5.png' style = 'width: 512px'>
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/6.png' style = 'width: 512px'>
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/7.png' style = 'width: 512px'>
<img src = 'https://github.com/GreeneryScenery/Sketch2Image/blob/main/examples/8.png' style = 'width: 512px'>

## Limitations:
- A few cents to generate each image
- CNN limited to 30 classes, and does not coincide with the ControlNet model
- Prompt generator does not refer to the sketch

## Improvements:
- Better UI
- Train CNN to recognise more classes / train CNN on SketchyCOCO
- Train ControlNet on more epochs, more data, and better prompts
  - May use LAION or ImageNet dataset
- Use image to text models such as blip-image-captioning-large (https://huggingface.co/Salesforce/blip-image-captioning-large)

## Links:
- [Hugging Face](https://huggingface.co/GreeneryScenery)
- [Replicate](https://replicate.com/greeneryscenery)

## References / Resources:
- https://github.com/huggingface/diffusers/tree/main/examples/controlnet (ControlNet)
- https://huggingface.co/blog/train-your-controlnet (ControlNet)
- https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/controlnet (ControlNet)
- https://mikexuq.github.io/test_building_pages/index.html (SketchyCOCO)
- https://www.youtube.com/watch?v=AALBGpLbj6Q (GAN)
- https://www.youtube.com/watch?v=jztwpsIzEGc (CNN)
- https://quickdraw.readthedocs.io/en/latest/ (QuickDraw)
- http://www.michieldb.nl/other/cursors/ (Posy’s Cursor)
- https://replicate.com/docs/guides/get-a-gpu-machine (Replicate)
- https://replicate.com/docs/guides/push-a-model (Replicate)
- https://lambdalabs.com/blog/set-up-a-tensorflow-gpu-docker-container-using-lambda-stack-dockerfile (Docker in Lambda Labs)
- https://github.com/replicate/cog/blob/main/docs/python.md (Cog for Replicate)
