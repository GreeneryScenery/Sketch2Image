# Sketch2Image

## Objective:
Sketch to image in one click.

## Methodology:
### Primary:
- Sketch -> CNN (Classify image) -> GPT-2 (Generate prompt) -> ControlNet (Generate image)
### Others:
- OpenAssistant (Generate prompt)
- MagicPrompt-Stable-Diffusion (Generate prompt)
- Stable-Diffusion-img2img (Generate image)
- ControlNet-Scribble (Generate image)
- EdgeGAN (Generate edge map)

## Models:
### Trained:
- CNN
- Fine-tuned GPT-2 (https://github.com/minimaxir/gpt-2-simple)
- Fine-tuned Stable Diffusion (ControlNet) (https://huggingface.co/GreeneryScenery/SheepsControlV3)
### Borrowed:
- OpenAssistant (Hugging Face) (https://huggingface.co/OpenAssistant/oasst-sft-1-pythia-12b)
- MagicPrompt-Stable-Diffusion (Hugging face) (https://huggingface.co/Gustavosta/MagicPrompt-Stable-Diffusion)
- Stable-Diffusion-img2img (Replicate) (https://replicate.com/stability-ai/stable-diffusion-img2img)
- ControlNet-Scribble (Replicate) (https://replicate.com/jagilley/controlnet-scribble)
- EdgeGAN (GitHub) (https://github.com/sysu-imsl/EdgeGAN)

## Datasets:
- QuickDraw-Dataset (GitHub) (https://github.com/googlecreativelab/quickdraw-dataset) -> CNN
- SketchyCOCO (GitHub) (https://github.com/sysu-imsl/SketchyCOCO) -> ControlNet
- Stable-Diffusion-Prompts (Hugging Face) (https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts) -> GPT-2

## Link to models and datasets:
### Models:
- [SheepsControl](https://huggingface.co/GreeneryScenery/SheepsControl)
- [SheepsControlV2](https://huggingface.co/GreeneryScenery/SheepsControlV2)
- [SheepsControlV3](https://huggingface.co/GreeneryScenery/SheepsControlV3) [SheepsControlV3 on Replicate](https://replicate.com/greeneryscenery/sheeps-control-v3)
- [SheepsControlV4 (Under training)](https://huggingface.co/GreeneryScenery/SheepsControlV4)
### Datasets:
- [SheepsNet](https://huggingface.co/datasets/GreeneryScenery/SheepsNet)
- [SheepsNetV2](https://huggingface.co/datasets/GreeneryScenery/SheepsNetV2)
- [SheepsCanny](https://huggingface.co/datasets/GreeneryScenery/SheepsCanny)

## Limitations:
- A few cents to generate each image
- CNN limited to 30 classes, and does not coincide with the ControlNet model
- Prompt generator does not refer to the sketch
- Image generator (SheepsNetV3) does not refer to the sketch

## Improvements:
- Better UI
- Train CNN to recognise more classes / train CNN on SketchyCOCO
- Train ControlNet on more epochs, more data, and better prompts
  - May use LAION or ImageNet dataset
- Use image to text models such as blip-image-captioning-large (https://huggingface.co/Salesforce/blip-image-captioning-large)

## Links:
- [Hugging Face](https://huggingface.co/GreeneryScenery)
- [Replicate](https://replicate.com/greeneryscenery)

## References / resources:
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
