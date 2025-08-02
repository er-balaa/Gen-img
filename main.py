from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from IPython.display import display

model_id = "runwayml/stable-diffusion-v1-5"  
pipe = StableDiffusionPipeline.from_pretrained(model_id)


if torch.cuda.is_available():
    pipe = pipe.to("cuda")


prompt = input("Enter the Prompt here : ")


image = pipe(prompt).images[0]

display(image)
