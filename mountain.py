from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image 
from segment import segmentation


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "ksyint/mountain_landscape",
    torch_dtype=torch.float16,
)
pipe.to("cuda")

image=Image.open("input.png")
image,mask_image=segmentation(image)

prompt="\u003Cmountain-landscape> advertisement"

negative_prompt="worst quality, letters or words are written on the background, english is displayed, shoe is displayed"

image = pipe(prompt=prompt,negative_prompt=negative_prompt, image=image, mask_image=mask_image).images[0]
image.save("output.png")
