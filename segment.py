import numpy as np
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from huggingface_hub import hf_hub_download

def segmentation(img0):

    image = np.array(img0)
    max_y=image.shape[0]
    max_x=image.shape[1]

    chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
    sam_checkpoint = chkpt_path 
    model_type = "vit_b"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)

    from PIL import Image 
    for i in range(len(masks)):
        
        if 4>masks[i]["bbox"][0]>-1 and 4>masks[i]["bbox"][1]>-1 and  max_x+3>masks[i]["bbox"][2]>max_x-2  and max_y+3>masks[i]["bbox"][3]>max_y-2:
            out=Image.fromarray(masks[i]['segmentation'])
    

    return img0,out
    
    
    
