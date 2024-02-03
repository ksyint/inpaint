from PIL import Image
import numpy as np

def padding(img0):
    img0 = Image.fromarray(img0)

    width, height = img0.size

    new_width = int(width * 1.3)
    new_height = int(height * 1.3)

    new_img = Image.new("RGB", (new_width, new_height), "white")

    paste_x = int((new_width - width) / 2)
    paste_y = int((new_height - height) / 2)

    new_img.paste(img0, (paste_x, paste_y))

    result_image = np.array(new_img)
    return result_image


