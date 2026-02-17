import os
import random
import string
from PIL import ImageFilter, ImageFont, ImageDraw, Image


OUT_DIR = "Captcha"
os.makedirs(OUT_DIR, exist_ok=True)


WIDTH, HEIGHT = 200, 100
N_SAMPLES = 10
CHARS = string.ascii_lowercase + string.digits
FONT_SIZE = 40


font = ImageFont.truetype("arial.ttf", FONT_SIZE)

def generate_captcha(text):
    
    img = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (WIDTH - text_w) // 2
    y = (HEIGHT - text_h) // 2

    
    draw.text((x, y), text, font=font, fill=(0, 0, 0))

   
    text_mask = Image.new("L", (WIDTH, HEIGHT), 255)  
    mask_draw = ImageDraw.Draw(text_mask)
    mask_draw.text((x, y), text, font=font, fill=0)

    return img


for i in range(N_SAMPLES):
    text = ''.join(random.choices(CHARS, k=3))
    img = generate_captcha(text)
    img.save(os.path.join(OUT_DIR, f"{text}_{i:02d}.png"))

print(f"{N_SAMPLES} samples '{OUT_DIR}'")
