from PIL import Image, ImageDraw, ImageOps
import random
import math
from pathlib import Path

output_file = Path("data") / "bos_pattern.png"

# Parameters
width, height = 1920,1080
radius = 1                  # size of the dots
density = 0.4              # proportion of the surface with dots

# Total surface
image_area = width * height

# Dot surface
spot_area = math.pi * radius**2

# Number of dots needed
num_spots = int((density * image_area) / spot_area)

# White image
image = Image.new("L", (width, height), 255)
draw = ImageDraw.Draw(image)

# Generate dots
for _ in range(num_spots):
    x = random.randint(0, width-1)
    y = random.randint(0, height-1)
    if radius ==1:
        image.putpixel((x, y), 0)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)

image = ImageOps.invert(image)

# Save
image.save(output_file, dpi=(300, 300))

