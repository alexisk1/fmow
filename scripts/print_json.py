import json
from pprint import pprint

with open('./0_msrgb.json') as data_file:    
    data = json.load(data_file)

pprint(data)

bounding_box = data ['bounding_boxes'][0]['box']

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

im = np.array(Image.open('0_msrgb.jpg'), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect = patches.Rectangle((bounding_box[0],bounding_box[1]),bounding_box[2],bounding_box[3],linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
