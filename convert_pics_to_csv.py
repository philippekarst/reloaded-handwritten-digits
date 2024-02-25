from PIL import Image
import PIL.ImageOps 
from PIL import ImageFilter
import numpy as np
import pandas as pd

img = Image.open(f'./pics/{0}.png').convert('L')
img = PIL.ImageOps.invert(img)

img = img.filter(ImageFilter.SHARPEN)
img = img.filter(ImageFilter.SHARPEN)

#img.show()

arr = np.asarray(img)
arr = arr.flatten()
arry = np.insert(arr,0,0)
for i in range(29):
    img = Image.open(f'./pics/{i+1}.png').convert('L')
    img = PIL.ImageOps.invert(img)

    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)

    #img.show()

    arr = np.asarray(img).flatten()
    arr = np.insert(arr,0,(i+1)%10)
    arry = np.c_[arry,arr]
np.savetxt(f'pics.csv',arry, fmt = '%d', delimiter=",")