import pandas as pd
import numpy as np
import requests
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from loguru import logger


# For extracting and displaying the images


def DisplayImages(recipesData, arrayRecipeID, numRandomChoices):
    

    fig=(plt.figure(figsize=(8,8)))

    for elem in range(numRandomChoices):

        # Getting the image URL, and then splitting it to get the image file name
        imageURL=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].image_url.values[0]

        imageFile=imageURL.split("/")[-1]
        
        # Get the recipe title
        imageTitle=recipesData.loc[recipesData['id']==arrayRecipeID[elem]].title.values[0]

        r=requests.get(imageURL, stream=True)
        r.raw.decode_content=True

        with open(imageFile, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        img=mpimg.imread(imageFile)

        fig.add_subplot(1, numRandomChoices, elem+1)
        plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        # Include the image title (imageTitle) in the image
        plt.title(imageTitle, fontsize=10)
        # Use elem+1 to add a number below each recipe image
        plt.xlabel(elem+1)

    plt.show()

def DisplayImagesURL(recipesData, arrayRecipeID, numRandomChoices):
