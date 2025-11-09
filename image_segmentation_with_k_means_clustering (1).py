# -*- coding: utf-8 -*-
"""image-segmentation-with-k-means-clustering.ipynb
Original file is located at
    https://colab.research.google.com/drive/1II-Hr2nccpJGqo_6bdkeOOtplgSXgTtV
"""

# Colab-Ready Environment Setup

import numpy as np          # Linear algebra
import pandas as pd         # CSV/file processing
import matplotlib.pyplot as plt
import os
from PIL import Image       # Image handling
from io import BytesIO      # For in-memory byte streams
import cv2                  # OpenCV for image processing
import requests             # For downloading images from URL

from google.colab import files

#import cv2
#import matplotlib.pyplot as plt
# -----------------------------
# Image Loading and Display
# -----------------------------

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


image_path = "/content/Blanquate.jpg"#"/content/milkyway.jpg"#'/content/flowerimage.jpg'  # Replace with image filename

# Load the image
try:
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    print(f"Image '{image_path}' loaded successfully. Shape: {img.shape}")
except FileNotFoundError:
    print(f"File '{image_path}' not found. Please upload the image first.")
    img = None

# Display the original image
if img is not None:
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

"""## 1. Definition and Explanation of Image Segmentation:

<font color="green">
Image segmentation is the classification of an image into different groups.
    
Image segmentation is the task of partitioning an image into multiple segments. In semantic segmentation, all pixels that are part of the same object type get assigned to the same segment. For example, in a selfdriving car’s vision system, all pixels that are part of a pedestrian’s image might be assigned to the “pedestrian” segment (there would be one segment containing all the pedestrians).In some applications, this may be sufficient. For example, if you want to analyze satellite images to measure how much total forest area there is in a region, color segmentation may be just fine.
    
    
    
Image segmentation is the process of partitioning a digital image into multiple distinct regions containing each pixel(sets of pixels, also known as superpixels) with similar attributes.   
The goal of Image segmentation is to change the representation of an image into something that is more meaningful and easier to analyze. 
Image segmentation is typically used to locate objects and boundaries(lines, curves, etc.) in images. More precisely, Image Segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

## 2.Where Can We Use Image Segmentation:

<font color="purple">
1. In  Autonomous Vehicles: they need sensory input devices like cameras, radar, and lasers to allow the car to perceive the world around it, creating a digital map. Autonomous driving is not even possible without object detection which itself involves image classification/segmentation.
    
    
    
2. Healthcare Industry:if we talk about Cancer, even in today’s age of technological advancements, cancer can be fatal if we don’t identify it at an early stage. Detecting cancerous cell(s) as quickly as possible can potentially save millions of lives. The shape of the cancerous cells plays a vital role in determining the severity of cancer which can be identified using image classification algorithms.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Path to uploaded image in Colab
image_path = "/content/Blanquate.jpg"#"/content/milkyway.jpg"#"/content/flowerimage.jpg"

# Load the image
try:
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    print(f"Image '{image_path}' loaded successfully. Shape: {img.shape}")
except FileNotFoundError:
    print(f"File '{image_path}' not found. Please upload the image first.")
    img = None
except Exception as e:
    print(f"Error loading image: {e}")
    img = None

# Display the image
if img is not None:
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np

if img is None:
    print("Image 'img' is not loaded. Please ensure the image loading cell was executed successfully.")
else:
    # Split channels
    r, g, b = cv2.split(img)
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r, g, b, c=np.stack([r,g,b], axis=1)/255.0, marker='o', s=1)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('3D RGB Scatter Plot')

    plt.show()

"""## 3. K Means Algorithm for Image Segmentation:

<font color="purple">
K Means clustering algorithm is an unsupervised algorithm and it is used to segment the interest area from the background. It clusters, or partitions the given data into K-clusters or parts based on the K-centroids.
    

The algorithm is used when you have unlabeled data(i.e. data without defined categories or groups). The goal is to find certain groups based on some kind of similarity in the data with the number of groups represented by K.
"""

if img is None:
    print("Image 'img' is not loaded. Please ensure the image loading cell was executed successfully.")
else:
    plt.figure(figsize=(13,10))
    plt.imshow(img)

img.shape #the first is height, the second is width and the third is the color channel of the image

#Next, converts the HxWx3 image into a Kx3 matrix where K=HxW and each row is now a vector in the 3-D space of RGB.
vectorized_img = img.reshape((-1,3))
vectorized_img.shape

#We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV.
vectorized_img= np.float32(vectorized_img)
vectorized_img

"""<font color="purple">
We are going to cluster with k = 3 because if you look at the image above it has 3 colors, green-colored grass and forest, blue sea and the greenish-blue seashore.

<font color="purple">
OpenCV provides cv2.kmeans(samples, nclusters(K), criteria, attempts, flags) function for color clustering.

1. samples: It should be of np.float32 data type, and each feature should be put in a single column.

2. nclusters(K): Number of clusters required at the end

3. criteria: It is the iteration termination criteria. When this criterion is satisfied, the algorithm iteration stops. Actually, it should be a tuple of 3 parameters. They are `( type, max_iter, epsilon )`:

Type of termination criteria. It has 3 flags as below:

cv.TERM_CRITERIA_EPS — stop the algorithm iteration if specified accuracy, epsilon, is reached.
cv.TERM_CRITERIA_MAX_ITER — stop the algorithm after the specified number of iterations, max_iter.
cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER — stop the iteration when any of the above condition is met.
4. attempts: Flag to specify the number of times the algorithm is executed using different initial labelings. The algorithm returns the labels that yield the best compactness. This compactness is returned as output.

5. flags: This flag is used to specify how initial centers are taken. Normally two flags are used for this: cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS.
"""

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 3
attempts=10
ret,label,center=cv2.kmeans(vectorized_img,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
center

#Next, we have to access the labels to regenerate the clustered image
res = center[label.flatten()]
result_image = res.reshape((img.shape))

plt.figure(figsize=(15,10))
plt.imshow(result_image)

plt.figure(figsize=(15,12))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K)
plt.show()

"""<font color="purple">
Let's see what happens when we change the value of K=5:
"""

K = 5
attempts=10
ret,label,center=cv2.kmeans(vectorized_img,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)
center

res = center[label.flatten()]
result_image = res.reshape((img.shape))

plt.figure(figsize=(15,12))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K)
plt.show()

"""<font color="purple">
As you can see with an increase in the value of K, the image becomes clearer because the K-means algorithm can classify more classes/cluster of colors.
"""

K = 20
attempts=10
ret,label,center=cv2.kmeans(vectorized_img,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))
plt.figure(figsize=(15,12))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K)
plt.show()
