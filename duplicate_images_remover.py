import os
from PIL import Image
import distance
import re
import numpy as np
from imagehash import dhash
from numpy import array
import cv2
from google.colab.patches import cv2_imshow
import argparse
import warnings
warnings.filterwarnings('ignore')

valid_images = (".jpg",".gif",".png",".tga")
_nsre = re.compile('([0-9]+)')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)] 


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def image_file_count(path, info):
  for index, filename in  enumerate(sorted(os.listdir(path), key=natural_sort_key)):  
    if filename.endswith(valid_images) and info == True:
      print(filename)

  return index+1

def grayimage_file2hash(path, info):
  hashes = {}
  imghashes =[]
  imgpaths = []
  for index, filename in  enumerate(sorted(os.listdir(path), key=natural_sort_key)):  
    if filename.endswith(valid_images):
      filepath = os.path.join(path,filename)
      imgpaths.append(filepath) 
      img = Image.open(filepath)
      img = img.resize((100,100))
      x = img.convert('L') #makes it grayscale
      dhash1 = dhash(x)  #hash_image(x, 16)
      
      p = hashes.get(dhash1, [])
      p.append(filepath)
      hashes[dhash1] = p
      imghashes.append(str(dhash1))
      if info == True:
        print("dHash: ",index, filename,"-->",dhash1)

  return index+1, imghashes, imgpaths 


def duplicate_images_remover(imghashes, imgpaths):
  # loop over the image hashes
  visited = []
  th = int(args["threshold"])
  for i, imhase1 in enumerate(imghashes):
    a = imhase1
    montage = None
    z = np.inf
    fg = 0
    for j, imhase2 in enumerate(imghashes):  
      if i != j and i not in visited:
        p = imgpaths[i]
        image1 = cv2.imread(p)
        image1 = cv2.resize(image1, (150, 150))
        if montage is None:
          montage = image1
        
        b = imhase2 
        z = distance.hamming(a, b)
      
        # loop over all image paths with the same hash
        if z <= th:
          #print(z)
          visited.append(j)
          # load the input image and resize it to a fixed width
          # and heightG
          q = imgpaths[j]
          image2 = cv2.imread(q)
          image2 = cv2.resize(image2, (150, 150))
          # if our montage is None, initialize it
          montage = np.hstack([montage, image2])
          os.remove(imgpaths[j])
          fg = 1
    
    if fg == 1:
      # show the montage for the hash
      print("[INFO] hash: {}".format(imhase1))
      cv2_imshow(montage)


def delete_image_folder(path, info):
  rem = 0
  # Remove the specified  
  # file path 
  try: 
    for index, filename in  enumerate(sorted(os.listdir(path), key=natural_sort_key)):  
      if filename.endswith(valid_images):
        filepath = os.path.join(path,filename)
        if info == True:
          print(filepath)
        os.remove(filepath)
        rem += 1

    os.rmdir(path.split('/content/')[1]) 
  except OSError as error: 
      print(error) 
      print("Folder can not be removed") 

  return rem

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="path to input image folder to be processed")
ap.add_argument("-t", "--threshold", required=True,
	help="hamming distance threshold value")
ap.add_argument("-i", "--info", type=str2bool, required=True,
	help="print all details or not")
args = vars(ap.parse_args())

total_imgs = image_file_count(args["folder"], False) 
print("Total Images: ", image_file_count(args["folder"], args["info"]))
hash_no, imghashes, imgpaths = grayimage_file2hash(args["folder"], args["info"])
print("Total Images Converted to Hash: ", hash_no)
duplicate_images_remover(imghashes, imgpaths)
print("Remaining Images: ", image_file_count(args["folder"], args["info"]))
deleted_imgs = delete_image_folder(args["folder"], args["info"])
print("Deleted: ", total_imgs - deleted_imgs)