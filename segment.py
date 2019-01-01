import numpy as np
import cv2
import os
import random
import config

bins = config.bins
# to map quantifiy the value of each pixel to the corresponding bin
coef = int(256/bins)

skin_proba = np.load("proba_skin.npy")[0]
histo_skin = np.load("histo_skin.npy")
histo_non_skin = np.load("histo_non_skin.npy")
histo_color = np.load("histo_color.npy")

''' now let's try to segment an image based on this rule 
for each pixel color if P(S|C)/P(nS|C) > thresh = 1 then the pixels will be classified as skin pixel
'''
non_skin_proba = 1.0-skin_proba
PS = histo_skin#*skin_proba
epsilon =0.000000000000001 
PNS = histo_non_skin + epsilon

P = (PS/PNS >1.0) + 0 

PSkin = (histo_skin*skin_proba)/(histo_color+epsilon) # TO NOT DIVIDE BY 0
PNSkin =  (histo_non_skin*non_skin_proba)/(histo_color+epsilon)




def segment(img):
    mask_pixels_based = np.zeros((img.shape[0],img.shape[1]))
    mask_segment = np.zeros((img.shape[0],img.shape[1]))

    # segment using quick shift
    from skimage.segmentation import quickshift,mark_boundaries

    segments_quick = quickshift(img, kernel_size=2, max_dist=3, ratio=0.9)
    dict_proba = {}
    dict_count = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b , g , r = img[i][j][0]//coef , img[i][j][1]//coef , img[i][j][2]//coef
            mask_pixels_based[i][j] = P[r][g][b]
            key = segments_quick[i][j]
            value = PSkin[r][g][b]
            mask_segment[i][j] = value
            if(key in dict_proba):
                dict_proba[key]+=value
                dict_count[key]+=1.0
            else:
                
                dict_proba.update({key:value})
                dict_count.update({key:1.0})
                


    dict_avg_proba = {k: float(dict_proba[k])/dict_count[k] for k in dict_proba}

    def vec_translate(a, my_dict):
        return np.vectorize(my_dict.__getitem__)(a)

    segment = vec_translate(segments_quick,dict_avg_proba) 

    mean_segment = (segment >segment.mean())+0.0
    fixed_segment = (segment >0.4)+0.0
    return mask_pixels_based , mask_segment , mean_segment , fixed_segment

image_directory = "./test_images"
skin_directory = "./skin-images"
#skin_images_names = sorted(os.listdir(skin_directory))

test_images_names = sorted(os.listdir(image_directory))

rnd  = random.randint(0, len(image_directory))
path = image_directory +"/"+test_images_names[rnd]
img = cv2.imread(path,-1)
mask_pixels_based , mask_segment , mean_segment , fixed_segment = segment(img)
cv2.imshow('image',img)
'''
vis1 = np.concatenate((mask_pixels_based, mask_segment), axis=1)
vis2 = np.concatenate((mean_segment, fixed_segment), axis=1)
concat_img = np.concatenate((vis1, vis2), axis=0)
cv2.imshow('results',concat_img)
'''


cv2.imshow('pixel_based',mask_pixels_based)
cv2.imshow('superpixel_mask',mask_segment)
cv2.imshow('superpixel_segmentation_mean_thresh',mean_segment)
cv2.imshow('superpixel_segmentation_fixed_thresh',fixed_segment)
cv2.waitKey(0)
cv2.destroyAllWindows()

