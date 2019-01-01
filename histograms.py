import os , sys
import numpy as np
import PIL
from PIL import Image
import config
import helper

bins = config.bins
directory_skin = config.directory_skin
directory_non_skin = "./non-skin-images"

def compute_histograms(directory_skin = directory_skin ,directory_non_skin = directory_non_skin, bins = bins ,verbose=False ):

    # to map quantifiy the value of each pixel to the corresponding bin
    coef = int(256/bins) 

    # images in the skin directory
    names_images_skin = sorted(os.listdir(directory_skin))

    # skin colors histogram
    histo_skin = np.zeros((bins,bins,bins))

    # non skin colors histogram
    histo_non_skin = np.zeros((bins,bins,bins))

    # total number of skin pixels
    total_skin = 0

    # total number of pixels in all images
    total_pixels = 0


    count_progress = 0
    total_skin_images = len(names_images_skin)
    if(verbose) : print("Processing skin images...")
    for name in names_images_skin:
        count_progress+=1
        if(verbose): helper.print_progress(count_progress,total_skin_images)
        # path of the image
        path = os.path.join(directory_skin,name)
        id = name.split('.')[0] + ".pbm"
        # look for the mask of the image 
        path_mask = os.path.join("./masks",id)
        # read the image in RGB mode
        img = Image.open(path).convert('RGB')
        # read the mask (values in {True,False})
        mask = Image.open(path_mask)

        # convert mask to array
        pix_mask =  np.array(mask) + 0
        # number of skin pixels in the image
        nb_skin = (pix_mask == 1 ).sum()
        
        total_skin+=nb_skin
        # total number of pixels in the image
        total_pixels+= pix_mask.shape[0]*pix_mask.shape[1]

        # convert image to np array
        pix_img = np.array(img)

        # chack that the sizes of the image and its mask are equals
        assert(pix_img.shape[0]==pix_mask.shape[0] and pix_img.shape[1]==pix_mask.shape[1])

        for i in range(pix_mask.shape[0]):
            for j in range(pix_mask.shape[1]):
                # quantify the value of each color 
                r , g , b = pix_img[i][j]//coef
                # if it is a skin pixel
                if(pix_mask[i][j]==1):
                    # increment the value of the color in the skin-histogram
                    histo_skin[r][g][b] +=1
                # else
                else : 
                    # increment the value of the color in the non-skin-histogram
                    histo_non_skin[r][g][b]+=1

        '''
        We do the same for non-skin-images
        '''
        
        non_skin_files = sorted(os.listdir(directory_non_skin))

        total_non_skin = 0
        total_non_skin_images = len(non_skin_files)
        if(verbose) : print("\n Processing non skin images...")
        count_progress=0
        for name in non_skin_files:
            count_progress+=1
            if(verbose):helper.print_progress(count_progress,total_non_skin_images)
            path = os.path.join("./non-skin-images",name)
            img = Image.open(path).convert('RGB')
            pix_img = np.array(img)
            
            for i in range(pix_img.shape[0]):
                for j in range(pix_img.shape[1]):
                    r , g , b = pix_img[i][j]//coef
                    histo_non_skin[r][g][b] +=1
            total_non_skin+=pix_img.shape[0]*pix_img.shape[1]

        total_non_skin_in_skin_directory = total_pixels - total_skin

        total_pixels+=total_non_skin

        total_non_skin+= total_non_skin_in_skin_directory


        histo_skin/=total_skin # histo skin is equivalent to the probability of each color to appear given it is a skin pixel : P(Color = c | S = skin)
        histo_non_skin/=total_non_skin # histo non skin is equivalent to the probability of each color to appear given it is a non skin pixel : P(Color = c | S = non-skin)
        # P_s prior probability of being a skin pixel
        p_s = round(total_skin/total_pixels,4)
        # P_ns prior probability of being a non skin pixel
        p_ns = 1.0-p_s
        # hence the histo color is the following
        histo_color = p_s*histo_skin + p_ns*histo_non_skin # P(C) = P(C|S)*P(S) + P(C|nS)*P(nS)

        histos = {
            "histo_skin" : histo_skin,
            "histo_non_skin" : histo_non_skin,
            "histo_color" : histo_color,
            "proba_skin" : p_s
        }
        return histos



if __name__ == "__main__":
    
    histograms = compute_histograms(verbose=False)

    np.save("histo_skin",histograms["histo_skin"])
    np.save("histo_non_skin",histograms["histo_non_skin"])
    np.save("histo_color",histograms["histo_color"])
    p_s = histograms["proba_skin"]
    np.save("proba_skin",np.array([p_s]))


