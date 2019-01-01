from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import config

def visualize_histo(P,bins = config.bins):
    coef = int(256/bins)
    fig = plt.figure()
    fig.suptitle('skin pixels ', fontsize=16)
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(bins)
    y = np.arange(bins)
    z = np.arange(bins)

    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                if(P[i][j][k]==1):
                    c_i,c_j,c_k = x[i]*coef/255 , y[j]*coef/255, z[k]*coef/255
                    ax.scatter(x[i], y[j], z[k], c=[[c_i,c_j,c_k]] , cmap=plt.hot())
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()

if __name__ == "__main__":
        
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

    visualize_histo(P)