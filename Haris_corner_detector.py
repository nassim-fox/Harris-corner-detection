
# coding: utf-8

# 
# <font color=blue|red|green|pink|yellow>author</font> Nassim Bouhadouf g03
# <font color=blue|red|green|pink|yellow>date création</font> 20/06/2019
# 
# 

# In[1]:



import numpy as np
from matplotlib import pyplot as p
from PIL import Image


# In[3]:


image = Image.open("A.jpg").convert("L") #lecture de l'image et conversion en niveau de gris


# In[4]:


im = np.asarray(image).astype('float32') #convertir l'image en numpy array float pour la manipuler


# ## calcule des dérivés 

# In[49]:


from scipy import ndimage

filtre_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], np.float32)*0.5
filtre_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], np.float32)*0.5

ix = ndimage.convolve(im, filtre_x)
iy = ndimage.convolve(im, filtre_y)


# In[50]:


f , (ax1 , ax2) = p.subplots(1,2,figsize=(15,13))
p.sca(ax1)
p.imshow(ix,cmap='gray')
p.sca(ax2)
p.imshow(iy,cmap='gray')


# In[56]:


A = np.power(ix,2)
B = np.power(iy,2)
C = ix*iy


# ## lissage

# In[58]:


def gaussian(taille, sigma=1):
    taille = int(taille) // 2
    x, y = np.mgrid[-taille:taille+1, -taille:taille+1]
    normal = 1 / (2.0* np.pi * sigma**2)
    s =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return s


# Pour appliquer le filtre gaussian :
#       - Utilisation de la fonction gaussian ci dessus qui retourne un kernel gaussian et l'appliquer sur les images
#       - Utilisation de la fonction prédéfinie gaussian_filter ( que j'ai choisis pour rapidité ) 

# In[59]:


#Appliquer le filtre de Gaussian
Ab = ndimage.gaussian_filter(A, sigma=2) # écart type = 2 
Bb = ndimage.gaussian_filter(B, sigma=2)
Cb = ndimage.gaussian_filter(C, sigma=2)


# ## calcule des coins

# In[67]:


R = np.zeros(im.shape) # initialisation de R ( ou C ) la mesure de Haris
result = np.zeros(im.shape) # matrice des résultats


# trouver si le point est maxima

# In[71]:


def localMax(p,i,j) :  # trouver si le point est maxima 
   # if i > 0 and j > 0 and i < p.shape[0]-1 and j < p.shape[1]-1 : 
        return ( p[i][j] > np.array([p[i-1][j],p[i][j-1],p[i-1][j-1],p[i+1][j],p[i][j+1],p[i+1][j+1],p[i-1][j+1],p[i+1][j-1]]).max() )#.min()


# Classification des points et choix des contours![image.png](attachment:image.png)

# In[84]:


R = np.zeros(im.shape)
result = np.zeros(im.shape) 
max = 0 
for i in range(im.shape[0]) :
    for j in range(im.shape[1]) : 
        M = np.array([[Ab[i][j],Cb[i][j]],[Cb[i][j],Bb[i][j]]])
        R[i][j] = np.linalg.det(M) - 0.04 * (np.power(np.trace(M), 2)) # norme des coins ( en utilisant numpy)
                # Ab[i][j]*Bb[i][j] - np.power(Cb[i][j],2) - 0.04*np.power(Ab[i][j]+Bb[i][j],2) (calcule direct)
        if R[i][j] > max :  # pour éliminer directement les weak corners 
            max = R[i][j]

#mesure de cornerness
for i in range(im.shape[0]-1) : 
    for j in range(im.shape[1]-1) : 
        if R[i][j] > 0.05*max  and localMax(R,i,j) : 
            result[i][j] = 1 

a, b = np.where(result == 1)
p.plot(b, a, 'r+')
p.imshow(im, 'gray')
p.savefig("Haris.png")
p.show()
