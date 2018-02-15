# coding: utf-8

# In[2]:


import cv2, os, imutils
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from matplotlib import pyplot as plt


# In[3]:


def resize_maintain_ar(img_unsquare):
    out_size = 128
    in_size = img_unsquare.shape[:2]
    convert_ratio = float(out_size)/max(in_size)
    img_copy_size = tuple(int(dim*convert_ratio) for dim in in_size)
    
    res_unsquare = cv2.resize(img_unsquare,(img_copy_size[1],img_copy_size[0]))
    
    d_width = out_size - img_copy_size[1]
    d_height = out_size - img_copy_size[0]
    
    border_top, border_bottom = d_height//2, d_height-(d_height//2)
    border_left, border_right = d_width//2, d_width-(d_width//2)
    
    color = [255,255,255]
    img_square = cv2.copyMakeBorder(res_unsquare, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, value=color)
    return img_square


# In[4]:


def normalize_rgb_vals(img):
    #bmax = np.max(img[:,:,0])
    #gmax = np.max(img[:,:,1])
    #rmax = np.max(img[:,:,2])
    
    #b = img[:,:,0] / (img[:,:,0] * bmax )
    #g = img[:,:,1] / (img[:,:,1] * gmax )
    #r = img[:,:,2] / (img[:,:,2] * rmax )
    b_norm = normalize(img[:,:,0], norm='l1')
    g_norm = normalize(img[:,:,1], norm='l1')
    r_norm = normalize(img[:,:,2], norm='l1')

    img_norm = cv2.merge((b_norm,g_norm,r_norm))
    return img_norm


# In[ ]:


for subdir, dirs, files in os.walk('.\gear_images'):
    arr_std = []
    
    for file in files:
        filepath = subdir + os.sep + file
        
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_res = resize_maintain_ar(img)
        res_norm = normalize_rgb_vals(img_res)
        
        arr_std.append(img_res)
    
    if arr_std != []:
        arr_std = np.asarray([x.reshape(3*128*128,) for x in arr_std], dtype=np.uint8)

        scaler = StandardScaler()

        scaler.fit(arr_std)

        #arr_std = [np.reshape(x,(128,128,3)) for x in arr_std]
        
        if not os.path.exists('.\gear_images_proc'):
            os.makedirs('.\gear_images_proc')

        for i, e in enumerate(scaler.transform(arr_std)):
            e_zero2one = (e-np.min(e))*255/(np.max(e) - np.min(e))
            img_out = np.reshape(e_zero2one,(128,128,3))
            for angle in np.arange(0, 180, 90):
                rotated = imutils.rotate(img_out, angle)
                if not os.path.exists('.\gear_images_proc' + os.sep + subdir):
                    os.makedirs('.\gear_images_proc' + os.sep + subdir)
                cv2.imwrite('.\gear_images_proc' + os.sep + subdir + os.sep + str(i) + '-' + str(angle) + '.jpg', rotated)
                plt.imshow(rotated)
                plt.show()
