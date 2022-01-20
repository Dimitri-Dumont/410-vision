#Dimitri Dumont
#cs410-Computer-Vision

import cv2 
import numpy as np
import sys
import time

def convert_color_space_BGR_to_RGB(img):
    temp = np.zeros_like(img, dtype=np.float32)
    temp[...,0] = img[...,2]
    temp[...,1] = img[...,1]
    temp[...,2] = img[...,0]

    img[...,0] = temp[...,0]
    img[...,1] = temp[...,1]
    img[...,2] = temp[...,2]

    return img

def convert_color_space_RGB_to_BGR(img):
    temp = np.zeros_like(img, dtype=np.float32)

    temp[...,0] = img[...,2]
    temp[...,2] = img[...,0]

    img[...,0] = temp[...,0]
    img[...,2] = temp[...,2]
    return img

def convert_color_space_RGB_to_CIECAM97s(img_RGB):
    '''
    convert image color space RGB to CIECAM97s
    '''
    img_CIECAM97s = np.zeros_like(img_RGB,dtype=np.float32)
    # to be completed ...

    return img_CIECAM97s

def convert_color_space_CIECAM97s_to_RGB(img_CIECAM97s):
    '''
    convert image color space CIECAM97s to RGB
    '''
    img_RGB = np.zeros_like(img_CIECAM97s,dtype=np.float32)
    # to be completed ...

    return img_RGB


#-----------------------------------------------#


def convert_color_space_RGB_to_Lab(img_RGB):
    img = np.zeros_like(img_RGB,dtype=np.float32)

    RGB2XYZ = [
    [.5141,.3239,.1604],
    [.2651,.6702,.0641],
    [.0241, .1228,.8444]
    ]
    XYZ2LMS = [ 
        [.3897,.6890,-.0787],
        [-.2298, 1.1834, .0464],
        [0,0,1]
    ]
    LMS2LAB_2 = [ 
        [(1/np.sqrt(3)),0,0],
        [0,(1/(np.sqrt(6))),0],
        [0,0,(1/np.sqrt(2))]
    ]
    LMS2LAB_1 = [ 
        [1,1,1],
        [1,1,-2],
        [1,-1,0]
    ]
    # rgb -> xyz -> lms -> LMS -> LAB
    img = cvt(np.array(RGB2XYZ),img_RGB)
    img = cvt(np.array(XYZ2LMS),img)
    # img = lms2LMS(img)
    # cv2.imshow("lms->LMS",img)
    img = cvt(np.array(LMS2LAB_1),img)
    img = cvt(np.array(LMS2LAB_2),img)

    return img


def convert_color_space_Lab_to_RGB(lab):
# Now back lab -> LMS -> rgb
    img = np.zeros_like(lab,dtype=np.float32)

    LAB2LMS_1 = [  
        [(np.sqrt(3)/3),0,0],
        [0,(np.sqrt(6)/6),0],
        [0,0,(np.sqrt(2)/2)]
    ]
    LAB2LMS_2 = [ 
        [1,1,1],
        [1,1,-1],
        [1,-2,0]
    ]
    LMS2RGB = [  
        [4.4679,-3.5873,.1193],
        [-1.2186,2.3809,-.1624],
        [.0497,-.2439,1.2045]
    ]
    img = cvt(np.array(LAB2LMS_1),lab)
    img = cvt(np.array(LAB2LMS_2),img)

    # img[...,0] = 10**img[...,0] 
    # img[...,1] = 10**img[...,0] 
    # img[...,2] = 10**img[...,0] 

    img = cvt(np.array(LMS2RGB),img)


    return img




def color_transfer_in_Lab(src, trgt):
    print('===== color_transfer_in_Lab =====')

    srcLab = convert_color_space_RGB_to_Lab(src)
    trgtLab = convert_color_space_RGB_to_Lab(trgt)

    l_src = srcLab[...,0]
    a_src = srcLab[...,1]
    b_src = srcLab[...,2]

    l_trgt = trgtLab[...,0]
    a_trgt = trgtLab[...,1]
    b_trgt = trgtLab[...,2]


    # subtract mean to get x* channel values   to src
    l_splat = l_src - np.full_like(l_src,np.mean(l_src)) #dtype bug maybe
    a_splat = a_src - np.full_like(a_src,np.mean(a_src))
    b_splat = b_src - np.full_like(b_src,np.mean(b_src))

    
    #apply scalar
    newL = ((np.std(l_trgt)/np.std(l_src))*l_splat) + np.mean(l_trgt)
    newA = ((np.std(a_trgt)/np.std(a_src))*a_splat) + np.mean(a_trgt)
    newB = ((np.std(b_trgt)/np.std(b_src))*b_splat) + np.mean(b_trgt)

  
    srcLab[...,0] = newL
    srcLab[...,1] = newA
    srcLab[...,2] = newB

    return convert_color_space_Lab_to_RGB(srcLab)
    
def color_transfer_in_RGB(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_RGB =====')
    # to be completed ...

def color_transfer_in_CIECAM97s(src, trgt):
    print('===== color_transfer_in_CIECAM97s =====')

    # to be completed ...

def color_transfer(img_RGB_source, img_RGB_target, option):
    if option == 'in_RGB':
        img_RGB_new = color_transfer_in_RGB(img_RGB_source, img_RGB_target)
    elif option == 'in_Lab':
        img_RGB_new = color_transfer_in_Lab(img_RGB_source, img_RGB_target)
    elif option == 'in_CIECAM97s':
        img_RGB_new = color_transfer_in_CIECAM97s(img_RGB_source, img_RGB_target)
    return img_RGB_new

def rmse(apath,bpath):
    """
    This is the cvt function to get RMSE score.
    apath: path to your result
    bpath: path to our reference image
    when saving your result to disk, please clip it to 0,255:
    .clip(0.0, 255.0).astype(np.uint8))
    """
    a = cv2.imread(apath).astype(np.float32)
    b = cv2.imread(bpath).astype(np.float32)
    print(np.sqrt(np.mean((a-b)**2)))

#conversion matrix helper function 
def cvt(matrix, img):
    newimg = np.zeros_like(img, dtype=np.float32)
    newimg[...,0] = img[...,0]* matrix[0][0] + img[...,1]* matrix[0][1] + img[...,2]* matrix[0][2]
    newimg[...,1] = img[...,0]* matrix[1][0] + img[...,1]* matrix[1][1] + img[...,2]* matrix[1][2]
    newimg[...,2] = img[...,0]* matrix[2][0] + img[...,1]* matrix[2][1] + img[...,2]* matrix[2][2]
    return newimg

def lms2LMS(lms):
    lms[...,0] = np.log10(lms[...,0])
    lms[...,1] = np.log10(lms[...,1])
    lms[...,2] = np.log10(lms[...,2])
    return lms


if __name__ == "__main__":
    print('==================================================')
    print('PSU CS 410/510, Winter 2022, HW1: color transfer')
    print('==================================================')

    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]
    path_file_image_result_in_RGB = sys.argv[4]
    path_file_image_result_in_CIECAM97s = sys.argv[5]
   
   
    # ===== read input images

    src = cv2.imread(path_file_image_source)
    trgt = cv2.imread(path_file_image_target)
   
    img_RGB_new_Lab       = color_transfer(convert_color_space_BGR_to_RGB(src), 
    convert_color_space_BGR_to_RGB(trgt), option='in_Lab')

    cv2.imshow("plz gawd",convert_color_space_RGB_to_BGR(img_RGB_new_Lab.clip(0.0, 255.0).astype(np.uint8)))
    cv2.waitKey(0)
    # # todo: save image to path_file_image_result_in_Lab
    # cv2.imwrite('result.png',img_RGB_new_Lab)
    
    cv2.imwrite(path_file_image_result_in_Lab,convert_color_space_RGB_to_BGR(img_RGB_new_Lab.clip(0.0, 255.0).astype(np.uint8)))

    rmse(path_file_image_result_in_Lab,r"result1.png")
    

    img_RGB_new_RGB       = color_transfer(src, trgt, option='in_RGB')
    # cv2.imwrite(path_file_image_result_in_RGB,convert_color_space_RGB_to_BGR(img_RGB_new_Lab.clip(0.0, 255.0).astype(np.uint8)))

    img_RGB_new_CIECAM97s = color_transfer(src, trgt, option='in_CIECAM97s')
    # cv2.imwrite(path_file_image_result_in_CIECAM97s,convert_color_space_RGB_to_BGR(img_RGB_new_Lab.clip(0.0, 255.0).astype(np.uint8)))
