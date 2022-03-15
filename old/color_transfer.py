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

    img_LMS = np.zeros_like(img_RGB,dtype=np.float32)
    # RGB to LMS  based on (4) and logarithmic space

    b = img_RGB[...,2] 
    g = img_RGB[...,1]
    r = img_RGB[...,0]
    print('lms before log',np.mean(img_RGB))
    
    img_LMS[...,0] = np.log10(0.3811*r + 0.5783*g + 0.0402*b) 
    img_LMS[...,1] =  np.log10(0.1967*r + 0.7244*g + 0.0782*b) 
    img_LMS[...,2] = np.log10(0.0241*r + 0.1288*g + 0.8444*b)
    

    #tmp = np.array([0.3811,0.5783,0.0402,0.1967,0.7244,0.0782,0.0241,0.1288,0.8444]).reshape(3,3)
    ##img_LMS = helper3by3(tmp, img_RGB)
    #img_LMS = np.log10(img_LMS)
    print('lms after log',np.mean(img_LMS))

    #print(img_LMS)

    img_Lab_ = np.zeros_like(img_RGB,dtype=np.float32)
    # LMS to Lab based on (6)

    matrix_right = np.array([[1,1,1],[1,1,-2],[1,-1,0]])

    matrix_left = np.array([[1/np.sqrt(3),0,0],[0,1/np.sqrt(6),0],[0,0,1/np.sqrt(2)]])

    matrix_LMS_Lab = np.matmul(matrix_left,matrix_right)
     
   
    img_Lab_ = cvt(matrix_LMS_Lab, img_LMS)
    print("here: ",np.mean(img_Lab_))

    return img_Lab_


def convert_color_space_Lab_to_RGB(img_Lab):
# Now back lab -> LMS -> rgb
    img_LMS = np.zeros_like(img_Lab,dtype=np.float32)
     
    
    matrix_left = np.array([[1,1,1],[1,1,-1],[1,-2,0]])

    matrix_right = np.array([[np.sqrt(3)/3,0,0],[0,np.sqrt(6)/6,0],[0,0,np.sqrt(2)/2]])

    matrix_lab_lms = np.matmul(matrix_left,matrix_right)

    img_LMS = cvt(matrix_lab_lms, img_Lab)

    img_RGB = np.zeros_like(img_Lab,dtype=np.float32)

    # equaltion 9
    img_RGB_matrix = np.array([[ 4.4679, -3.5873, 0.1193],[-1.2186, 2.3809 , -0.1624],[0.0497, -0.2439, 1.2045]])
    
    img_LMS = np.power(10, img_LMS)
    img_RGB = cvt(img_RGB_matrix, img_LMS)
  



    return img_RGB




def color_transfer_in_Lab(src, trgt):
    print('===== color_transfer_in_Lab =====')

    print('===== color_transfer_in_Lab =====')
    # to be completed ...
    # first convert to lab space 
    img_lab_source =  convert_color_space_RGB_to_Lab(src)

    img_lab_target =  convert_color_space_RGB_to_Lab(trgt)
    
   # img_lab_source = cv2.cvtColor(src=img_RGB_source_, code=cv2.COLOR_BGR2Lab)
   # img_lab_target = cv2.cvtColor(src=img_RGB_target_, code=cv2.COLOR_BGR2Lab)
    
    # bgr corresponding to lab 
    ## begin eqation 10 & 11
    print('before transfer source in lab', np.mean(img_lab_source))
    bt = img_lab_target[:, :, 0]
    gt = img_lab_target[:, :, 1]
    rt = img_lab_target[:, :, 2]

    bs = img_lab_source[:, :, 0]
    gs = img_lab_source[:, :, 1]
    rs = img_lab_source[:, :, 2]
                  
    mean_bs = np.mean(bs)   
    mean_gs = np.mean(gs)  
    mean_rs = np.mean(rs)   
                         
    mean_bt = np.mean(bt)
    mean_gt = np.mean(gt)
    mean_rt = np.mean(rt)

 
    bs = bs - mean_bs   
    gs = gs - mean_gs
    rs = rs - mean_rs

    bs = (np.std(bt)/np.std(bs)) * bs
    gs = (np.std(gt)/np.std(gs)) * gs
    rs = (np.std(rt)/np.std(rs)) * rs


    bs = bs + mean_bt
    gs = gs + mean_gt
    rs = rs + mean_rt
 
    result = np.zeros_like(img_lab_source,dtype=np.float32)

    result[:, :, 0] = bs     
    result[:, :, 1] = gs
    result[:, :, 2] = rs
    print('lab after transfer',np.mean(result))
    ## end equation 10 11 

    # first lab to rbg
    result_RGB = convert_color_space_Lab_to_RGB(result)


    
    return result_RGB 
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
   
    img_RGB_new_Lab       = color_transfer(convert_color_space_BGR_to_RGB(src), convert_color_space_BGR_to_RGB(trgt), option='in_Lab')

    #cv2.imshow("plz gawd",convert_color_space_RGB_to_BGR(img_RGB_new_Lab.clip(0.0, 255.0).astype(np.uint8)))
    #cv2.waitKey(0)
    # # todo: save image to path_file_image_result_in_Lab
    # cv2.imwrite('result.png',img_RGB_new_Lab)
    
    cv2.imwrite(path_file_image_result_in_Lab,
    convert_color_space_RGB_to_BGR(img_RGB_new_Lab).clip(0,255.0).astype(np.uint8))

    rmse(path_file_image_result_in_Lab,r"result1.png")
    

    img_RGB_new_RGB       = color_transfer(src, trgt, option='in_RGB')
    # cv2.imwrite(path_file_image_result_in_RGB,convert_color_space_RGB_to_BGR(img_RGB_new_Lab.clip(0.0, 255.0).astype(np.uint8)))

    img_RGB_new_CIECAM97s = color_transfer(src, trgt, option='in_CIECAM97s')
    # cv2.imwrite(path_file_image_result_in_CIECAM97s,convert_color_space_RGB_to_BGR(img_RGB_new_Lab.clip(0.0, 255.0).astype(np.uint8)))
