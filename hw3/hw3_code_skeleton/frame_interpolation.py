import cv2 as cv
import sys
import numpy as np
import pickle
import numpy as np
import os
import math
BLUR_OCC = 3

def bilinear(im,x,y):
    '''
    linear interpolation to get pixel value of a pixel [x,y] in image im
    :return pixel: pixel value
    :return inbounds: 1=in boundary, 0=out of boundary
    '''
    sh = im.shape
    w = sh[1]
    h = sh[0]
    inBounds = 1
    if x<0:
        inBounds=0
        x=0.0
    elif x>=(w-1):
        inBounds=0
        x = w-1.000001
    if y<0:
        inBounds=0
        y=0.0
    elif y>=(h-1):
        inBounds=0
        y = h-1.000001

    x0 = int(x)
    y0 = int(y)
    if x0<(w-1):
        x1 = x0+1.0
    else:
        x1 = x0
    if y0<(h-1):
        y1 = y0+1.0
    else:
        y1 = y0
    dx = np.float32(x) - np.float32(x0)
    dy = np.float32(y) - np.float32(y0)

    p00 = im[np.int32(y0),np.int32(x0)].astype(np.float32)
    p10 = im[np.int32(y0),np.int32(x1)].astype(np.float32)
    p01 = im[np.int32(y1),np.int32(x0)].astype(np.float32)
    p11 = im[np.int32(y1),np.int32(x1)].astype(np.float32)

    pixel = (1.0-dx) * (1.0-dy) * p00+\
                 dx  * (1.0-dy) * p10+\
            (1.0-dx) *      dy  * p01+\
                 dx  *      dy  * p11
    return pixel, inBounds





def readFlowFile(file):
    '''
    credit: this function code is obtained from: https://github.com/Johswald/flow-code-python
    '''
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def find_holes(flow):
    '''
    Find a mask of holes in a given flow matrix
    Determine it is a hole if a vector length is too long: >10^9, of it contains NAN, of INF
    :param flow: an dense optical flow matrix of shape [h,w,2], containing a vector [ux,uy] for each pixel
    :return: a mask annotated 0=hole, 1=no hole
    '''
    holes=np.ones(shape=(380,420))
    for y in range(380):
        for x in range(420):
            if(abs(flow[y,x,0]) > np.power(10,9) or np.isnan(flow[y,x,0]) or np.isinf(flow[y,x,0])):
                holes[y,x] = 0

    
    return holes



def holefill(flow, holes):
    '''
    :param flow: matrix of dense optical flow, it has shape [h,w,2]
    :param holes: a binary mask that annotate the location of a hole, 0=hole, 1=no hole
    :return: flow: updated flow
    '''
    h,w,_ = flow.shape

    change=1
    while change==1:
        change=0
        for y in range(0, h):
            ym1 = np.maximum(0, y - 1)
            yp1 = np.minimum(y + 1, h - 1)
            for x in range(0,w):
                # // if the flow is unknown
                # if holes[y,x] == 0:
                if unknown_flow(flow[y,x,0],flow[y,x,1])==True:
                    sx2 = 0.0
                    sy2 = 0.0
                    count = 0.0

                    xm1 = np.maximum(0, x - 1)
                    xp1 = np.minimum(x + 1, w - 1)

                    for y2 in range(ym1, yp1+1):
                        for x2 in range(xm1,xp1+1):

                            fx2 = flow[y2, x2, 0]
                            fy2 = flow[y2, x2, 1]
                            # // if its neiboring is not a hole
                            # if holes[y2,x2]!=0:
                            if unknown_flow(fx2,fy2)==False:
                                sx2 += fx2
                                sy2 += fy2
                                count+=1
                    if count>0:
                        flow[y,x,0] = np.float32(sx2/count)
                        flow[y,x,1] = np.float32(sy2/count)
                        holes[y,x]=1
                        change=1
    return flow

def holeHelp(flow,holes,y,x):
    #sp = surrounding pixels
    sp = {flow[y+1,x-1], flow[y+1,x],flow[y+1,x+1],flow[y,x-1],flow[y,x+1],flow[y-1,x-1],flow[y-1,x],flow[y-1,x+1]}
    holeCheck = {holes[y+1,x-1], holes[y+1,x],holes[y+1,x+1],holes[y,x-1],holes[y,x+1],holes[y-1,x-1],holes[y-1,x],holes[y-1,x+1]}
    #remove any surrounding holes from sp
    index = 0
    for hole in holeCheck:
        if(hole == 0):
            sp.pop(index)
        index+=1

    sum = np.sum(sp)
    return sum/len(sp)

def occlusions(flow0, frame0, frame1):
    '''
    Follow the step 3 in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.
    :param flow0: dense optical flow
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :return:
    '''
    h,w,_ = flow0.shape
    occ0 = np.zeros([h,w],dtype=np.float32)
    occ1 = np.zeros([h,w],dtype=np.float32)

    # ==================================================
    # ===== step 4/ warp flow field to target frame
    # ==================================================
    flow1 = interpflow(flow0, frame0, frame1, 1.0)
    pickle.dump(flow1, open('flow1.step4.data', 'wb'))
    # ====== score
    flow1       = pickle.load(open('flow1.step4.data', 'rb'))
    flow1_step4 = pickle.load(open('flow1.step4.sample', 'rb'))
    diff = np.sum(np.abs(flow1-flow1_step4))
    print('flow1_step4',diff)

    # ==================================================
    # ===== main part of step 5
    # ==================================================
    for y in range(0, h):
        for x in range(0,w):
            fx = np.float32(flow0[y,x,0])
            fy =  np.float32(flow0[y,x,1])

            assert (unknown_flow(fy,fx)==False)

            x1=np.int32(x+fx+0.5)
            y1=np.int32(y+fy+0.5)

            if (x1>=0) and (x1<w) and (y1>=0) and (y1<h):
                fx1 = np.float32(flow1[y1,x1,0])
                fy1 = np.float32(flow1[y1,x1,1])
                absdiff = abs(fx1-fx) + abs(fy1-fy)

                if(absdiff > 0.5):
                    occ0[y,x] = 1
            else:
                occ0[y,x] = 1
            
            fx = np.float32(flow1[y,x,0])
            fy = np.float32(flow1[y,x,1])
            if fx > np.power(10,9) or fy > np.power(10, 9):
                occ1[y,x] = 1

    return occ0,occ1

def unknown_flow(fx,fy):
    '''
    determine if a flow vector is unknown
    :param fx: component x of a flow vector
    :param fy: component y of a flow vector
    :return:
    '''
    if (fx>np.power(10,9)) or (fy>np.power(10,9)) or np.isnan(fx) or np.isnan(fy):
        return True
    return False
# def unknown_flow(y,x,flow):
#     y = int(y)
#     x = int(x)
#     if x > 380 or y > 420:
#         return False
#     if(abs(flow[y,x,0]) > np.power(10,9) or np.isnan(flow[y,x,0]) or np.isinf(flow[y,x,0])):
#         return True
#     else:
#         return False

def interpflow(flow, frame0, frame1, t):
    '''
    Forward warping flow (from frame0 to frame1) to a position t in the middle of the 2 frames
    Follow the algorithm (1) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
    for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param flow: dense optical flow from frame0 to frame1
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param t: the intermiddite position in the middle of the 2 input frames
    :return: a warped flow
    '''
    iflow = np.full_like(flow, np.array([np.power(10, 10), np.power(10, 10)]))    
    h,w,_ = flow.shape
    colorDiffTable = np.ones([h,w],dtype=np.float32)*np.power(10,5).astype(np.float32)

    for y in range(0, h):
            for x in range(0,w):
                p0 =  frame0[y,x]
                for yy in np.arange(-0.5,0.51,0.5):
                    for xx in np.arange(-0.5,0.51,0.5):
                        fx,fy = flow[y,x]
                        nx = np.int32(x + t*fx+xx+0.5)
                        ny = np.int32(y + t*fy+yy+0.5)
                        p1,_ = bilinear(frame1, x+xx+fx, y+yy+fy)
                        #p1 = temp[y,x]
                        #p1 = 0
                        colordiff = np.sum(np.abs(p0-p1)) 
                        # Store flow (fx,fy) in iflow[ny, nx] with smallest colordiff with difference table: 
                        if (nx < w) and (ny < h) and nx >=0 and ny >=0 :
                            if colordiff < colorDiffTable[ny,nx]:
                                iflow[ny,nx,0] = fx
                                iflow[ny,nx,1] = fy
                                colorDiffTable[ny,nx] = colordiff

    return iflow

# def bilinear(img,x,y):
#     temp =  cv.resize(frame0,(380,420), interpolation = cv.INTER_LINEAR)
#     x = int(x)
#     y = int(y)
#     try:
#         ret = temp[y,x]
#         #img[j,i] = (1-a)*(1-b) * img[j,i] + a*(1-b) * img[j,i+1] + (a * b * img[j+1,i+1]) + ((1-a)*b * img[j+1,i])  
#     except:
#         return 0, False

#     return ret, True


def warpimages(iflow, frame0, frame1, occ0, occ1, t):
    '''
    Compute the colors of the interpolated pixels by inverse-warping frame 0 and frame 1 to the postion t based on the
    forwarded-warped flow iflow at t
    Follow the algorithm (4) described in 3.3.2 of
    Simon Baker, Daniel Scharstein, J. P. Lewis, Stefan Roth, Michael J. Black, and Richard Szeliski. A Database and Evaluation Methodology
     for Optical Flow, International Journal of Computer Vision, 92(1):1-31, March 2011.

    :param iflow: forwarded-warped (from flow0) at position t
    :param frame0: input image frame 0
    :param frame1: input image frame 1
    :param occ0: occlusion mask of frame 0
    :param occ1: occlusion mask of frame 1
    :param t: interpolated position t
    :return: interpolated image at position t in the middle of the 2 input frames
    '''
    h,w,_ = frame0.shape

    iframe = np.zeros_like(frame0).astype(np.float32)
    t0 = t
    t1 = 1.0 -t

    for y in range (h):
        for x in range (w):
            fx = iflow[y,x,0].astype(np.float32)
            fy = iflow[y,x,1].astype(np.float32)

            if unknown_flow(fy,fx):
                iframe[y,x,:]=0
                continue
            x0 = np.float32(x)-t0*fx
            y0 = np.float32(y)-t0*fy
            x1 = np.float32(x)+t1*fx
            y1 = np.float32(y)+t1*fy

            p0, inB0 = bilinear(frame0,x0,y0)
            p1, inB1 = bilinear(frame1,x1,y1)

            o0, _ = bilinear(occ0,x0,y0)
            o1, _ = bilinear(occ1,x1,y1)
            oc0 = round(o0)
            oc1 = round(o1)

            if inB0 and inB1 and oc0 ==0 and oc1==0:
                w0=t1
                w1=t0
            else:
                if oc0>oc1:
                    w0 = 0
                    w1 = 1
                else:
                    w0 = 1
                    w1 = 0 
            iframe[y,x,:] = np.int32(w0*p0 + w1*p1+0.5)

    return iframe

def blur(im):
    '''
    blur using a gaussian kernel [5,5] using opencv function: cv.GaussianBlur, sigma=0
    :param im:
    :return updated im:
    '''
    
    return cv.GaussianBlur(im,(5,5),0)


def internp(frame0, frame1, t=0.5, flow0=None):
    '''
    :param frame0: beggining frame
    :param frame1: ending frame
    :return frame_t: an interpolated frame at time t
    '''
    print('==============================')
    print('===== interpolate an intermediate frame at t=',str(t))
    print('==============================')

    # ==================================================
    # ===== 1/ find the optical flow between the two given images: from frame0 to frame1,
    #  if there is no given flow0, run opencv function to extract it
    # ==================================================
    if flow0 is None:
        i1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        i2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        flow0 = cv.calcOpticalFlowFarneback(i1, i2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # ==================================================
    # ===== 2/ find holes in the flow
    # ==================================================
    holes0 = find_holes(flow0)
    pickle.dump(holes0,open('holes0.step2.data','wb'))  # save your intermediate result
    # ====== score
    holes0       = pickle.load(open('holes0.step2.data','rb')) # load your intermediate result
    holes0_step2 = pickle.load(open('holes0.step2.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes0-holes0_step2))
    print('holes0_step2',diff)

    # ==================================================
    # ===== 3/ fill in any hole using an outside-in strategy
    # ==================================================
    flow0 = holefill(flow0,holes0)
    pickle.dump(flow0, open('flow0.step3.data', 'wb')) # save your intermediate result
    # ====== score
    flow0       = pickle.load(open('flow0.step3.data', 'rb')) # load your intermediate result
    flow0_step3 = pickle.load(open('flow0.step3.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow0-flow0_step3))
    print('flow0_step3',diff)

    # ==================================================
    # ===== 5/ estimate occlusion mask
    # ==================================================
    occ0, occ1 = occlusions(flow0,frame0,frame1)
    pickle.dump(occ0, open('occ0.step5.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step5.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step5.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step5.data', 'rb')) # load your intermediate result
    occ0_step5  = pickle.load(open('occ0.step5.sample', 'rb')) # load sample result
    occ1_step5  = pickle.load(open('occ1.step5.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step5 - occ0))
    print('occ0_step5',diff)
    diff = np.sum(np.abs(occ1_step5 - occ1))
    print('occ1_step5',diff)

    # ==================================================
    # ===== step 6/ blur occlusion mask
    # ==================================================
    for iblur in range(0,BLUR_OCC):
        occ0 = blur(occ0)
        occ1 = blur(occ1)
    pickle.dump(occ0, open('occ0.step6.data', 'wb')) # save your intermediate result
    pickle.dump(occ1, open('occ1.step6.data', 'wb')) # save your intermediate result
    # ===== score
    occ0        = pickle.load(open('occ0.step6.data', 'rb')) # load your intermediate result
    occ1        = pickle.load(open('occ1.step6.data', 'rb')) # load your intermediate result
    occ0_step6  = pickle.load(open('occ0.step6.sample', 'rb')) # load sample result
    occ1_step6  = pickle.load(open('occ1.step6.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(occ0_step6 - occ0))
    print('occ0_step6',diff)
    diff = np.sum(np.abs(occ1_step6 - occ1))
    print('occ1_step6',diff)

    # ==================================================
    # ===== step 7/ forward-warp the flow to time t to get flow_t
    # ==================================================
    flow_t = interpflow(flow0, frame0, frame1, t)
    pickle.dump(flow_t, open('flow_t.step7.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step7.data', 'rb')) # load your intermediate result
    flow_t_step7 = pickle.load(open('flow_t.step7.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step7))
    print('flow_t_step7',diff)

    # ==================================================
    # ===== step 8/ find holes in the estimated flow_t
    # ==================================================
    holes1 = find_holes(flow_t)
    pickle.dump(holes1, open('holes1.step8.data', 'wb')) # save your intermediate result
    # ====== score
    holes1       = pickle.load(open('holes1.step8.data','rb')) # load your intermediate result
    holes1_step8 = pickle.load(open('holes1.step8.sample','rb')) # load sample result
    diff = np.sum(np.abs(holes1-holes1_step8))
    print('holes1_step8',diff)

    # ===== fill in any hole in flow_t using an outside-in strategy
    flow_t = holefill(flow_t, holes1)
    pickle.dump(flow_t, open('flow_t.step8.data', 'wb')) # save your intermediate result
    # ====== score
    flow_t       = pickle.load(open('flow_t.step8.data', 'rb')) # load your intermediate result
    flow_t_step8 = pickle.load(open('flow_t.step8.sample', 'rb')) # load sample result
    diff = np.sum(np.abs(flow_t-flow_t_step8))
    print('flow_t_step8',diff)

    # ==================================================
    # ===== 9/ inverse-warp frame 0 and frame 1 to the target time t
    # ==================================================
    frame_t = warpimages(flow_t, frame0, frame1, occ0, occ1, t)
    pickle.dump(frame_t, open('frame_t.step9.data', 'wb')) # save your intermediate result
    # ====== score
    frame_t       = pickle.load(open('frame_t.step9.data', 'rb')) # load your intermediate result
    frame_t_step9 = pickle.load(open('frame_t.step9.sample', 'rb')) # load sample result
    diff = np.sqrt(np.mean(np.square(frame_t.astype(np.float32)-frame_t_step9.astype(np.float32))))
    print('frame_t',diff)

    return frame_t


if __name__ == "__main__":

    print('==================================================')
    print('PSU CS 410/510, Winter 2022, HW3: video frame interpolation')
    print('==================================================')

    # ===================================
    # example:
    # python interp_skeleton.py frame0.png frame1.png flow0.flo frame05.png
    # ===================================
    # path_file_image_0 = sys.argv[1]
    # path_file_image_1 = sys.argv[2]
    # path_file_flow    = sys.argv[3]
    # path_file_image_result = sys.argv[4]

    path_file_image_0 = 'frame0.png'
    path_file_image_1 =  'frame1.png'
    path_file_flow    = 'flow0.flo'
    path_file_image_result = 'result.png'

    # ===== read 2 input images and flow
    frame0 = cv.imread(path_file_image_0)
    frame1 = cv.imread(path_file_image_1)
    flow0  = readFlowFile(path_file_flow)

    # ===== interpolate an intermediate frame at t, t in [0,1]
    frame_t= internp(frame0=frame0, frame1=frame1, t=0.5, flow0=flow0)
    cv.imwrite(filename=path_file_image_result, img=(frame_t * 1.0).clip(0.0, 255.0).astype(np.uint8))