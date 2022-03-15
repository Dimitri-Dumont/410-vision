import math
import cv2 as cv
import sys
import numpy as np
import random


def ex_find_homography_ransac(
    list_pairs_matched_keypoints,
    threshold_ratio_inliers=0.85,
    threshold_reprojtion_error=3,
    max_num_trial=1000,
):
    """
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points, transform the first set of feature point to the second (e.g. warp image 1 to image 2)
    :param list_pairs_matched_keypoints: has the format as a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],....]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples, accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojtion_error: threshold of reprojection error (measured as euclidean distance, in pixels) to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to do take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    """
    best_H = None
    best_ratio = 0
    for i in range(max_num_trial):
        a, b, c, d = random.choices(list_pairs_matched_keypoints, k=4)
        x1, y1, xp1, yp1 = a[0][0], a[0][1], a[1][0], a[1][1]
        x2, y2, xp2, yp2 = b[0][0], b[0][1], b[1][0], b[1][1]
        x3, y3, xp3, yp3 = c[0][0], c[0][1], c[1][0], c[1][1]
        x4, y4, xp4, yp4 = d[0][0], d[0][1], d[1][0], d[1][1]

        A = np.array([
            [-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1],
            [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
            [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2],
            [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
            [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3],
            [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
            [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4],
            [0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]
        ])
        U, S, V = np.linalg.svd(np.array(A, np.float32))

        H = V[-1, :].reshape(3, 3)/V[-1, -1]
        inLiners = NumInliners(H, list_pairs_matched_keypoints)
        ratio = inLiners/len(list_pairs_matched_keypoints)
        if ratio > threshold_ratio_inliers:
            # if best_ratio < ratio:
            #     best_ratio = ratio
            #     best_H = H

            print(ratio)
            return H


def NumInliners(H, matches):
    inLine = 0
    for a in matches:
        x1, y1, x2, y2 = a[0][0], a[0][1], a[1][0], a[1][1]

        p1h = np.transpose([x1, y1, 1])
        p2h = np.transpose([x2, y2, 1])
        p1p = np.dot(H, p1h)
        
        # standard predicted form
        p1p = p1p / p1p[2]
        error = np.linalg.norm(p2h-p1p)
        if error < 3:
            inLine += 1

    return inLine


def ex_extract_and_match_feature(img_1, img_2, ratio_robustness=0.7):
    """
    1/ extract SIFT feature from image 1 and image 2,
    2/ use a bruteforce search to find pairs of matched features: for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points
    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_robustness: ratio for the robustness test
    :return list_pairs_matched_keypoints: has the format as list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    """
    # ==============================
    # ===== 1/ extract features from input image 1 and image 2
    # ==============================
    gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    #img_kp = cv.drawKeypoints(gray, kp, img_1)

    gray2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    #img_kp2 = cv.drawKeypoints(gray2, kp2, img_2)

    # ==============================
    # ===== 2/ use bruteforce search to find a list of pairs of matched feature points
    # ==============================

    list_pairs_matched_keypoints = []

    for i in range(len(kp)):
        tmplist = []
        for j in range(len(kp2)):
            dist = np.linalg.norm(des[i]-des2[j])
            tmplist.append([j, dist])

        tmplist.sort(key=lambda x: x[1])
        ratio = tmplist[0][1]/tmplist[1][1]
        if (ratio < ratio_robustness):
            p = kp[i]
            q = kp2[tmplist[0][0]]
            p1x = p.pt[0]
            p1y = p.pt[1]
            p2x = q.pt[0]
            p2y = q.pt[1]
            list_pairs_matched_keypoints.append([[p1x, p1y], [p2x, p2y]])
    print(len(list_pairs_matched_keypoints))
    return list_pairs_matched_keypoints


def ex_warp_blend_crop_image(img_1, H_1, img_2):
    """
    1/ warp image img_1 using the homography H_1 to align it with image img_2 (using backward warping and bilinear resampling)
    2/ stitch image img_1 to image img_2 and apply average blending to blend the 2 images into a single panorama image
    3/ find the best bounding box for the resulting stitched image
    :param img_1:
    :param H_1:
    :param img_2:
    :return img_panorama: resulting panorama image
    """
    img_panorama = None
    # =====  use a backward warping algorithm to warp the source
    # 1/ to do so, we first create the inverse transform; 2/ use bilinear interpolation for resampling
    maskInput = np.ones(img_1.shape[0:2], np.float32)
    h = img_1.shape[0]
    w = img_1.shape[1]
    inv_h = np.linalg.inv(H_1)
    canvas_im1 = np.zeros((3*h, 3*w, 3), dtype=np.float32)
    mask_im1 = np.zeros((3*h, 3*w), dtype=np.float32)

    for Y in range(-h, 2*h):
        for X in range(-w, 2*w):
            DstCord_H = np.array([X, Y, 1.0], np.float32)
            srcCord_H = np.dot(inv_h, DstCord_H)
            srcCordStd = srcCord_H/srcCord_H[2]
            if((srcCordStd[0] > 0.0 and srcCordStd[0] < w-1) and (srcCordStd[1] > 0.0 and srcCordStd[1] < h-1)):
                i = int(math.floor(srcCordStd[0]))
                j = int(math.floor(srcCordStd[1]))
                a = srcCordStd[0] - i
                b = srcCordStd[1] - j
                canvas_im1[Y+h, X+w] = (1-a)*(1-b) * img_1[j,i] + a*(1-b) * img_1[j,i+1] + (a * b * img_1[j+1,i+1]) + ((1-a)*b * img_1[j+1,i])  
                mask_im1[Y+h, X+w] = (1-a)*(1-b) * maskInput[j,i] + a*(1-b) * maskInput[j,i+1] + (a * b * maskInput[j+1,i+1]) + ((1-a)*b * maskInput[j+1,i])  

    canvas_im2 = np.zeros((3*h, 3*w, 3), dtype=np.float32)
    mask_im2 = np.zeros((3*h, 3*w), dtype=np.float32)
    canvas_im2[h:h*2, w:w*2] = img_2
    mask_im2[h:h*2, w:w*2] = 1.0
    img = canvas_im1 + canvas_im2
    mask = mask_im1+mask_im2
    mask = np.tile(np.expand_dims(mask, 2), (1, 1, 3))
    img = np.divide(img, mask)

    

    # ===== blend images: average blending
    mask_check = 1.0-np.float32(mask[:,:,0]>0)
    Check_h = np.sum(mask_check[:, :], 1)
    Check_w = np.sum(mask_check[:, :],0)
    left = np.min(np.where(Check_w < h*3))
    Right = np.max(np.where(Check_w < h*3))
    Bottom = np.min(np.where(Check_h < w*3))
    top = np.max(np.where(Check_h < w*3))
    img_panorama = img[Bottom:top,left:Right]

    cv.imshow("poop", img_panorama)
    cv.waitKey(0)

    return img_panorama


def stitch_images(img_1, img_2):
    """
    :param img_1: input image 1. We warp this image to align and stich it to the image 2
    :param img_2: is the reference image. We will not warp this image
    :return img_panorama: the resulting stiched image
    """
    print("==============================")
    print("===== stitch two images to generate one panorama image")
    print("==============================")

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = ex_extract_and_match_feature(
        img_1=img_1, img_2=img_2, ratio_robustness=0.7
    )

    # ===== use RANSAC algorithm to find homography to warp image 1 to align it to image 2
    H_1 = ex_find_homography_ransac(
        list_pairs_matched_keypoints,
        threshold_ratio_inliers=0.85,
        threshold_reprojtion_error=3,
        max_num_trial=1000,
    )
    print(H_1)

    # ===== warp image 1, blend it with image 2 using average blending to produce the resulting panorama image
    img_panorama = ex_warp_blend_crop_image(img_1=img_1, H_1=H_1, img_2=img_2)

    return img_panorama


if __name__ == "__main__":
    print("==================================================")
    print("PSU CS 410/510, Winter 2022, HW2: image stitching")
    print("==================================================")

    path_file_image_1 = sys.argv[1]
    path_file_image_2 = sys.argv[2]
    path_file_image_result = sys.argv[3]
   
    # ===== read 2 input images
    img_1 = cv.imread(path_file_image_1)
    img_2 = cv.imread(path_file_image_2)

    # ===== create a panorama image by stitch image 1 to image 2
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)
    #stitch_images(img_1, img_2)
    # ===== save panorama image
    cv.imwrite(
        filename=path_file_image_result,
        img=(img_panorama).clip(0.0, 255.0).astype(np.uint8),
    )
