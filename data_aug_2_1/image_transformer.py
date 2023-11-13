import numpy as np
import cv2 as cv
import util as util
#import data_augmentation_yolo.util as util
import random
import logging
import math

class_id_to_name = {
    0: 'person',
    1: 'bicycle',
    2: 'dog',
    3: 'snake'
    # Add more class mappings as needed
}

SCALE_MODIFIED_TH = 1000
logger = logging.getLogger(__name__)

def array_mult(A, B):
    if len(B) != len(A[0]) and len(A) != len(B[0]):
        return 'Invalid'

    result = [[0 for x in range(len(B[0]))] for y in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    result_arr = np.array(result)
    return result_arr

class SampleImgTransformer:
    def __init__(self, *, image, bg_color, bboxes, classes):
        size = np.shape(image)
        self.height = size[0]
        self.width = size[1] 
        self.channels = size[2]
        self.image = bg_color * np.ones(
            (self.height, self.width, self.channels), np.uint8
        )
        self.image[
            0 : self.height, 0 : self.width
        ] = np.copy(image[0 : size[0], 0 : size[1]])
        self.modified_flag = 0
        self.no_modified_flag = 0
        self.modified_image = np.copy(self.image)
        self.finalImgDraw = np.copy(self.modified_image)
        self.bboxes = np.copy(bboxes)
        self.modified_bboxes = np.copy(bboxes)
        self.classes = classes

    def addGaussianNoise(self, *, noiseMean, noiseVariance) -> None:
        """Adds Gaussian Noise to the image with a given mean and variance"""
        noiseSigma = noiseVariance ** 0.5

        height, width, _ = np.shape(self.modified_image)
        gaussImg = np.random.normal(
            noiseMean, noiseSigma, (height, width, self.channels)
        )
        self.modified_image = np.float32(self.modified_image)
        self.modified_image = (
            self.modified_image + gaussImg
        )
        self.finalImgDraw = np.copy(self.modified_image)
        return self.modified_image

    def addPeperSaltNoise(self,*, Threshold) -> None:
        image_width = self.modified_image.shape[1]
        image_height = self.modified_image.shape[0]
        a = np.random.randint(0,255, size = (image_height, image_width, 3))
        p_b = np.where(a[:,:,0] > Threshold, 255, 0)
        p_g = np.where(a[:,:,1] > Threshold, 255, 0)
        p_r = np.where(a[:,:,2] > Threshold, 255, 0)
        c = np.copy(np.int8(self.modified_image))
        self.modified_image[:,:,0] =np.float64(np.bitwise_or(p_b,c[:,:,0]))
        self.modified_image[:,:,1] =np.float64(np.bitwise_or(p_g,c[:,:,1]))
        self.modified_image[:,:,2] =np.float64(np.bitwise_or(p_r,c[:,:,2]))
        

    def get_wapper_M(self, *, width, height, angle):
        center = (width / 2, height / 2)
        scale = 1.0
        # 2.1获取M矩阵
        """
        M矩阵
        [
        cosA -sinA (1-cosA)*centerX+sinA*centerY
        sinA cosA  -sinA*centerX+(1-cosA)*centerY
        ]
        """
        M = cv.getRotationMatrix2D(center, angle, scale)
        new_H = int(width * math.fabs(math.sin(math.radians(angle))) + height * math.fabs(math.cos(math.radians(angle))))
        new_W = int(height * math.fabs(math.sin(math.radians(angle))) + width * math.fabs(math.cos(math.radians(angle))))
        # 2.3 平移
        M[0, 2] += (new_W - width) / 2
        M[1, 2] += (new_H - height) / 2

        return M, new_W, new_H
    
    def reset_modified_image(self):
        self.modified_image = self.image
        self.modified_bboxes = np.copy(self.bboxes)
        self.no_modified_flag = 0
        
    def affineRotate(self, *, maxXangle, bgColor=255) -> None:
        angle = np.random.uniform(-maxXangle, maxXangle)
        height, width, _ = np.shape(self.modified_image)
        bboxes = np.copy(self.modified_bboxes)
        process_image = np.copy(self.modified_image)
        M, nW, nH = self.get_wapper_M(width = width, height = height, angle = angle)
        self.modified_image = cv.warpAffine(
            process_image,
            M,
            (nW, nH),
            borderValue=(bgColor, bgColor, bgColor)
        )
        n = 0
        for bbox in bboxes:
                l_t_c = np.array([bbox[0],bbox[1],1]).T
                l_t_c = np.reshape(l_t_c, (3,1))
                l_t_c_r = array_mult(M, l_t_c)

                l_b_c = np.array([bbox[2],bbox[3],1]).T
                l_b_c = np.reshape(l_b_c, (3,1))
                l_b_c_r = array_mult(M, l_b_c)
                
                r_t_c = np.array([bbox[4],bbox[5],1]).T
                r_t_c = np.reshape(r_t_c, (3,1))
                r_t_c_r = array_mult(M, r_t_c)
                
                r_b_c = np.array([bbox[6], bbox[7],1]).T
                r_b_c = np.reshape(r_b_c, (3,1))                
                r_b_c_r = array_mult(M, r_b_c)
                
                left = min(l_t_c_r[0,0], l_b_c_r[0,0], r_t_c_r[0,0], r_b_c_r[0,0]) 
                top =  min(l_t_c_r[1,0], l_b_c_r[1,0], r_t_c_r[1,0], r_b_c_r[1,0]) 
                right = max(l_t_c_r[0,0], l_b_c_r[0,0], r_t_c_r[0,0], r_b_c_r[0,0]) 
                bottom = max(l_t_c_r[1,0], l_b_c_r[1,0], r_t_c_r[1,0], r_b_c_r[1,0]) 
                self.modified_bboxes[n,:] = np.array([left, top, left, bottom, right, top, right, bottom])
                n = n + 1
        print("newH,newW=",nH,nW)
        self.finalImgDraw = np.copy(self.modified_image)

    """ Get Perspective Projection Matrix """

    def get_M(self, *, theta, phi, gamma, dx, dy, dz):

        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )

        RY = np.array(
            [
                [np.cos(phi), 0, -np.sin(phi), 0],
                [0, 1, 0, 0],
                [np.sin(phi), 0, np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )

        RZ = np.array(
            [
                [np.cos(gamma), -np.sin(gamma), 0, 0],
                [np.sin(gamma), np.cos(gamma), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Composed rotation matrix with (RX, RY, RZ)
        R1 = array_mult(RX, RY)
        R = array_mult(R1, RZ)
        # Translation matrix
        T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])

        # Final transformation matrix
        A11 = array_mult(R, A1)
        A12 = array_mult(T, A11)
        A_R = array_mult(A2, A12)
        return A_R

    def perspectiveTransform(self, *, maxXangle, maxYangle, maxZangle, bgColor=255, bounding_boxes, class_nums, bkImg):
        #the location of perspective transform is:
        angX = np.random.uniform(-maxXangle, maxXangle)
        angY = np.random.uniform(-maxYangle, maxYangle)
        angZ = np.random.uniform(-maxZangle, maxZangle)
        rtheta, rphi, rgamma = util.get_rad(angX, angY, angZ)

        d = np.sqrt(self.modified_image.shape[0] ** 2 + self.modified_image.shape[1] ** 2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        mat = self.get_M(
            theta=rtheta, phi=rphi, gamma=rgamma, dx=0, dy=0, dz=dz
        )
        i_w_h_arr = np.array([self.modified_image.shape[1], self.modified_image.shape[0], 1]).T
        i_w_h_arr = np.reshape(i_w_h_arr, (3,1))
        i_w_h_p = array_mult(mat, i_w_h_arr)
        npW = int(i_w_h_p[0,0] / i_w_h_p[2,0])
        npH = int(i_w_h_p[1,0] / i_w_h_p[2,0])
        
        self.modified_image = cv.warpPerspective(
            self.modified_image.copy(),
            mat,
            (npW, npH),
            borderMode=cv.BORDER_CONSTANT,
            borderValue=(bgColor, bgColor, bgColor),
        )
        self.finalImgDraw = np.copy(self.modified_image)
        bboxes = np.copy(self.modified_bboxes)
        n = 0
        for bbox in bboxes:
            l_t_c = np.array([bbox[0],bbox[1],1]).T
            l_t_c = np.reshape(l_t_c, (3,1))
            l_t_c_p = array_mult(mat, l_t_c)
            l_t_c_p_n = np.zeros((3,1), dtype = np.float64)
            l_t_c_p_n[0,0] = l_t_c_p[0,0] / l_t_c_p[2,0]
            l_t_c_p_n[1,0] = l_t_c_p[1,0] / l_t_c_p[2,0]
            l_t_c_p_n[2,0] = l_t_c_p[2,0] / l_t_c_p[2,0]
            
            l_b_c = np.array([bbox[2], bbox[3],1]).T
            l_b_c = np.reshape(l_b_c, (3,1))
            l_b_c_p = array_mult(mat, l_b_c)
            l_b_c_p_n = np.zeros((3,1), dtype = np.float64)
            l_b_c_p_n[0,0] = l_b_c_p[0,0] / l_b_c_p[2,0]
            l_b_c_p_n[1,0] = l_b_c_p[1,0] / l_b_c_p[2,0]
            l_b_c_p_n[2,0] = l_b_c_p[2,0] / l_b_c_p[2,0]
            
            r_t_c = np.array([bbox[4], bbox[5],1]).T
            r_t_c = np.reshape(r_t_c, (3,1))
            r_t_c_p = array_mult(mat, r_t_c)
            r_t_c_p_n = np.zeros((3,1), dtype = np.float64)
            r_t_c_p_n[0,0] = r_t_c_p[0,0] / r_t_c_p[2,0]
            r_t_c_p_n[1,0] = r_t_c_p[1,0] / r_t_c_p[2,0]
            r_t_c_p_n[2,0] = r_t_c_p[2,0] / r_t_c_p[2,0]
            
            r_b_c = np.array([bbox[6], bbox[7],1]).T
            r_b_c = np.reshape(r_b_c, (3,1))                
            r_b_c_p = array_mult(mat, r_b_c)
            r_b_c_p_n = np.zeros((3,1), dtype = np.float64)
            r_b_c_p_n[0,0] = r_b_c_p[0,0] / r_b_c_p[2,0]
            r_b_c_p_n[1,0] = r_b_c_p[1,0] / r_b_c_p[2,0]
            r_b_c_p_n[2,0] = r_b_c_p[2,0] / r_b_c_p[2,0]
            
            left = int(min(l_t_c_p_n[0,0],l_b_c_p_n[0,0], r_t_c_p_n[0,0], r_b_c_p_n[0,0]))
            right = int(max(l_t_c_p_n[0,0],l_b_c_p_n[0,0], r_t_c_p_n[0,0], r_b_c_p_n[0,0]))
            top = int(min(l_t_c_p_n[1,0],l_b_c_p_n[1,0], r_t_c_p_n[1,0], r_b_c_p_n[1,0]))
            bottom = int(max(l_t_c_p_n[1,0],l_b_c_p_n[1,0], r_t_c_p_n[1,0], r_b_c_p_n[1,0]))
            
            self.modified_bboxes[n,:] = np.array([left, top, left, bottom, right, top, right, bottom])
            n = n + 1
        # cv.imshow("modified",self.modified_image)
        # cv.waitKey(1000)

    def sharpenImage(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.modified_image = cv.filter2D(self.modified_image, -1, kernel)
        self.finalImgDraw = np.copy(self.modified_image)

    def scaleImage(self, *, scale):
        bboxes = np.copy(self.modified_bboxes)
        n = 0
        for bbox in bboxes:
            # if (n > 0): #only rescale once, if bbox number larger than 1, do not rescale.
            #     break
            bbox_area = (bbox[6] - bbox[0]) * (bbox[7] - bbox[1]) # if bbox area small than SCALE_MODIFIED_TH, do not rescale
            print("BBOX area = ", bbox_area)
            if (bbox_area < SCALE_MODIFIED_TH):
                print("BBox too small! No rescale!")
                self.no_modified_flag = 1
                break
            n = n + 1
        if self.no_modified_flag == 0:
            n = 0
            for bbox in bboxes:
                bbox_s =np.multiply(bbox, scale)
                self.modified_bboxes[n,:] = bbox_s.astype(int)
                n = n + 1
            self.modified_image = cv.resize(self.modified_image, None, fx=scale, fy=scale)
            self.finalImgDraw = np.copy(self.modified_image)

    def modifybrightness(self, *, hsv_scale, percent=1):
        process_img = np.zeros_like(self.modified_image, dtype = np.uint8)
        i_a_not_m_1 = self.modified_image != -1
        process_img[i_a_not_m_1] = self.modified_image[i_a_not_m_1]
        
        imageHSV = cv.cvtColor(process_img, cv.COLOR_BGR2HSV)
        a = np.copy(imageHSV)
        a[:, 2] = hsv_scale * a[:, 2]
        imageHSV = np.copy(a)
        process_img = cv.cvtColor(imageHSV, cv.COLOR_HSV2BGR)
        self.modified_image[i_a_not_m_1] = process_img[i_a_not_m_1]
        self.finalImgDraw = np.copy(self.modified_image)
        cv.waitKey(1000)
    def FusionBackground(self,*, bkg_img):
        if (self.no_modified_flag == 1):
            self.no_modified_flag = 0
            return
        process_img = np.copy(bkg_img)  
        #防止 self.modified_image > bkg_img 造成影像拼貼錯誤 #20231113 justin
        if self.modified_image.shape[0] > bkg_img.shape[0] :
            crop_img = self.modified_image[0:bkg_img.shape[0], 0: self.modified_image.shape[1]]
            self.modified_image = np.copy(crop_img)
        if self.modified_image.shape[1] > bkg_img.shape[1] :  
            crop_img = self.modified_image[0:self.modified_image.shape[0], 0: bkg_img.shape[1]]
            self.modified_image = np.copy(crop_img)            
        # Calculate the centers of A and B
        center_modified_image = (self.modified_image.shape[1] // 2, self.modified_image.shape[0] // 2)
        center_bkg_img = (process_img.shape[1] // 2, process_img.shape[0] // 2)
        
        # Calculate the shift required to align the centers
        shift_step = (center_bkg_img[0] - center_modified_image[0], center_bkg_img[1] - center_modified_image[1])
        
        # Determine the region of B where A should be placed
        start_x = shift_step[0]
        end_x = start_x + self.modified_image.shape[1]
        start_y = shift_step[1]
        end_y = start_y + self.modified_image.shape[0]
        #因為有防止 self.modified_image > bkg_img 造成影像拼貼錯誤 start_x < 0 start_y < 0 不會發生，故略去
        # # Ensure that A does not go out of bounds
        # if start_x < 0:
        #     end_x += abs(start_x)
        #     start_x = 0
        # if start_y < 0:
        #     end_y += abs(start_y)
        #     start_y = 0
        
        # Place array A in the center of array B
        m = process_img[start_y:end_y, start_x:end_x, :] #若strat_x, stary_y < 0, 切割出來的矩陣大小會與self.modified_image 不合!!
        i_a_not_m_1 = self.modified_image != -1
        m[i_a_not_m_1] = self.modified_image[i_a_not_m_1]
        
        process_img[start_y:end_y, start_x:end_x, :] = m
        self.modified_image = np.copy(process_img)
        self.finalImgDraw = np.copy(self.modified_image)
        bboxes = np.copy(self.modified_bboxes)
        n = 0
        for bbox in bboxes:
            bbox[0]
            self.modified_bboxes[n,0] = bbox[0] + shift_step[0]
            self.modified_bboxes[n,2] = bbox[2] + shift_step[0]
            self.modified_bboxes[n,4] = bbox[4] + shift_step[0]
            self.modified_bboxes[n,6] = bbox[6] + shift_step[0]
            self.modified_bboxes[n,1] = bbox[1] + shift_step[1]
            self.modified_bboxes[n,3] = bbox[3] + shift_step[1]
            self.modified_bboxes[n,5] = bbox[5] + shift_step[1]
            self.modified_bboxes[n,7] = bbox[7] + shift_step[1]
            n = n + 1
        
    def finalImgOutput(self):
        return self.modified_image, self.modified_bboxes, self.classes
    def finalROIImgDraw(self):
        n = 0
        for bbox in self.modified_bboxes:
            class_name = class_id_to_name.get(int(self.classes[n]), 'Unknown')
            # Draw a rectangle with the class name on the image to visualize the ROI
            color = (0, 255, 0)  # Green color (BGR format)
            thickness = 2
            left = int(bbox[0])
            top = int(bbox[1])
            right = int(bbox[6])
            bottom = int(bbox[7])
            print(left, top, right, bottom)
            self.finalImgDraw = cv.rectangle(self.finalImgDraw, (left, top), (right, bottom), color, thickness)
            text = f'{class_name}: {self.classes[n]}'
            cv.putText(self.finalImgDraw, text, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            n = n + 1
        return self.finalImgDraw
    
