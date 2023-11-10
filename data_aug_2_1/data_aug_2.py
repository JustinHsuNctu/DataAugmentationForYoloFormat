import numpy as np
import cv2 as cv
import os
import glob
from image_transformer import SampleImgTransformer
from bkg_files_loader import BackgroundFileLoader

BACKGROUND_COLOR = -1
OUTPUT_PER_SAMPLE = 3
MAX_X_ANGLE = 30
MAX_Y_ANGLE = 60
MAX_Z_ANGLE = 60
MAX_AFFINE_ANGLE = 60
PEPER_SALT_NOISE_RANDOM_TH = 250
if __name__ == "__main__":
    outputfolder = "./data/output/"
    dataPath = "./data/train/"
    backgroundFilePath = "./data/bkimg/"
    bkgFileLoader = BackgroundFileLoader()
    bkgFileLoader.loadbkgFiles(backgroundFilePath)
    
    if not (os.path.isdir(outputfolder)):
        os.makedirs(outputfolder)

    image_files = glob.glob(os.path.join(dataPath + "images/", "*.jpg"))
    
    # Iterate through the pairs of paths using zip
    for sampleImgPath in image_files:
        imgfilenameWithExt = os.path.split(sampleImgPath)[1]
        imgfilename = os.path.splitext(imgfilenameWithExt)[0]
        labelfilename = os.path.join(dataPath + "labels/", imgfilename+".txt")
        org_Img = cv.imread(sampleImgPath)
        org_Img = org_Img.astype(np.uint8)
        image_height, image_width, _ = np.shape(org_Img)
        count = 0
        bounding_boxes = []
        class_nums = []
        # Open the text file and read its lines
        with open(labelfilename, 'r') as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                parts = line.split()  # Split the line by spaces
                if len(parts) == 5:
                    class_num, center_x, center_y, b_width, b_height = map(float, parts)
                    left = int((center_x - b_width / 2) * image_width) 
                    top = int((center_y - b_height / 2) * image_height)
                    right = int((center_x + b_width / 2) * image_width)
                    bottom = int((center_y + b_height / 2) * image_height)
                    bounding_boxes.append(np.array([left, top, left, bottom, right, top, right, bottom]))
                    class_nums.append(class_num)
        image_modifier = SampleImgTransformer(
            image=org_Img, bg_color=BACKGROUND_COLOR, bboxes = bounding_boxes, classes = class_nums
        )


        while count < OUTPUT_PER_SAMPLE:
            bounding_boxes.clear()
            image_modifier.reset_modified_image()
            bkg_img = bkgFileLoader.bkgImgList[np.random.randint(0, bkgFileLoader.count)]
            # image_modifier.sharpenImage()
            # image_modifier.addGaussianNoise(noiseMean = 0.0, noiseVariance = 0.1)
            # image_modifier.addPeperSaltNoise(Threshold = PEPER_SALT_NOISE_RANDOM_TH)
            image_modifier.affineRotate(maxXangle=MAX_AFFINE_ANGLE, bgColor=BACKGROUND_COLOR)
            image_modifier.perspectiveTransform(maxXangle=MAX_X_ANGLE,maxYangle=MAX_Y_ANGLE,maxZangle=MAX_Z_ANGLE,bgColor=BACKGROUND_COLOR, bounding_boxes =[], class_nums=[], bkImg=bkg_img)
            scale = np.random.uniform(0.1,0.3)
            image_modifier.scaleImage(scale=scale)

            hsv_scale = np.random.uniform(0.5, 1)
            image_modifier.modifybrightness(hsv_scale=hsv_scale)
            image_modifier.FusionBackground(bkg_img = bkg_img)
            finalImg, bboxes, classes = image_modifier.finalImgOutput()
            finalDrawROIimg = image_modifier.finalROIImgDraw()
            finalImgName = imgfilename + "_" + str(count)
            finalROIImgName = imgfilename + "_" + str(count) + "_ROI"
            print("finalImgName:", finalImgName)
            cv.imwrite(outputfolder + "images/" + finalImgName + ".jpg", finalImg)
            cv.imwrite(outputfolder + finalROIImgName + ".jpg", finalDrawROIimg)    
            with open(outputfolder + "labels/" + finalImgName + ".txt", "w") as f:
                for class_c, bbox in zip(classes, bboxes):
                    center_x = np.float64(((bbox[0] + bbox[6]) / 2) / finalImg.shape[1])
                    center_y = np.float64(((bbox[1] + bbox[7]) / 2) / finalImg.shape[0])
                    width = np.float64((bbox[6] - bbox[0])  / finalImg.shape[1])
                    height = np.float64((bbox[7] - bbox[1]) / finalImg.shape[0]) 
                    class_c = int(class_c)
                    class_bboxes_str = str(class_c) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height)
                    f.write(class_bboxes_str)

            count = count + 1

