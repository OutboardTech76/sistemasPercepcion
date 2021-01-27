import cv2
import pyrealsense2 as rs
import numpy as np
import argparse
import json
import sys
import os
from openpose import pyopenpose as op


# Point class with x,y,z values and point as a tuple (x,y)
class Point():
    def __init__(self,pt):
        # Convert to numpy float or there will be error when printing lines
        pt =  np.float32(pt)
        self.npPoint = pt[0:2]
        self.x = pt[0]
        self.y = pt[1]
        self.z = 0
        self.score = pt[2]
        self.point = (pt[0], pt[1])
        if self.point > (0,0):
            self.draw = True
        else:
            self.draw = False

class Pose:
    def __init__(self,keypoints):
        self.rightShoulder = Point(keypoints[2])
        self.rightElbow = Point(keypoints[3])
        self.rightWrist = Point(keypoints[4])
        self.leftShoulder = Point(keypoints[5])
        self.leftElbow = Point(keypoints[6])
        self.leftWrist = Point(keypoints[7])
        self.center = Point(keypoints[8])
 
# Classes used to define hands and fingers
# handKeypoint[0] -> right hand
# handKeypoint[1] -> left hand
class Hand:
    def __init__(self, keypoints):
        # Second dimension limits num person
        right = keypoints[1][0]
        left = keypoints[0][0]
         
        self.rightBase = Point(right[0])
        self.rightThumb = Point(right[4])
        self.rightIndex = Point(right[8])
        self.rightMiddle = Point(right[12])
        self.rightRing = Point(right[16])
        self.rightPinky = Point(right[20])
         
        self.leftBase = Point(left[0])
        self.leftThumb = Point(left[4])
        self.leftIndex = Point(left[8])
        self.leftMiddle = Point(left[12])
        self.leftRing = Point(left[16])
        self.leftPinky = Point(left[20])

    # Mesures distance between base and fingers returning the biggest
    def largerMeasure(self) -> float:
        #np.linalg.norm -> euclidean distance numpy
        norm = np.linalg.norm
        dist1 = abs(norm(self.rightBase.npPoint - self.rightThumb.npPoint))
        dist2 = abs(norm(self.rightBase.npPoint - self.rightIndex.npPoint))
        dist3 = abs(norm(self.rightBase.npPoint - self.rightMiddle.npPoint))
        dist4 = abs(norm(self.rightBase.npPoint - self.rightRing.npPoint))
        dist5 = abs(norm(self.rightBase.npPoint - self.rightPinky.npPoint))

        return max(dist1, dist2, dist3, dist4, dist5)
     
def setReferenceFrame(data):
    try:
        peopleNum = len(data.poseKeypoints)

        # Detectmore than one person
        # for i in range(peopleNum):
            # pose = Pose(data.poseKeypoints[i])

        # Just one person at a time
        if peopleNum > 0:
            pose = Pose(data.poseKeypoints[0])
            hand = Hand(data.handKeypoints)
             
            # poseChanged, handChanged = convertToRefFrame(hand, pose, depth_frame)
            
            return pose, hand
    except:
        pass
    
def setParams():
    # args = parser.parse_args()
    params = dict()
    params["model_folder"] = "/home/paco/openpose/models"
    params["model_pose"] = "BODY_25"
    params["hand"] = True
    params["face"] = False
    # Reduce net resolution in order to avoid memory problems
    params["net_resolution"] = "-1x128"
    # params["write_json"] = args.jsonPath 
    return params

def createMaskFormDepth(depth, thresh, tp):
    cv2.threshold(depth, thresh, 255, tp, depth)
    depth = cv2.dilate(depth, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4), (3,3)))
    depth = cv2.erode(depth, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7),(6,6)))
    return depth

def removeBackground(depth, pose, img, colorImg) -> np.ndarray:
    centerDist = depth.get_distance(int(pose.center.x), int(pose.center.y))
    armDist = depth.get_distance(int(pose.rightWrist.x), int(pose.rightWrist.y))
     
    maxValueDist = centerDist + 0.3
    # maxValueDist2 = centerDist + (abs(centerDist - armDist))
    # maxValueDist = max(maxValueDist1, maxValueDist2)
    minValueDist = centerDist - 0.2

    kernel = np.ones((3,3), np.uint8)
     
    h, w = img.shape
    imgData = np.asarray(img, dtype='uint8')
    auxImg = np.copy(imgData)
    for y in range(0,h):
        for x in range(0,w):
            dist = depth.get_distance(x, y)
            if dist >= maxValueDist:
                auxImg[y, x] = 0
            elif dist == 0:
                auxImg[y, x] = 0
            elif dist < minValueDist:
                auxImg[y, x] = 0
            # else:
                # auxImg[y, x] = 0

    # Use gaussian for noise reduction
    auxImg = cv2.GaussianBlur(auxImg, (5,5),cv2.BORDER_DEFAULT)
    # Apply threshold using Otsu algorith
    _, auxImg = cv2.threshold(auxImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Open image to reduce noise. After dilate to extract sure background 
    openImg = cv2.morphologyEx(auxImg,cv2.MORPH_OPEN,kernel, iterations = 2)
    sureBg = cv2.dilate(openImg,kernel,iterations=3)
    # Finding sure foreground area using euclidean distance with (3,3) mask and threshold with 1/10 max value
    distTransform = cv2.distanceTransform(openImg, cv2.DIST_L2, 3)
    _, sureFg = cv2.threshold(distTransform, 0.10*distTransform.max(), 255, 0)
     
    # Finding unknown region. Difference between sureBg and sureFg
    sureFg = np.uint8(sureFg)
    unknownRegion = cv2.subtract(sureBg, sureFg)
     
    # Marker labelling
    _, markers = cv2.connectedComponents(sureFg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknownRegion==255] = 0
     
    # Apply watershed algorith to extract siluette in image
    
    markers = cv2.watershed(colorImg, markers) 
    colorImg[markers == -1] = [255,0,0]
    
    markImg = np.zeros(auxImg.shape, dtype='uint8')
    markImg[markers == -1] = 255
     
    img2 = cv2.bitwise_and(markImg, auxImg)
    
    auxImg = cv2.bitwise_not(auxImg)
    
    cv2.imshow("fg", markImg)
    cv2.imshow("2", auxImg)
      
    # auxImg = cv2.erode(auxImg, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)))
    # auxImg = cv2.dilate(auxImg, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)))

    return auxImg
 
def moments(img):
    threshold = 100
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, dstCn = cv2.CV_8UC1)
    
    canny_output = cv2.Canny(img, threshold, threshold * 2)
    print("1")
    cv2.imshow("1", canny_output)
    
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("2")
    
    # Get the moments
    mu = [None]*len(contours)
    print("3")
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])
    # Get the mass centers
    print("4")
    mc = [None]*len(contours)
    print("5")
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    # Draw contours
    print("6")
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    print("7")
    
    for i in range(len(contours)):
        color = (0, 255, 0)
        cv2.drawContours(drawing, contours, i, color, 2)
        cv2.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)
    
    
    cv2.imshow('Contours', drawing)
 
 
 
def kMeans(img):
    auxImg = np.copy(img)
    pixelValues = auxImg.reshape((-1, 3))
    pixelValues = np.float32(pixelValues)

    stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    nAttemps = 10
    centroid = cv2.KMEANS_RANDOM_CENTERS
    clusters = 6

    _, labels, centers = cv2.kmeans(pixelValues, clusters, None, stop, nAttemps, centroid)
    centers = np.uint8(centers)
    segmentedData = centers[labels.flatten()]

    segmentedImg = segmentedData.reshape(auxImg.shape)
    cv2.imshow("K", segmentedImg)
  



if __name__ == '__main__':
     
    # Openpose config
    params = setParams()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Realsense config
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    pipeline.start(config)

    fgModel = np.zeros((1, 65), dtype='float')
    bgModel = np.zeros((1, 65), dtype='float')

    colorize = rs.colorizer()
    colorize.set_option(rs.option.color_scheme,2)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
             
            depth_colormap = np.asanyarray(colorize.colorize(depth_frame).get_data())
            if not depth_frame or not color_frame:
                continue

            datum = op.Datum()
             
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
             
            img = np.copy(depth_colormap)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, dstCn = cv2.CV_8UC1)
            # near = np.copy(depth_colormap)
            # near = cv2.cvtColor(near,cv2.COLOR_BGR2GRAY,  dstCn = cv2.CV_8UC1)
            
            # nearMask = createMaskFormDepth(near, 100, cv2.THRESH_BINARY)
            
            # far = np.copy(depth_colormap)
            # far = cv2.cvtColor(far,cv2.COLOR_BGR2GRAY,  dstCn = cv2.CV_8UC1)
            # # far[far == 0] = 255
            # farMask = createMaskFormDepth(far, 80, cv2.THRESH_BINARY_INV)
            # mask = np.zeros(near.shape, dtype="uint8")
            # mask[far == 0] = 255
            # mask[near == 0] = 0
            # # mask = np.zeros(near.shape, dtype="uint8")
            # # mask[True] = cv2.GC_BGD
            # # mask[farMask == 0] = cv2.GC_PR_BGD
            # # mask[nearMask == 255] = cv2.GC_FGD
            # # rect = (0,0,0,0)
            # # mask, _, _ = cv2.grabCut(color_image, mask,rect, bgModel, fgModel, 1, mode = cv2.GC_INIT_WITH_MASK)

            # # outputMask = np.zeros(color_image.shape)
            # # outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
            # outputMask = (nearMask * 255).astype("uint8")
            # output = cv2.bitwise_and(color_image, color_image, mask=outputMask)
            # cv2.imshow("out", output)
            
            # Convert images to numpy arrays
             
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
             
            # (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel, fgModel, iterCount=args["iter"], mode=cv2.GC_INIT_WITH_RECT)
             
            datum.cvInputData = color_image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            pose, hand = setReferenceFrame(datum)
             
            output = datum.cvOutputData
            try:
                # xAxis = pose.rightShoulder.x
                # height, width, _ = output.shape
                # cv2.rectangle(output, (xAxis, 0), (0, width), (0,255,0), 3)
                mask = removeBackground(depth_frame, pose, img, color_image)
                 
                imgWoutBg = cv2.bitwise_and(color_image, color_image, mask=mask)
                 
                # cv2.imshow("mask", mask)
                # kMeans(imgWoutBg)
                # moments(imgWoutBg)

                # print("Right base X: {}, Y: {}".format(hand.rightBase.x, hand.rightBase.y))
                # endX = int(hand.rightBase.x +20)
                # endY = int(hand.rightBase.y +20)
                # initX = int(hand.rightBase.x - hand.largerMeasure())
                # initY = int(hand.rightBase.y - hand.largerMeasure())
                # cv2.rectangle(output, (200, 200),(100, 500), (0, 255, 0),3)
                # cv2.rectangle(output, ((hand.rightBase.x + 20), (hand.rightBase.y -20)),((hand.rightBase.x + hand.largerMeasure()), (hand.rightBase.y + hand.largerMeasure())), (0, 255, 0),3)
                # cv2.rectangle(output, (initX, initY),(endX, endY), (0, 255, 0),3)

            except:
                pass

             
            # Stack both images horizontally
            images = np.hstack((output, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    finally:
        pipeline.stop()

