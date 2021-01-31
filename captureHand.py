import cv2
import pyrealsense2 as rs
import numpy as np
import argparse
import json
import sys
import os
import random 
import ros
from typing import Tuple, Any, Dict, List, Union
from nptyping import NDArray
from openpose import pyopenpose as op
import calibration_routine as calib

random.seed(123)

# Type aliases
Keypoint = NDArray[(Any, Any, Any), Any]
PoseKeypoints = op.Datum.poseKeypoints
HandKeypoints = op.Datum.handKeypoints
Image = NDArray[(Any, Any), int]
ColorImage = NDArray[(Any, Any, 3), int]
DepthFrame = rs.depth_frame
Contour = List[Any]
Point2D = NDArray[(Any, 1), int]

class Point:
    """
    Point class.
    Creates Point as numpy array (npPoint), with x, y and z (x,y,z) and as a tuple of (x,y) points (point).
    Also creates score returned from openpose's keypoints.
    Args:
        pt -> numpy array with keypoints containing [x, y, score]. Keypoints returned from OpenPose.
    """
    def __init__(self, pt: Keypoint):
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
    """
    Class with arms and hip points captured from OpenPose.
    Args:
        keypoints -> Keypoint vector extracted from OpenPose with Points of body. 
    """
    def __init__(self, keypoints: PoseKeypoints):
        self.rightShoulder = Point(keypoints[2])
        self.rightElbow = Point(keypoints[3])
        self.rightWrist = Point(keypoints[4])
        self.leftShoulder = Point(keypoints[5])
        self.leftElbow = Point(keypoints[6])
        self.leftWrist = Point(keypoints[7])
        self.center = Point(keypoints[8])

    def distRightArm(self) -> float:
        """
        Euclidean distance between shoulder and wrist.
        """
        norm = np.linalg.norm
        dist =  abs(norm(self.rightShoulder.npPoint - self.rightWrist.npPoint))
        return dist
     
    def distLeftArm(self) -> float:
        """
        Euclidean distance between shoulder and wrist.
        """
        norm = np.linalg.norm
        dist =  abs(norm(self.leftShoulder.npPoint - self.leftWrist.npPoint))
        return dist

class Hand:
    """
    Class with finger points captured from OpenPose.
    HandKeypoint[0] -> right hand.
    HandKeypoint[1] -> left hand.
    Args:
        keypoints -> Keypoint vector extracted from OpenPose with hand points. 
    """
    def __init__(self, keypoints: HandKeypoints):
        # Second dimension limits num person
        right = keypoints[1][0]
        left = keypoints[0][0]
        self.keypointsRight = right
        self.keypointsLeft = left
         
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
         
    def centerRight(self) -> Tuple[int, int]:
        """
        Center point of right hand.
        Returns:
            (x, y) position.
        """
        # base = pose.rightWrist.npPoint
        base = self.rightBase.npPoint
        baseThumb = (base + self.rightThumb.npPoint)/2
        baseIndex = (base + self.rightIndex.npPoint)/2
        baseMiddle = (base + self.rightMiddle.npPoint)/2
        baseRing = (base + self.rightRing.npPoint)/2
        basePinky = (base + self.rightPinky.npPoint)/2
        center = (baseThumb + baseIndex + baseMiddle + baseRing + basePinky)/5
        center = np.asarray(center, dtype='int')
        center = tuple(center)
        return center
     
    def centerLeft(self) -> Tuple[int, int]:
        """
        Center point of left hand.
        Returns:
            (x, y) position.
        """
        base = self.leftBase.npPoint
        baseThumb = (base + self.leftThumb.npPoint)/2
        baseIndex = (base + self.leftIndex.npPoint)/2
        baseMiddle = (base + self.leftMiddle.npPoint)/2
        baseRing = (base + self.leftRing.npPoint)/2
        basePinky = (base + self.leftPinky.npPoint)/2
        
        center = (baseThumb + baseIndex + baseMiddle + baseRing + basePinky)/5
        center = np.asarray(center, dtype='int')
        center = tuple(center)
        return center
            
    def maxDistanceRight(self, pose: Pose) -> float:
        """
        Measures distance between wrist and every finger of the right hand.
        Args:
            pose -> Pose class to extract wrist point.
        Returns:
            Maximum distance between points.
        """
        #np.linalg.norm -> euclidean distance numpy
        norm = np.linalg.norm
        base = pose.rightWrist.npPoint
        keypoints = self.keypointsRight
         
        lowThumb = Point(keypoints[2]) 
        lowIndex = Point(keypoints[6]) 
        lowMiddle = Point(keypoints[10]) 
        lowRing = Point(keypoints[14]) 
        lowPinky = Point(keypoints[18]) 
         
        midThumb = Point(keypoints[3]) 
        midIndex = Point(keypoints[7]) 
        midMiddle = Point(keypoints[11]) 
        midRing = Point(keypoints[15]) 
        midPinky = Point(keypoints[19]) 
          
           
        dist1 = abs(norm(base - self.rightThumb.npPoint))
        dist2 = abs(norm(base - self.rightIndex.npPoint))
        dist3 = abs(norm(base - self.rightMiddle.npPoint))
        dist4 = abs(norm(base - self.rightRing.npPoint))
        dist5 = abs(norm(base - self.rightPinky.npPoint))
         
        dist6 = abs(norm(base - lowThumb.npPoint))
        dist7 = abs(norm(base - lowIndex.npPoint))
        dist8 = abs(norm(base - lowMiddle.npPoint))
        dist9 = abs(norm(base - lowRing.npPoint))
        dist10 = abs(norm(base -lowPinky.npPoint))
         
        dist11 = abs(norm(base - midThumb.npPoint))
        dist12 = abs(norm(base - midIndex.npPoint))
        dist13 = abs(norm(base - midMiddle.npPoint))
        dist14 = abs(norm(base - midRing.npPoint))
        dist15 = abs(norm(base - midPinky.npPoint))

        max1 = max(dist1, dist2, dist3, dist4, dist5)
        max2 = max(dist6, dist7, dist8, dist9, dist10)
        max3 = max(dist11, dist12, dist13, dist14, dist15)

        return max(max1, max2, max3)
     
    def maxDistanceLeft(self, pose: Pose) -> float:
        """
        Measures distance between wrist and every finger of the left hand.
        Args:
            pose -> Pose class to extract wrist point.
        Returns:
            Maximum distance between points.
        """
        #np.linalg.norm -> euclidean distance numpy
        norm = np.linalg.norm
        base = pose.leftWrist.npPoint
        keypoints = self.keypointsLeft
         
        lowThumb = Point(keypoints[2]) 
        lowIndex = Point(keypoints[6]) 
        lowMiddle = Point(keypoints[10]) 
        lowRing = Point(keypoints[14]) 
        lowPinky = Point(keypoints[18]) 
         
        midThumb = Point(keypoints[3]) 
        midIndex = Point(keypoints[7]) 
        midMiddle = Point(keypoints[11]) 
        midRing = Point(keypoints[15]) 
        midPinky = Point(keypoints[19]) 
         
        dist1 = abs(norm(base - self.leftThumb.npPoint))
        dist2 = abs(norm(base - self.leftIndex.npPoint))
        dist3 = abs(norm(base - self.leftMiddle.npPoint))
        dist4 = abs(norm(base - self.leftRing.npPoint))
        dist5 = abs(norm(base - self.leftPinky.npPoint))
         
        dist6 = abs(norm(base - lowThumb.npPoint))
        dist7 = abs(norm(base - lowIndex.npPoint))
        dist8 = abs(norm(base - lowMiddle.npPoint))
        dist9 = abs(norm(base - lowRing.npPoint))
        dist10 = abs(norm(base -lowPinky.npPoint))
         
        dist11 = abs(norm(base - midThumb.npPoint))
        dist12 = abs(norm(base - midIndex.npPoint))
        dist13 = abs(norm(base - midMiddle.npPoint))
        dist14 = abs(norm(base - midRing.npPoint))
        dist15 = abs(norm(base - midPinky.npPoint))
         
        max1 = max(dist1, dist2, dist3, dist4, dist5)
        max2 = max(dist6, dist7, dist8, dist9, dist10)
        max3 = max(dist11, dist12, dist13, dist14, dist15)
         
        return max(max1, max2, max3)

def setReferenceFrame(data: op.Datum) -> Union[Pose, Hand]:
    """
    Creates Hand and Pose classes if one person is detected.
    Args:
        data -> Datum parameter given by OpenPose.
    Returns:
        Pose, Hand classes.
    """
    if data.poseKeypoints is None:
        return None, None
    else:
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
    
def setParams() -> List[str]:
    """
    Initial OpenPose params.
    Returns:
        Params.
    """
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

def removeBackground(depth: DepthFrame, pose: Pose, img: Image) -> Image:
    """
    Function to remove brackground using depth sensor.

    Args:
        depth -> Depth frame used to calculate distance
        pose -> Pose class used to calculate center point where measure distance
        img -> Depth image in gray scale

    Returns:
        Binary image without barkground, just person's siluette

    """
    centerDist = depth.get_distance(int(pose.center.x), int(pose.center.y))
    armDist = depth.get_distance(int(pose.rightWrist.x), int(pose.rightWrist.y))
     
    maxValueDist = centerDist + 0.3
    # maxValueDist2 = centerDist + (abs(centerDist - armDist))
    # maxValueDist = max(maxValueDist1, maxValueDist2)
    minValueDist = centerDist - 0.2

    kernel = np.ones((3,3), np.uint8)
     
    h, w = img.shape[:2]
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

    # Use gaussian for noise reduction
    auxImg = cv2.GaussianBlur(auxImg, (5,5),cv2.BORDER_DEFAULT)
    # Apply threshold using Otsu algorith
    _, auxImg = cv2.threshold(auxImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # auxImg = cv2.erode(auxImg, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)))
    # auxImg = cv2.dilate(auxImg, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)))

    return auxImg
 
def extractHand(img: ColorImage, pose: Pose, hand: Hand) -> Tuple[Image, Contour]:
    """
    Function to segmentate hand from a given image.
     
    Args:
        img -> Color image where segment hand
        pose -> Pose class used to extract hands position
        hand -> Hand class used to extract hand position

    Returns:
        Binary image with the segmented hand
    """
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    kernel = np.ones((3,3), np.uint8)
    imgColor = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, dstCn = cv2.CV_8UC1)
    centerRHand = hand.centerRight()
    dist = hand.maxDistanceRight(pose)
    maskCircle = np.zeros((h,w), dtype='uint8')
    cv2.circle(maskCircle, centerRHand, int(2*dist/3), 255, -1)
     
    # Extract circle from original img 
    maskedImg = cv2.bitwise_and(r, r, mask=maskCircle)
     
    auxImg = cv2.GaussianBlur(maskedImg, (3,3),cv2.BORDER_DEFAULT)
    _, thresh = cv2.threshold(auxImg, 180, 255, cv2.THRESH_BINARY)
    openImg = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
     
    contours, _ = cv2.findContours(openImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    contourImg = np.zeros((h, w), dtype='uint8')
    
    for i in range(len(contours)):
        cv2.drawContours(contourImg, contours, i, 255, -1)
    

    return contourImg, contours

def calcMoments(contours: Contour) ->  float: 
    """
    Calculate moments of given contours.

    Args:
        contours -> Contour image where calculate moments

    Returns:
        Total area in the given contours
    """
    
    # Get the moments
    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])
    # Get the mass centers
    mc = [None]*len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    
    totalArea = 0
    for i in range(len(contours)):
        totalArea += mu[i]['m00']
        
    return totalArea
 
 
def calcMassCenter(contours: Contour) ->  Point2D: 
    """
    Calculate mass center of given contours.

    Args:
        contours -> Contour image where calculate mass center

    Returns:
        Mass center position as a numpy array
    """
    
    # Get the moments
    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv2.moments(contours[i])
    # Get the mass centers
    mc = [None]*len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    
    massCenter = np.zeros((2), dtype='uint8')
    count = 0
    # Check area of every contour and calc mass center
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 1500:
            massCenter[0] += mc[i][0]
            massCenter[1] += mc[i][1]
            count += 1
             
    # Mean between mass center and contours number within area limits
    if count > 0:
        massCenter = massCenter/count
    return massCenter
        
 
def extractArms(img: Image) -> Tuple[Image, Contour]:
    """
    Function to extract arms from a given mask. Mask must be binary image without barkground.
    Args:
        img -> Binary image without background.
    Returns:
        Binary image wihout body and head, just arms.
    """
    hImg, wImg = img.shape[:2]
    kernel = np.ones((3,3), np.uint8)
    armsImg = np.zeros((hImg, wImg), dtype='uint8')
    bodyBox = np.zeros((hImg, wImg), dtype='uint8')
     
    # Open image to reduce noise. After dilate to extract sure background 
    openImg = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations = 2)
    # Finding sure foreground area using euclidean distance with (3,3) mask and threshold with 4/10 max value
    distTransform = cv2.distanceTransform(openImg, cv2.DIST_L2, 3)
    _, sureFg = cv2.threshold(distTransform, 0.40*distTransform.max(), 255, 0)
    sureFg = np.uint8(sureFg)
     
    # Dilete sureFg (body) and extract its contours
    dilatedFg = cv2.dilate(sureFg, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)),iterations=8)
    contours, _ = cv2.findContours(dilatedFg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # Extract max countour and draw a box from body center to image size to remove everything except right arm
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(bodyBox,(x,y),(0,hImg),255, -1)
         
    bodyBox = cv2.bitwise_not(bodyBox)
    # Remove bodyBox from openImg to get only right arm. Calc its contours
    imgWoutBody = cv2.subtract(openImg, bodyBox)
    contours, _ = cv2.findContours(imgWoutBody, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     
    # Check area of every contour and draw the biggest one (arm)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 1500:
            cv2.drawContours(armsImg, contours, i, 255, -1)

    return armsImg, contours


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

    colorize = rs.colorizer()
    colorize.set_option(rs.option.color_scheme, 2)

    # Init ros node
    ros.rospy.init_node('VisualControl')

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
             
            datum.cvInputData = color_image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            pose, hand = setReferenceFrame(datum)
             
            output = datum.cvOutputData
             
            # Stack both images horizontally
            images = np.hstack((output, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            try:
                if calib.calibration.calib_done is False:
                    calib.calibration(color_image, depth_frame, pose)
                     
                if calib.calibration.calib_done is True:
                     
                    mask = removeBackground(depth_frame, pose, img)
                    imgWoutBg = cv2.bitwise_and(color_image, color_image, mask=mask)
                    handSegmented, handContour = extractHand(imgWoutBg, pose, hand)
                    armsSegmented, armContour = extractArms(mask)

                    massCenter = calcMassCenter(armContour)
                    x, y = massCenter[:2]
                    if (x > 190) and (y < 140):
                        # Arm down
                        ros.movement(1)
                        print("Forward")
                    elif (x > 190) and (y > 180):
                        # Armd up
                        ros.movement(2)
                        print("Turn right")
                    else:
                        # Arm straight
                        ros.movement(0)
                        print("Stop")
                    
                    if calcMoments(handContour) > 1000:
                        ros.gripper(True)
                        print("Open gripper")
                    else:
                        ros.gripper(False)
                        print("Close gripper")
                     
                    distCenter = depth_frame.get_distance(int(pose.center.x), int(pose.center.y))
                    distWrist = depth_frame.get_distance(int(pose.leftWrist.x), int(pose.leftWrist.y))
                    dist = distCenter - distWrist
                    if len(pose.leftWrist.npPoint) == 2:
                        pose.leftWrist.npPoint = np.append(pose.leftWrist.npPoint, dist)
                    else:
                        pose.leftWrist.npPoint[2] = dist
                    ros.position(pose.leftWrist.npPoint[0],pose.leftWrist.npPoint[1],pose.leftWrist.npPoint[2])
                 
            except:
                pass
             
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    finally:
        pipeline.stop()

