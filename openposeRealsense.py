import pyrealsense2 as rs
import argparse
import json
import sys
import cv2
import os
import argparse
import numpy as np
# import rospy
from openpose import pyopenpose as op
 
  
# Classes used to define hands and fingers
# handKeypoint[0] -> left hand
# handKeypoint[1] -> right hand
class Hand:
    def __init__(self, keypoints):
        # Second dimension limits num person
        right = keypoints[0][0]
        left = keypoints[1][0]
         
        self.rightBase = Point(right[0])
        self.leftBase = Point(left[0])
        self.rightMiddle = Point(right[12])
        self.leftMiddle = Point(left[12])


     
 
  
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

# Calc arm size between shoulder-elbow and elbow-wrist
    def rightArmSize(self):
        point1 = self.rightShoulder.npPoint
        point2 = self.rightElbow.npPoint
        size = np.lianlg.norm(point1 - point2)
        return size
         
    def rightForearmSize(self):
        point1 = self.rightElbow.npPoint
        point2 = self.rightWrist.npPoint
        size = np.lianlg.norm(point1 - point2)
        return size
     
    def leftArmSize(self):
        point1 = self.leftShoulder.npPoint
        point2 = self.leftElbow.npPoint
        size = np.lianlg.norm(point1 - point2)
        return size
         
    def leftForearmSize(self):
        point1 = self.leftElbow.npPoint
        point2 = self.leftWrist.npPoint
        size = np.lianlg.norm(point1 - point2)
        return size

# Distance between reference frame and both shoulders.
    def distRefFrameArms(self):
        point1 = self.center.npPoint
        point2 = self.rightShoulder.npPoint
        point3 = self.leftShoulder.npPoint

        dist1 = np.lianlg.norm(point1-point2)
        dist2 = np.lianlg.norm(point1-point3)
        dist = (dist1+dist2)/2
        return dist
    
# Convert points to reference frame placed in middle hip
# P = point, O = ref frame, -> NewPoint in O = [(P-O)*(1,-1)].flip
# Now axis are in the same direction that Gazebo's axis
def convertToRefFrame(hand, pose, depth_frame):
    changeValue = np.float32([1, -1])
    center = pose.center.npPoint
    centerDist = depth_frame.get_distance(int(pose.center.x), int(pose.center.y))
    pose.center.npPoint = np.append(center, centerDist)

    # Convert pose position to ref frame
    point = pose.rightShoulder
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    pose.rightShoulder.npPoint = np.append(newPoint,dist)
     
    point = pose.rightElbow
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    pose.rightElbow.npPoint = np.append(newPoint, dist)
     
    point = pose.rightWrist
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    pose.rightWrist.npPoint = np.append(newPoint, dist)
     
      
    point = pose.leftShoulder
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    pose.leftShoulder.npPoint = np.append(newPoint, dist)
     
    point = pose.leftElbow
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    pose.leftElbow.npPoint = np.append(newPoint, dist)
     
    point = pose.leftWrist
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    pose.leftWrist.npPoint = np.append(newPoint, dist)

    
    # Convert hand keypoints to ref frame
     
    point = hand.rightBase
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    hand.rightBase.npPoint = np.append(newPoint, dist)
     
    point = hand.rightMiddle
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    hand.rightMiddle.npPoint = np.append(newPoint, dist)
     
    point = hand.leftBase
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    hand.leftBase.npPoint = np.append(newPoint, dist)
     
    point = hand.leftMiddle
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    dist = depth_frame.get_distance(int(point.x), int(point.y))
    dist -= centerDist
    dist *= -1
    hand.leftMiddle.npPoint = np.append(newPoint, dist)
     

    # return pose and hand classes
    return pose, hand
    
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

def setReferenceFrame(data, depth_frame):
    try:
        peopleNum = len(data.poseKeypoints)

        # Detectmore than one person
        # for i in range(peopleNum):
            # pose = Pose(data.poseKeypoints[i])

        # Just one person at a time
        if peopleNum > 0:
            pose = Pose(data.poseKeypoints[0])
            hand = Hand(data.handKeypoints)
             
            poseChanged, handChanged = convertToRefFrame(hand, pose, depth_frame)
            
            return poseChanged, handChanged
    except:
        pass

def convertToRobotValues(pose):
    height = pose.distRefFrameArms()
    rArm = pose.rightArmSize()
    rForearm = pose.rightForearmSize()
    lArm = pose.leftArmSize()
    lForearm = pose.leftForearmSize()

    robotHeight = 3.4
    robotRArm = 1.2
    robotRForearm = 1.2
    robotLArm = 1.2
    robotLForearm = 1.2

    rightShoulder = convertToRefFrame(pose.center, pose.rightShoulder)
    rightElbow = convertToRefFrame(pose.center, pose.rightElbow)
    rightWrist = convertToRefFrame(pose.center, pose.rightWrist)
    leftShoulder = convertToRefFrame(pose.center, pose.leftShoulder)
    leftElbow = convertToRefFrame(pose.center, pose.leftElbow)
    leftWrist = convertToRefFrame(pose.center, pose.leftWrist)

if __name__ == '__main__':

    # rospy.init_node("senderOpenpose")
     
    # Openpose config
    params = setParams()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
     
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)

    pipeline.start(config)
     
    print("Camara arrancada")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
             
            if not depth_frame or not color_frame:
                continue
             
             
            datum = op.Datum()
             
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
             
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


             
            datum.cvInputData = color_image
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            pose, hand = setReferenceFrame(datum, depth_frame)
            if pose is not None:
                print("Cam dist: {}".format(pose.center.npPoint))
                print("R wrist: {}".format(pose.rightWrist.npPoint))
                # print("Hand right base: {}".format(hand.rightBase.npPoint))
                # print("Hand right middle: {}".format(hand.rightMiddle.npPoint))
                # print("Dist between finger: {}".format(hand.rightBase.npPoint - hand.rightMiddle.npPoint))



            output = datum.cvOutputData
             
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
