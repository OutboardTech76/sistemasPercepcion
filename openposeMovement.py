import argparse
import json
import sys
import cv2
import os
import argparse
import numpy as np
# import rospy
from openpose import pyopenpose as op
# import tensorflow as tf
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# For model BODY_25 :
#   {0,  "Nose"},
#   {1,  "Neck"},
#   {2,  "RShoulder"},
#   {3,  "RElbow"},
#   {4,  "RWrist"},
#   {5,  "LShoulder"},
#   {6,  "LElbow"},
#   {7,  "LWrist"},
#   {8,  "MidHip"},
#   {9,  "RHip"},
#   {10, "RKnee"},
#   {11, "RAnkle"},
#   {12, "LHip"},
#   {13, "LKnee"},
#   {14, "LAnkle"},
#   {15, "REye"},
#   {16, "LEye"},
#   {17, "REar"},
#   {18, "LEar"},
#   {19, "LBigToe"},
#   {20, "LSmallToe"},
#   {21, "LHeel"},
#   {22, "RBigToe"},
#   {23, "RSmallToe"},
#   {24, "RHeel"},
#   {25, "Background"}

 
parser = argparse.ArgumentParser(description='Predict images using camera')
parser.add_argument("--json", dest="jsonPath", type=str, default="jsonFiles/", help="Directory where save .json files (default = jsonFiles/)")
  
# Point class with x,y,z values and point as a tuple (x,y)
class Point():
    def __init__(self,pt):
        # Convert to numpy float or there will be error when printing lines
        pt =  np.float32(pt)
        self.npPoint = pt[0:2]
        self.x = pt[0]
        self.y = pt[1]
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
        self.center = Point(keypoints[0])
    
# Convert points to reference frame placed in middle hip
def convertToRefFrame(ref, point):
    changeValue = np.float32([1, -1])
    center = ref.npPoint
    newPoint = point.npPoint - center
    newPoint = newPoint*changeValue
    newPoint = np.flip(newPoint,0)
    return newPoint
    

    
def setParams():
    args = parser.parse_args()
    params = dict()
    params["model_folder"] = "/home/paco/openpose/models"
    params["model_pose"] = "BODY_25"
    params["hand"] = True
    params["face"] = False
    # Reduce net resolution in order to avoid memory problems
    params["net_resolution"] = "-1x128"
    # params["write_json"] = args.jsonPath 
    return params

def setReferenceFrame(data):
    peopleNum = len(data.poseKeypoints)
    for i in range(peopleNum):
        # center = Point(data.poseKeypoints[i][0])
        pose = Pose(data.poseKeypoints[i])
        print("Center: "+str(pose.center.npPoint))
        print("RightShoulder: "+str(pose.rightShoulder.npPoint))
        print("Right Shoulder fixed: "+str(convertToRefFrame(pose.center, pose.rightShoulder)))

        # print("x: "+str(center.x)+" y: "+str(center.y))
        # print("Array: "+str(center.npPoint))
    try:
        print("First dimension : ")
        print(data.poseKeypoints[1][1])
        print("done")
    except:
        pass



if __name__ == '__main__':

    # rospy.init_node("senderOpenpose")
     
    # Openpose config
    params = setParams()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
     
    cap = cv2.VideoCapture(0)

    # config = ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)

    # tic = time.time()
    while(True):
        ret, frame = cap.read()
         
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        setReferenceFrame(datum)

        output = datum.cvOutputData
        cv2.imshow('output', output)
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # toc = time.time()
            # print("Time: " +str(toc-tic))
            break

    cap.release()
    cv2.destroyAllWindows()

