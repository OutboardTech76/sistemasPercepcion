import argparse
import json
import sys
import cv2
import os
import argparse
# import rospy
from openpose import pyopenpose as op
# import tensorflow as tf
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

 
parser = argparse.ArgumentParser(description='Predict images using camera')
parser.add_argument("--json", dest="jsonPath", type=str, default="jsonFiles/", help="Directory where save .json files (default = jsonFiles/)")
  
def setParams():
    args = parser.parse_args()
    params = dict()
    params["model_folder"] = "/home/paco/openpose/models"
    params["model_pose"] = "BODY_25"
    params["hand"] = True
    params["face"] = False
    # Reduce net resolution in order to avoid memory problems
    params["net_resolution"] = "-1x128"
    params["write_json"] = args.jsonPath 
    return params


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

        output = datum.cvOutputData
        cv2.imshow('output', output)
         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # toc = time.time()
            # print("Time: " +str(toc-tic))
            break

    cap.release()
    cv2.destroyAllWindows()

