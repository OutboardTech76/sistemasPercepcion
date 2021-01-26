import cv2
import pyrealsense2 as rs
import numpy as np
import argparse
import json
import sys
import os
from openpose import pyopenpose as op


def createMaskFormDepth(depth, thresh, tp):
    cv2.threshold(depth, thresh, 255, tp, depth)
    depth = cv2.dilate(depth, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4), (3,3)))
    depth = cv2.erode(depth, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7),(6,6)))
    return depth
 
# Classes used to define hands and fingers
# handKeypoint[0] -> left hand
# handKeypoint[1] -> right hand
class Hand:
    def __init__(self, keypoints):
        # Second dimension limits num person
        right = keypoints[0][0]
        left = keypoints[1][0]
         
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
  


if __name__ == '__main__':

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
             
             
            # depth_colormap = depth_frame.apply_filter(colorize)
            depth_colormap = np.asanyarray(colorize.colorize(depth_frame).get_data())
            if not depth_frame or not color_frame:
                continue
             
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
             
            near = np.copy(depth_colormap)
            near = cv2.cvtColor(near,cv2.COLOR_BGR2GRAY,  dstCn = cv2.CV_8UC1)
            
            nearMask = createMaskFormDepth(near, 100, cv2.THRESH_BINARY)
            
            far = np.copy(depth_colormap)
            far = cv2.cvtColor(far,cv2.COLOR_BGR2GRAY,  dstCn = cv2.CV_8UC1)
            # far[far == 0] = 255
            farMask = createMaskFormDepth(far, 80, cv2.THRESH_BINARY_INV)
            mask = np.zeros(near.shape, dtype="uint8")
            mask[far == 0] = 255
            mask[near == 0] = 0
            # mask = np.zeros(near.shape, dtype="uint8")
            # mask[True] = cv2.GC_BGD
            # mask[farMask == 0] = cv2.GC_PR_BGD
            # mask[nearMask == 255] = cv2.GC_FGD
            # rect = (0,0,0,0)
            # mask, _, _ = cv2.grabCut(color_image, mask,rect, bgModel, fgModel, 1, mode = cv2.GC_INIT_WITH_MASK)

            # outputMask = np.zeros(color_image.shape)
            # outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
            outputMask = (nearMask * 255).astype("uint8")
            output = cv2.bitwise_and(color_image, color_image, mask=outputMask)
            # cv2.imshow("out", output)
            
            # Convert images to numpy arrays
             
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
             
            # (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel, fgModel, iterCount=args["iter"], mode=cv2.GC_INIT_WITH_RECT)


             
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
             
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    finally:
        pipeline.stop()

