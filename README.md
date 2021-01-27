
# Working paths
Source setPaths.sh to use openpose and pyrealsense. Both are installed from source.

# TODO
Change function setRefFrame() to modify Pose class directly for each joint.
Add Z dimension to setRefFrame. -> DONE
Add z point to Point class and update it with depth values. -> DONE
Add Hand class. Only middle finger is necesary to use end effector. Maybe we could use thumb to see end effector's orientation. 
Use thumb position (with image, not OpenPose) to extract orientation, in case that the previous option does not work. Use thumb up, down, front or rear.
Reduce image size in order to increase speed.
Add exceptions when no person or body part is detected to avoid segmantation fault.


## captureHand.py
Use openpose as a hand detector, extract a bounding box arround hand and segmentate it using traditional methods such has k-means. 
After that evaluate two modes: closed and opened hand by calculating the area. Bigger area means open while smaller area means close.
Use this values to control robot's gripper.

