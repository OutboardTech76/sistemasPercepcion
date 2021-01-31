
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
## Move robot's base
Calculate mass center of arms and create three states:
1) Arm extended -> mass center in the center
2) Arm +90º -> mass center in upper position 
3) Arm -90º -> mass center in lower position


## captureHand.py
Use openpose as a hand detector, extract a bounding box arround hand and segmentate it using traditional methods such has k-means. 
After that evaluate two modes: closed and opened hand by calculating the area. Bigger area means open while smaller area means close.
Use this values to control robot's gripper.

Use 3D view to segmantate person in image. Uses OpenPose to calc distance between camera and person's hip, when each pixel's distance is inside a max and min values created according to ref frame we assume thats part of the body, otherwise is discarted.
All of that generates a binary mask that's passed using bitwise_and to the original 2D image to segmentate body.
Also included a siluette deteector using watershed algorithm.
