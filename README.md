Update python path in order to work realsense
export PYTHONPATH=$PYTHONPATH:/home/paco/realsense/librealsense/build/wrappers/python
 
Update python path in order to work openpose
export PYTHONPATH=$PYTHONPATH:/usr/local/python/

Source setPaths.sh to do the above

TODO
Change function setRefFrame() to modify Pose class directly for each joint.
Add Z dimension to setRefFrame.
Add z point to Point class and update it with depth values.
Add Hand class. Only middle finger is necesary to use end effector. Maybe we could use thumb to see end effector's orientation.
Use thumb position (with image, not OpenPose) to extract orientation, in case that the previous option does not work. Use thumb up, down, front or rear.
Reduce image size in order to increase speed.
Add exceptions when no person or body part is detected to avoid segmantation fault.
