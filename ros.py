"""
Module with ros publishers
"""

 
import rospy
from std_msgs.msg import Bool, Float32, Int16


def gripper(mode: bool):
    """
    Open or close gripper
    Args:
        True -> Open gripper
        False -> Close gripper
    """
    pub = rospy.Publisher('gripperMode', Bool, queue_size=10)
    pub.publish(mode)


def position(x: float, y: float, z: float):
    """
    Cartesian positions to move robot's arm
    Args:
        x, y, z 
    """
    pubX = rospy.Publisher('cartPosition/x', Float32, queue_size=10 )
    pubY = rospy.Publisher('cartPosition/y', Float32, queue_size=10 )
    pubZ = rospy.Publisher('cartPosition/z', Float32, queue_size=10 )
    
    pubX.publish(x)
    pubY.publish(y)
    pubZ.publish(z)

def movement(moveType: int):
    """
    Move robot base.
    Args:
        0 -> Stop
        1 -> Forward
        2 -> Turn right
    """
    pub = rospy.Publisher('movementType', Int16, queue_size=10)
    pub.publish(moveType)
