"""
Modulo para la rutina de calibracion de la
Realsense D435 para la asignatura de Percepcion
"""

import cv2
import time
import pyrealsense2 as rs
import operator
import functools
from nptyping import NDArray
from typing import Tuple, Union, List, Callable, Any
from captureHand import Pose
import numpy as np


Image = NDArray[(Any, Any), int]
DepthFrame = rs.depth_frame
ColorImage = NDArray[(Any, Any, 3), int]
Color = Tuple[int, int]
Position = NDArray[(Any, Any, Any), int]

Img2Robot: Callable[[float, float,
                     float], Tuple[float, float, float]] = None


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
 


 
def img2robot(pos: Position) -> Position:
    """
    Function that transforms X, Y position from image to robot coordinates and reorders (x,y,z) from image coordinates to robot coordinates.
    Returns:
        Transformed position as a np.array (x,y,z) in robots coordinates
    """
    if calibDone is True:
        posXY = pos[:2]
        auxPos = posXY* 0.92 * robotLength / armMean
        auxPos = np.append(auxPos, pos[2])
        newPos = np.zeros(auxPos.shape, dtype='uint8')
        # (x, y, z) in user coordinates are (y, z, x) in robot coordinates
        # Change them and return coordinates in correct order for the robot
        newPos[0] = auxPos[1]
        newPos[1] = auxPos[2]
        newPos[2] = auxPos[0]

        
        return newPos



def _pon_rectangulo_centro(img: Union[Image, ColorImage],
                           color: Color) -> ColorImage:
    """
    Funcion que pone un rectangulo de
    color `color` en el centro de la
    imagen `img`

    Args:
        img: Una imagen en escala de grises o en color
                sobre la que se pondra un rectangulo
        color: Tupla de 3 enteros con los colores en
                BGR

    Returns:
        Imagen con el rectangulo en el centro
    """
    h, w = img.shape[:2]
    r_side = 40
    output = img.copy()
    rect = img.copy()
    rect = np.uint8(rect)
    xInit = int((w - r_side)/2)
    yInit = int((h - r_side)/2)
    xEnd = int((w + r_side)/2)
    yEnd = int((h + r_side)/2)
    cv2.rectangle(rect, (xInit, yInit), (xEnd, yEnd), color, 5)
    cv2.addWeighted(rect, 0.5, img, 0.5, 0, output)
    return output


def _get_distance_to_center(
        d_img: DepthFrame,
        width: int,
        heigth: int) -> float:
    return d_img.get_distance(int(width / 2), int(heigth / 2))


def _mean(xs: List[float]) -> float:
    return functools.reduce(operator.add, xs, 0.0) / len(xs)


@static_vars(time_to_start=10.0,
             lower_dist_th=1.0,
             higher_dist_th=1.75,
             it_to_start_cal=5,
             it_to_start_arm=5,
             time_begining=time.time(),
             curr_state=0,
             it_counter=0,
             arms_xs=[[], []],
             robot_mlen=1.5,
             calib_done=False)
def calibration(img: ColorImage, d_img: DepthFrame, pose: Pose) -> int:
    time_now: float = time.time()
    if calibration.curr_state == 0:
        output = _pon_rectangulo_centro(img, (0, 0, 255))
        print("State 1")
        cv2.imshow("out", output)
        if (time_now - calibration.time_begining) >= calibration.time_to_start:
            calibration.curr_state += 1
    elif calibration.curr_state == 1:
        print("State 2")
        c_dist: float = _get_distance_to_center(
            d_img, img.shape[0], img.shape[1])
        if (c_dist > calibration.lower_dist_th and c_dist <
                calibration.higher_dist_th):
            print("T pose")
            output = _pon_rectangulo_centro(img, (0, 255, 0))
            cv2.imshow("out", output)
            calibration.it_counter += 1
        else:
            output = _pon_rectangulo_centro(img, (0, 0, 255))
            cv2.imshow("out", output)
            calibration.it_counter = 0
        if calibration.it_counter >= calibration.it_to_start_cal:
            calibration.curr_state += 1
            calibration.it_counter = 0
    elif calibration.curr_state == 2:
        print("State 3")
        output = _pon_rectangulo_centro(img, (0, 255, 0))
        cv2.imshow("out", output)
        l_arm_d = pose.distLeftArm()
        r_arm_d = pose.distRightArm()
        calibration.arms_xs[0].append(l_arm_d)
        calibration.arms_xs[1].append(r_arm_d)
        calibration.it_counter += 1

        if calibration.it_counter >= calibration.it_to_start_arm:
            calibration.curr_state += 1
    elif calibration.curr_state == 3:
        print("State 4")
        output = _pon_rectangulo_centro(img, (0, 255, 0))
        cv2.imshow("out", output)
        calibration.arm_mean = [
            _mean(
                calibration.arms_xs[0]), _mean(
                calibration.arms_xs[1])]
        calibration.curr_state += 1
    elif calibration.curr_state == 4:
        global calibDone
        global armMean
        global robotLength
        calibration.calib_done = True

        armMean = calibration.arm_mean[0]
        robotLength = calibration.robot_mlen
        calibDone =  calibration.calib_done

        print("Calibration done")
        cv2.destroyWindow("out")
