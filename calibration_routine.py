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
# import captureHand as ch
from captureHand import Pose
import numpy as np


Image = NDArray[(Any, Any), int]
DepthFrame = rs.depth_frame
ColorImage = NDArray[(Any, Any, 3), int]
Color = Tuple[int, int]

Img2Robot: Callable[[float, float,
                     float], Tuple[float, float, float]] = None


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


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
             robot_mlen=1.5)
def Calibracion(img: ColorImage, d_img: DepthFrame, pose: Pose) -> int:
    time_now: float = time.time()
    if Calibracion.curr_state == 0:
        output = _pon_rectangulo_centro(img, (0, 0, 255))
        print("State 1")
        cv2.imshow("out", output)
        if (time_now - Calibracion.time_begining) >= Calibracion.time_to_start:
            Calibracion.curr_state += 1
    elif Calibracion.curr_state == 1:
        print("State 2")
        c_dist: float = _get_distance_to_center(
            d_img, img.shape[0], img.shape[1])
        if (c_dist > Calibracion.lower_dist_th and c_dist <
                Calibracion.higher_dist_th):
            print("T pose")
            output = _pon_rectangulo_centro(img, (0, 255, 0))
            cv2.imshow("out", output)
            Calibracion.it_counter += 1
        else:
            output = _pon_rectangulo_centro(img, (0, 0, 255))
            cv2.imshow("out", output)
            Calibracion.it_counter = 0
        if Calibracion.it_counter >= Calibracion.it_to_start_cal:
            Calibracion.curr_state += 1
            Calibracion.it_counter = 0
    elif Calibracion.curr_state == 2:
        print("State 3")
        l_arm_d = pose.distLeftArm()
        r_arm_d = pose.distRightArm()
        Calibracion.arms_xs[0].append(l_arm_d)
        Calibracion.arms_xs[1].append(r_arm_d)
        Calibracion.it_counter += 1

        if Calibracion.it_counter >= Calibracion.it_to_start_arm:
            Calibracion.curr_state += 1
    elif Calibracion.curr_state == 3:
        print("State 4")
        Calibracion.arm_mean = [
            _mean(
                Calibracion.arms_xs[0]), _mean(
                Calibracion.arms_xs[1])]
        Calibracion.curr_state += 1
    elif Calibracion.curr_state == 4:
        global Img2Robot

        def Img2Robot(pos): return pos * 0.92 * \
            Calibracion.robot_mlen[0] / Calibracion.arm_mean[0]
        Calibracion.curr_state = 5
    elif Calibracion.curr_state == 5:
        print("Calibration done")
        return 1
    return 0
