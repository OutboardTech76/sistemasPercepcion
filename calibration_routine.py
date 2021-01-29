"""
Modulo para la rutina de calibracion de la
Realsense D435 para la asignatura de Percepcion
"""

import cv2
import time
import pyrealsense2 as rs
from nptyping import NDArray
from typing import Tuple, Union
import captureHand as ch


Image = NDArray[(int, int), int]
DepthFrame = rs.depth_frame
ColorImage = NDArray[(int, int, 3), int]
Color = Tuple(int, int)


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
    img_w = img.shape[0]
    img_h = img.shape[1]
    r_side = 40
    output = img.copy()
    rect = img.copy()
    cv2.rectangle(rect, ((img_w - r_side) / 2, (img_h - r_side) / 2),
                  ((img_w + r_side) / 2, (img_h + r_side) / 2), color, -1)
    cv2.addWeighted(rect, 0.5, img, 0.5, 0, output)
    return output


def _get_distance_to_center(
        d_img: DepthFrame,
        width: int,
        heigth: int) -> float:
    return d_img.get_distance(d_img, int(width / 2), int(heigth / 2))


@static_vars(time_to_start=10.0,
             lower_dist_th=1.3,
             higher_dist_th=1.75,
             it_to_start=15,
             time_begining=time.time(),
             curr_state=0,
             it_counter=0,
             arms_xs=[[], []])
def Calibracion(img: ColorImage, d_img: DepthFrame) -> int:
    time_now: float = time.time()
    if Calibracion.curr_state == 0:
        if time_now - Calibracion.time_begining < Calibracion.time_to_start:
            _pon_rectangulo_centro(img, (0, 0, 255))
            Calibracion.curr_state += 1
    elif Calibracion.curr_state == 1:
        c_dist: float = _get_distance_to_center(
            d_img, img.shape[0], img.shape[1])
        if (c_dist > Calibracion.lower_dist_th and c_dist <
                Calibracion.higher_dist_th):
            _pon_rectangulo_centro(img, (0, 255, 0))
            Calibracion.it_counter += 1
        else:
            _pon_rectangulo_centro(img, (0, 0, 255))
            it_counter = 0
        if it_counter >= Calibracion.it_to_start:
            Calibracion.curr_state += 1
            Calibracion.it_counter = 0
    elif Calibracion.curr_state == 2:
        l_arm_d = ch.distLeftArm()
        r_arm_d = ch.distRightArm()
        Calibracion.arms_xs[0].append(l_arm_d)
        Calibracion.arms_xs[1].append(r_arm_d)
        Calibracion.it_counter += 1

    return 0