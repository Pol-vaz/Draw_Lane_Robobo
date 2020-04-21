import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ROBOBO_PY_PATH = '/home/pol/Escritorio/TFG_2019-2020/robobo.py-master'
# ROBOBO_VIDEO_PATH = '/home/pol/Escritorio/TFG_2019-2020/robobo-python-video-stream-master/robobo_video'
ROBOBO_PY_PATH = '/home/eadan97/Ferrol/robobo.py'
ROBOBO_VIDEO_PATH = '/home/eadan97/Ferrol/robobo-python-video-stream/robobo_video'
IP = '192.168.0.11'

sys.path.append(ROBOBO_PY_PATH)

from Robobo import Robobo

sys.path.append(ROBOBO_VIDEO_PATH)
from robobo_video import RoboboVideo


# TODO: Change parameter name of 'M'
def draw_line_pro(coeffs1, coeffs2, M, frame):
    """
    Draws the lines with coeffs1 and coeffs2 on a mask that is warped using the transformation matrix M, then its added to the frame
    :param coeffs1: coefficients of line 1
    :param coeffs2: coefficients of line 2
    :param M: transformation matrix
    :param frame: frame
    :return: frame with the lines drawn
    """
    # Obtain the height and width of the frame
    height, width, _ = frame.shape
    # Mask to draw the lines (avoid destroying the frame!, it is a bad practice)
    mask = np.zeros_like(frame)

    # Create an array with numbers from 0 to height-1 (equivalent of np.array(list(range(height)))) but with less calls)
    plot_y = np.linspace(0, height - 1, height)

    # Arrays with evaluations of the coeffs
    left_x = coeffs1['a'] * plot_y ** 2 \
             + coeffs1['b'] * plot_y \
             + coeffs1['c']
    right_x = coeffs2['a'] * plot_y ** 2 + \
              coeffs2['b'] * plot_y + \
              coeffs2['c']

    # Draw the lines (one red, one blue)
    cv2.polylines(mask, [np.int32(np.stack((plot_y, left_x), axis=1))], False, (255, 0, 0), 20)
    cv2.polylines(mask, [np.int32(np.stack((plot_y, right_x), axis=1))], False, (0, 0, 255), 20)

    # Warp the perspective
    mask = cv2.warpPerspective(mask, np.float32(M), (width, height))  # Warp back to original image space

    # Add the lines to the original frame
    img = cv2.addWeighted(frame, 1., mask, 0.3, 0)

    return img
    # x = np.arange(0, height)
    # a_1 = coeffs1['a']
    # b_1 = coeffs1['b']
    # c_1 = coeffs1['c']
    # a_2 = coeffs2['a']
    # b_2 = coeffs2['b']
    # c_2 = coeffs2['c']
    #
    # y1 = a_1 * x ** 2 + b_1 * x + c_1
    # y2 = a_2 * x ** 2 + b_2 * x + c_2
    # M = np.array([[0.6006944444444444, -0.16193181818181818, 57.0], [0.0, 0.4018308080808081, 70.0],
    #               [0.0, -0.0011343907828282828, 1.0]])
    # x1_t = (M[0][0] * x + M[0][1] * y1 + M[0][2]) / (M[2][0] * x + M[2][1] * y1 + M[2][2])
    # y1_t = (M[1][0] * x + M[1][1] * y1 + M[1][2]) / (M[2][0] * x + M[2][1] * y1 + M[2][2])
    #
    # x2_t = (M[0][0] * x + M[0][1] * y2 + M[0][2]) / (M[2][0] * x + M[2][1] * y2 + M[2][2])
    # y2_t = (M[1][0] * x + M[1][1] * y2 + M[1][2]) / (M[2][0] * x + M[2][1] * y2 + M[2][2])
    #
    # # implot = plt.imshow(frame)
    #
    # plt.scatter(x=x1_t, y=y1_t, c='r', s=1)
    # plt.scatter(x=x2_t, y=y2_t, c='b', s=1)
    # plt.scatter(x=x, y=y1, c='g', s=1)
    # plt.scatter(x=x, y=y2, c='y', s=1)
    # plt.show()
    #


def draw_line_Basic(coeffs1, coeffs2, frame):
    height, width, channels = frame.shape
    height -= 1  # Remember we need to 0-index
    mask = np.zeros_like(frame)
    # frame = cv2.flip(frame, 1)  # TODO: Remove this when patch rolls out

    # x = np.arange(0, height, 0.1)
    # Remember, y is the height of the frame, so we need the x (just solve the function)

    # Todo: use this when patch rolls out
    # x1_o = -coeffs1['b'] / coeffs1['a']
    # x2_o = -coeffs2['b'] / coeffs2['a']
    # x1 = (height - coeffs1['b']) / coeffs1['a']
    # x2 = (height - coeffs2['b']) / coeffs2['a']
    x1_o = -coeffs2['a'] / coeffs1['a']
    x2_o = -coeffs2['b'] / coeffs1['b']
    x1 = (height - coeffs2['a']) / coeffs1['a']
    x2 = (height - coeffs2['b']) / coeffs1['b']
    mask = cv2.line(mask, (int(x1_o), 0), (int(x1), height), (255, 0, 0), 15)
    mask = cv2.line(mask, (int(x2_o), 0), (int(x2), height), (0, 0, 255), 15)

    # Add the lines to the original frame
    img = cv2.addWeighted(frame, 1., mask, 0.5, 0)

    return img


def main(basic=True):
    rob = Robobo(IP)
    rob.connect()
    # rob.moveTiltTo(100, 20)
    video = RoboboVideo(IP)
    video.connect()
    # rob.toggleLaneColorInversion()
    # rob.moveWheels(10,10)

    while True:
        frame = video.getImage()
        # cv2.imshow('Smarphone Camera', frame)
        cv2.namedWindow('Smartphone Camera', cv2.WINDOW_NORMAL)
        if frame is None:
            continue

        if basic:
            obj = rob.readLaneBasic()
            coeffs1 = obj.coeffs1
            coeffs2 = obj.coeffs2
            frame = draw_line_Basic(coeffs1, coeffs2, frame)

        if not basic:
            obj = rob.readLanePro()
            coeffs1 = obj.coeffs1
            coeffs2 = obj.coeffs2
            M = obj.minv
            frame = draw_line_pro(coeffs1, coeffs2, M, frame)

        cv2.imshow('Smarphone Camera', frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    # rob.stopMotors()
    video.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(False)
