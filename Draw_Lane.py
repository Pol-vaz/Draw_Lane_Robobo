import sys 
sys.path.append('/home/pol/Escritorio/TFG_2019-2020/robobo.py-master')
from Robobo import Robobo
sys.path.append('/home/pol/Escritorio/TFG_2019-2020/robobo-python-video-stream-master/robobo_video')
from robobo_video import RoboboVideo
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_line_pro(coeffs1, coeffs2, M, frame):
    
    width, height, channels = frame.shape
    x=np.arange(0,height)
    a_1= coeffs1['a']
    b_1= coeffs1['b']
    c_1=  coeffs1['c']
    a_2= coeffs2['a']
    b_2= coeffs2['b']
    c_2= coeffs2['c']

    y1 = a_1* x**2 + b_1* x + c_1
    y2 = a_2*x**2 + b_2* x + c_2
    M= np.array([[0.6006944444444444, -0.16193181818181818, 57.0], [0.0, 0.4018308080808081, 70.0], [0.0, -0.0011343907828282828, 1.0]])
    x1_t = (M[0][0]*x + M[0][1]*y1 + M[0][2])/(M[2][0]*x + M[2][1]*y1 + M[2][2])
    y1_t = (M[1][0]*x + M[1][1]*y1 + M[1][2])/(M[2][0]*x + M[2][1]*y1 + M[2][2])

    x2_t = (M[0][0]*x + M[0][1]*y2 + M[0][2])/(M[2][0]*x + M[2][1]*y2 + M[2][2])
    y2_t = (M[1][0]*x + M[1][1]*y2 + M[1][2])/(M[2][0]*x + M[2][1]*y2 + M[2][2])

    implot = plt.imshow(frame)

    plt.scatter(x=x1_t, y= y1_t, c='r', s=1)
    plt.scatter(x=x2_t, y= y2_t, c='b', s=1)
    plt.scatter(x=x, y= y1, c='g', s=1)
    plt.scatter(x=x, y= y2, c='y', s=1)
    plt.show()


def draw_line_Basic(coeffs1, coeffs2,frame):
    
    width, height, channels  = frame.shape
    x = np.arange(0,height,0.1)
    y1_o = coeffs1['a']* 0.1 + coeffs1['b']
    y2_o = coeffs2['a']* 0.1 + coeffs2['b']
    y1 = coeffs1['a']* height + coeffs1['b']
    y2 = coeffs2['a']* height + coeffs2['b']
    img = cv2.line(frame,(0,int(y1_o)),(height,int(y1)),(0,255,0))
    img = cv2.line(frame,(0,int(y2_o)),(height,int(y2)),(0,255,0))
    
    return img
    
def main(basic=True):
    IP= '192.168.0.17'
    rob = Robobo(IP)
    rob.connect()
    rob.moveTiltTo(100,20)
    video = RoboboVideo(IP)
    video.connect()
    rob.toggleLaneColorInversion()  
    #rob.moveWheels(10,10)

    while True:

        if basic:
            
            obj=rob.readLaneBasic()
            coeffs1 = obj.coeffs1
            coeffs2 = obj.coeffs2
            frame = video.getImage()
            frame = draw_line_Basic(coeffs1, coeffs2,frame)
            
        if basic == False:
            obj=rob.readLanePro()
            coeffs1 = obj.coeffs1
            coeffs2 = obj.coeffs2
            M = obj.M
            frame = video.getImage()
            frame = draw_line_pro(coeffs1, coeffs2,M,frame)

        cv2.imshow('Smarphone Camera', frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    #rob.stopMotors()
    video.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(basic=True)

