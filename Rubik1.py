import cv2
import numpy as np


def DibujarCara(cara, orientacion):

    frame_sim = np.ones((400,400,3),dtype=np.uint8)
    m = 0
    n = 0
    color_BGR = (0,0,0)
    while m < 3:
        n = 0
        while n < 3:
            if (cara[n][m] == 0):
                color_BGR = (0,0,250)
            elif (cara[n][m] == 1):
                color_BGR = (0,80,250)
            elif (cara[n][m] == 2):
                color_BGR = (0,240,190)
            elif (cara[n][m] == 3):
                color_BGR = (10,188,0)
            elif (cara[n][m] == 4):
                color_BGR = (155,0,20)
            elif (cara[n][m] == 5):
                color_BGR = (200,200,200)
            else:
                color_BGR = (50,50,50)

            cv2.rectangle(frame_sim, ((25+(m*125)), (25+(n*125))), ((120+(m*125)), (120+(n*125))), color_BGR, -1)
            n = n + 1
        m = m + 1


    return (frame_sim)

def MascaraColor(frame, max_tot, show):

    #Parametros
    th_color = 0.20
    max_bl = 0
    act_bl = 0
    color = 10

    # Arreglos de colores
    #0
    rojo_l1 = np.array([160, 20, 10], np.uint8)
    rojo_h1 = np.array([179, 255, 205], np.uint8)

    rojo_l2 = np.array([179, 20, 20], np.uint8)
    rojo_h2 = np.array([160, 255, 205], np.uint8)

    rojo_l3 = np.array([0, 40, 20], np.uint8)
    rojo_h3 = np.array([11, 255, 255], np.uint8)
    #1
    nar_l1 = np.array([170, 20, 100], np.uint8)
    nar_h1 = np.array([179, 80, 255], np.uint8)

    nar_l2 = np.array([3, 30, 20], np.uint8)
    nar_h2 = np.array([15, 255, 255], np.uint8)
    #2
    ama_l = np.array([20, 30, 50], np.uint8)
    ama_h = np.array([45, 255, 255], np.uint8)
    #3
    ver_l = np.array([49, 100, 10], np.uint8)
    ver_h = np.array([85, 255, 205], np.uint8)
    #4
    azul_l = np.array([91, 100, 20], np.uint8)
    azul_h = np.array([148, 255, 205], np.uint8)
    #5
    bl_l = np.array([0, 0, 80], np.uint8)
    bl_h = np.array([179, 110, 255], np.uint8)

    dst_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    rojo_mask1 = cv2.inRange(dst_HSV, rojo_l1, rojo_h1)
    rojo_mask2 = cv2.inRange(dst_HSV, rojo_l2, rojo_h2)
    rojo_mask3 = cv2.inRange(dst_HSV, rojo_l3, rojo_h3)
    rojo_mask = cv2.add(rojo_mask1, rojo_mask2)
    rojo_mask = cv2.add(rojo_mask, rojo_mask3)

    act_bl=np.sum(rojo_mask==255)
    if (act_bl>(th_color*max_tot))and(act_bl>max_bl):
        color=0


    nar_mask1 = cv2.inRange(dst_HSV, nar_l1, nar_h1)
    nar_mask2 = cv2.inRange(dst_HSV, nar_l2, nar_h2)
    nar_mask = cv2.add(nar_mask1, nar_mask2)
    act_bl = np.sum(nar_mask == 255)
    if (act_bl > (th_color * max_tot)) and (act_bl > max_bl):
        color = 1

    ama_mask = cv2.inRange(dst_HSV, ama_l, ama_h)
    act_bl = np.sum(ama_mask == 255)
    if (act_bl > (th_color * max_tot)) and (act_bl > max_bl):
        color = 2

    ver_mask = cv2.inRange(dst_HSV, ver_l, ver_h)
    act_bl = np.sum(ver_mask == 255)
    if (act_bl > (th_color * max_tot)) and (act_bl > max_bl):
        color = 3

    azul_mask = cv2.inRange(dst_HSV, azul_l, azul_h)
    act_bl = np.sum(azul_mask == 255)
    if (act_bl > (th_color * max_tot)) and (act_bl > max_bl):
        color = 4

    bl_mask = cv2.inRange(dst_HSV, bl_l, bl_h)
    act_bl = np.sum(bl_mask == 255)
    if (act_bl > (th_color * max_tot)) and (act_bl > max_bl):
        color = 5

    if show == True:

        cv2.imshow('Cara rojo', rojo_mask)
        cv2.imshow('Cara naranja', nar_mask)
        cv2.imshow('Cara amarillo', ama_mask)
        cv2.imshow('Cara verde', ver_mask)
        cv2.imshow('Cara azul', azul_mask)
        cv2.imshow('Cara blanco', bl_mask)

    return (color)

def Rectangulo(frame,bool):

    captura=False
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray,20,100)

    canny = cv2.dilate(canny, None, iterations=4)
    canny = cv2.erode(canny, None, iterations=1)

    cnts,_ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]

    x_low = 999
    y_low = 999
    x_high = 0
    y_high = 0

    puntos = []
    for c in cnts:

        epsilon = 0.04 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        # Dibuja un rectángulo
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            aspect_ratio=float(w)/h
            if(h < 110 and h > 20):
                if(aspect_ratio<1.2 and aspect_ratio>0.7):
                    #cv2.drawContours(frame, [approx], 0, (0, 255, 255), 2)
                    puntos.append([x,y,w,h])
                    if x < x_low:
                        x_low=x
                    if y < y_low:
                        y_low = y
                    if x+w > x_high:
                        x_high=x+w
                    if y+h > y_high:
                        y_high = y+h

    aspect_ratio = float(x_high-x_low) / (y_high-y_low)

    if (aspect_ratio<1.2 and aspect_ratio>0.7):
        cv2.rectangle(frame,(x_low,y_low),(x_high,y_high),(0,255,0),2)
        dst_pt=np.float32([[x_low,y_low],[x_high,y_low],[x_low,y_high],[x_high,y_high]])
        dst_end=np.float32([[0,0],[400,0],[0,400],[400,400]])
        captura=True

    color_m = np.array([[11, 11, 11], [11, 11, 11], [11, 11, 11]])

    if (cv2.waitKey(1) == ord('a') and captura==True) or bool:

        M=cv2.getPerspectiveTransform(dst_pt,dst_end)
        dst = cv2.warpPerspective(frame, M, (400, 400))

        n=10
        a1=0
        while n < 400:
            m = 10
            a2=0
            while m < 400:
                dst_side_pt = np.float32([[m, n], [m+80, n], [m, n+80], [m+80, n+80]])
                dst_side_end = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
                M = cv2.getPerspectiveTransform(dst_side_pt, dst_side_end)
                dst_side = cv2.warpPerspective(dst, M, (200, 200))
                color=(MascaraColor(dst_side, 40000, False))
                color_m [a1][a2]=color
                cv2.rectangle(dst, (m, n), (m+80, n+80), (10, 10, 10), 1)
                a2 = a2 + 1
                m = m + 145
            a1 = a1 + 1
            n = n + 145

        print("\n*****************\nCara = \n{0}\n*****************".format(color_m))
        cv2.imshow('Cara Encontrada', DibujarCara(color_m,0))
        cv2.imshow('Cara', dst)

        cv2.waitKey(0)
        cv2.destroyWindow('Cara')
        cv2.destroyWindow('Cara Encontrada')

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if ret == True:

        Rectangulo(frame,False)

        cv2.imshow("Original", frame)

        if(cv2.waitKey(1) == ord('e')):
            break

    else:
        break

capture.release()
cv2.destroyAllWindows()