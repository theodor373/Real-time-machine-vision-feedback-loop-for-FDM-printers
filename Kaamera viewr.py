from decimal import ROUND_HALF_DOWN
from PIL import Image
from scipy.stats import trim_mean

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import cv2
import numpy as np
import math
import requests
import Keys.config as config
import requests

ddepth = cv2.CV_8U

printerURL = "http://192.168.1.49/printer/gcode/script?script="

# Vasak kaamera URL
URLV = "http://192.168.1.39"
# Parem kaamera URL
URLP = "http://192.168.1.37"
AWB = True

capV = cv2.VideoCapture(URLV + ":81/stream")
capP = cv2.VideoCapture(URLP + ":81/stream")

class HoughBundler:
    # code made not by me Stolen from: https://stackoverflow.com/a/70318827
    def __init__(self, min_distance=5, min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle

    def get_orientation(self, line):
        orientation = math.atan2(
            abs((line[3] - line[1])), abs((line[2] - line[0])))
        return math.degrees(orientation)

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge:
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(
                math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])

        if (len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            # sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            # sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0], points[-1]]])

    def process_lines(self, lines):
        lines_horizontal = []
        lines_vertical = []

        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical = sorted(lines_vertical, key=lambda line: line[1])
        lines_horizontal = sorted(lines_horizontal, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)

        return np.asarray(merged_lines_all)

def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frameV, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', frameV)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = frameV[y, x, 0]
        g = frameV[y, x, 1]
        r = frameV[y, x, 2]
        cv2.putText(frameV, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', frameV)

def camera_setup(url: str):

    requests.get(url + "/control?var=framesize&val=10")
    requests.get(url + "/control?var=quality&val=4")
    requests.get(url + "/control?var=brightness&val=2")
    requests.get(url + "/control?var=contrast&val=2")
    requests.get(url + "/control?var=saturation&val=-4")
    requests.get(url + "/control?var=sharpness&val=1")
    requests.get(url + "/control?var=special_effect&val=1")
    requests.get(url + "/control?var=agc_gain&val=2")

def jooneleidmine(frame):
    kernel = np.array([[0, -1, 0],
                       [-1, 8, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    grayXflip = cv2.flip(gray,1)
    grayYflip = cv2.flip(gray,0)

    gradient_sobelx = cv2.Sobel(gray,ddepth,1,0)
    gradient_sobelxneg = cv2.Sobel(grayXflip, ddepth,1,0)
    gradient_sobelxnegflip = cv2.flip(gradient_sobelxneg,1)

    gradient_sobely = cv2.Sobel(gray, ddepth,0,1)
    gradient_sobelyneg = cv2.Sobel(grayYflip, ddepth,0,1)
    gradient_sobelynegflip = cv2.flip(gradient_sobelyneg,0)

    gradient_sobelyfin = cv2.addWeighted(gradient_sobely,0.5, gradient_sobelynegflip,0.5,0)
    gradient_sobelxfin = cv2.addWeighted(gradient_sobelx,0.5, gradient_sobelxnegflip,0.5,0)
    gradient_fin = cv2.addWeighted(gradient_sobelxfin,0.5, gradient_sobelyfin,0.5,0)

    _, gradient_fin = cv2.threshold(gradient_fin, 50, 100, cv2.THRESH_BINARY)

    edges = cv2.Canny(gradient_fin, 150, 255, apertureSize=3)
    # edges = cv2.Canny(gray,35,150,apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,minLineLength=150, maxLineGap=50)

    width2 = 0
    if lines is not None and lines.size >1:
        #print(len(lines))
        bundler = HoughBundler(min_distance=20, min_angle=5)
        lines = bundler.process_lines(lines)
        if lines.shape[0]>1: 
            #print(lines.shape)
            midpoints = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(gradient_fin, (x1, y1), (x2, y2), (255, 255, 255), 2)
            line = lines[0][0]
            
            #print(line)
            x1, y1, x2, y2 = line
            x = int((x2+x1)/2)
            y = int((y2+y1)/2)
            midpoints.append(x)
            midpoints.append(y)

            line = lines[1][0]
            #print(line)
            x1, y1, x2, y2 = line
            x = int((x2+x1)/2)
            y = int((y2+y1)/2)
            midpoints.append(x)
            midpoints.append(y)


            #print(midpoints)

            line = lines[0][0]
            #print(lines)
            x1, y1, x2, y2 = line
            #width = math.dist([midpoints[0], midpoints[2]], [midpoints[1], midpoints[3]])
            width = math.sqrt( (midpoints[0] - midpoints[2])**2 + (midpoints[1] - midpoints[3])**2 )


            p1 = np.asarray([x1, y1])
            p2 = np.asarray([x2, y2])
            p3 = np.asarray([midpoints[2], midpoints[3]])
            #print(p1)
            #print(p2)
            #print(p3)
            width2 = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
            #print(width)
            #print(width2)


    return gradient_fin, width2

if __name__ == '__main__':
    camera_setup(URLV)
    camera_setup(URLP)
    i=0

    requests.post(printerURL+"G91")

    averageWidth=0
    avgWidth = [0]*10
    avgWidth2 = [0]*10
    while True:
        if capV.isOpened():
            _, frameV = capV.read()
            _, frameP = capP.read()
            
            #cv2.setMouseCallback('CapV', click_event)

            #cv2.circle(frameV, (766, 100), 3, (0, 255, 255), -1) # top left
            #cv2.circle(frameV, (848, 128), 3, (0, 255, 255), -1) # top right
            #cv2.circle(frameV, (782, 475), 3, (0, 255, 255), -1) # bottom left
            #cv2.circle(frameV, (863, 464), 3, (0, 255, 255), -1) # bottom right
        
            cv2.imshow('CapVeel', frameV)
            
            # cv2.circle(frameP, (164, 83), 5, (0, 0, 255), -1) # top left
            # cv2.circle(frameP, (295, 30), 5, (0, 0, 255), -1) # top right
            # cv2.circle(frameP, (169, 425), 5, (0, 0, 255), -1) # bottom left
            # cv2.circle(frameP, (300, 474), 5, (0, 0, 255), -1) # bottom right

            # 4mm = 400 px
            # 1mm = 100 px

            pts1 = np.float32(
                [[766, 100],  # top left
                 [847, 128],  # top right
                    [783, 475],  # bottom left
                    [862, 464]]  # bottom right
            )
            pts2 = np.float32(
                [[410, 10],  # top left
                 [810, 10],  # top right
                 [410, 810],  # bottom left
                 [810, 810]]  # bottom right
            )
            matrix1 = cv2.getPerspectiveTransform(pts1, pts2)       #Vasaku kaamera perspektiivi muutmine
            frameV = cv2.warpPerspective(frameV, matrix1, (820,820))
            cv2.imshow('CapVpost', frameV)

            #pts3 = np.float32(
            #    [[164, 84],  # top left
            #     [295, 30],  # top right
            #        [167, 425],  # bottom left
            #        [300, 474]]  # bottom right
            #)
            #pts4 = np.float32(
            #    [[50, 42],  # top left
            #     [650, 42],  # top right
            #        [50, 942],  # bottom left
            #        [650, 942]]  # bottom right
            #)
            #matrix2 = cv2.getPerspectiveTransform(pts3, pts4)       #Parema kaamera perspektiivi muutmine
            #frameP = cv2.warpPerspective(frameP, matrix2, (1000, 1000))
            
            ret, frameV = cv2.threshold(frameV, 20, 200, cv2.THRESH_BINARY)
            # ret, frameP = cv2.threshold(frameP, 50, 100, cv2.THRESH_BINARY)
            frameV = frameV[0:1000, 300:1000] #Lõikab vasaku kaamera ebavajaliku sektsiooni välja
            joonedV, widthpx = jooneleidmine(frameV) #joone leidmine laiusega    
            
            widthpx = int(widthpx*1000)
            widthmm = widthpx/115000
            widthmm = round(widthmm,3)
            
            avgWidth.append(widthmm)
            
            finWidth = round(trim_mean(avgWidth, 0.4),3)
            if(len(avgWidth)>9):
                avgWidth.pop(0)

            avgWidth2.append(finWidth)
            if(len(avgWidth2)>9):
                avgWidth2.pop(0)

            finWidth = np.median(avgWidth2)
            finWidth = round(finWidth,2)
            #if 0.1 < finWidth < 0.4:
                #requests.post(printerURL+"G1 Z-0.1")
            #if 0.41 < finWidth < 0.6 :
                #requests.post(printerURL+"G1 Z0.1")
            
            cv2.putText(joonedV,str(finWidth),(200,500),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
            cv2.imshow('Vasak', frameV)
            cv2.imshow('Jooned', joonedV)
            

            key = cv2.waitKey(1)
            if key == 27:
                break

    cv2.destroyAllWindows()
    capV.release()

