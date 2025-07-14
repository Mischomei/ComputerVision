import time
from xmlrpc.client import MININT

import cv2 as cv
import numpy as np
import imutils
import os
from pathlib import Path
from src.extras.trigonometry import len_line
from cv2 import aruco

MASK_REP = 5
CAMERA_PARAM_REP = 5

class ImageProcessor:
    debug = False

    def __init__(self, debug=False):
        self.debug = debug


    def undistort(self, folder, image, camera, destFolder=None):
        img = cv.imread(folder / image)
        #debug
        if self.debug:self.showimg(img, "img")
        h,w = img.shape[:2]

        #Optimisierung der Kameramatrix passend zur Auflösung des Bildes
        newcameramatrix, roi = cv.getOptimalNewCameraMatrix(camera.cameraMatrix, camera.dist, (w,h), 1, (w, h))
        #Unverzerrung
        dest = cv.undistort(img, camera.cameraMatrix, camera.dist, None, newcameramatrix)
        x, y, w, h = roi
        dest = dest[y:y+h, x:x+w]
        if destFolder:
            cv.imwrite(os.path.join(destFolder, f"undistorted_{image}"), dest)

        return dest


    def reprojectionerror(self, camera, objPoints, imgPoints):
        mean_error = 0
        for i in range(len(objPoints)):
            imgPoints2, _ = cv.projectPoints(objPoints[i], camera.rvecs[i], camera.tvecs[i], camera.cameraMatrix, camera.dist)
            error = cv.norm(imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints)
            mean_error += error
        print(f"Fehler: {mean_error/len(objPoints)}")


    def get_color_channels(self, img, mask):
        distparam = 10
        img_copy = img.copy()
        b, g, r = cv.split(img_copy)
        blank = np.zeros(img.shape[:2], dtype="uint8")


        #Gaussian Blur
        b_blur = cv.GaussianBlur(b, (3, 3), 0)
        g_blur = cv.GaussianBlur(g, (3, 3), 0)
        r_blur = cv.GaussianBlur(r, (3, 3), 0)

        #Edge detection on colorchannels
        b_edges = cv.Canny(b_blur, 25, 125, apertureSize=3)
        g_edges = cv.Canny(g_blur, 25, 125, apertureSize=3)
        r_edges = cv.Canny(r_blur, 25, 125, apertureSize=3)


        combthresh = cv.bitwise_or(b_edges, g_edges, r_edges)
        combthresh = cv.morphologyEx(combthresh, cv.MORPH_CLOSE, None, iterations=2)
        threshmask = self.edit_thresh_mask(mask, distparam)
        invmask = self.edit_inv_mask(mask, distparam)


        combthresh = cv.bitwise_and(combthresh, combthresh, mask=threshmask)
        combthresh = cv.bitwise_and(combthresh, combthresh, mask=invmask)

        #debug
        if self.debug: self.showimg(combthresh, "combthresh")

        return combthresh


    def lines(self, edge, eps=-1):

        #Approxpoly
        cons = cv.findContours(edge, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        cons = imutils.grab_contours(cons)
        c = max(cons, key=cv.contourArea)

        if eps != -1:
            arc = cv.arcLength(c, True)
            eps = eps * arc


        #poly = cv.approxPolyN(c, 6, approxCurve=np.ndarray([]), epsilon_percentage=eps, ensure_convex=True)

        poly = cv.approxPolyDP(c, 0.006 * cv.arcLength(c, True), True, approxCurve=np.ndarray([]))
        outpoly = poly.astype(int)
        outpoly = np.array([out[0] for out in outpoly])
        if len(outpoly) >= 5:
            poly = cv.approxPolyN(c, 6, approxCurve=np.ndarray([]), epsilon_percentage=eps, ensure_convex=True)
        elif len(outpoly) <= 4:
            poly = cv.approxPolyN(c, 4, approxCurve=np.ndarray([]), epsilon_percentage=eps, ensure_convex=True)
        outpoly = poly[0].astype(int)
        #debug


        #sorting so corner at bottom is 0
        maxpoly = (0, 0)
        maxlen = 0
        index = 0

        print(outpoly)
        for i in range(1, len(outpoly) ):

            vec = outpoly[i] - outpoly[i-1] if i <=len(outpoly)-1 else outpoly[0] - outpoly[i-1]
            lenvec = np.linalg.norm(vec, ord=2)
            if lenvec >= maxlen*0.9:
                index = i-1
                maxlen = lenvec

        outpoly = np.concatenate((outpoly[index:], outpoly[:index]), axis=0)
        if outpoly[0][1] == np.min(np.stack(outpoly, axis=1)[1]):
            outpoly = np.roll(outpoly, 3, axis=0)
        # if outpoly[0][0] == np.min(np.stack(outpoly, axis=1)[0]):
        #     outpoly = np.roll(outpoly, 3, axis=0)
        #outpoly = np.roll(outpoly, index, axis=0)
        print(outpoly)
        return poly, outpoly


    def drawcontour(self, img, points, poly=None, color=None):
        copyimg = img.copy()

        if poly is not None:
            cv.drawContours(copyimg, [poly], 0, (255, 0, 0), 5)
        else:
            for index, point in enumerate(points):
                if index+1 < np.size(points, 0):
                    cv.line(copyimg, points[index], points[index+1], (255, 0, 0), 5)
                else:
                    cv.line(copyimg, points[index], points[0], (255, 0, 0), 5)
        for index, point in enumerate(points):
            cv.circle(copyimg, (point[0], point[1]), 5, (0, 0, 255), -1)
            cv.putText(copyimg, str(index), (point[0] + 5, point[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if color:
            minpoint = min(points, key = lambda t:t[1])
            cv.putText(copyimg, color, minpoint-(15, 15), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)

        return copyimg


    #Drehen vom Bild
    def rotate(self, img, angle=90):
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        return img


    #Croppen vom Bild
    def crop(self, img, cropx_s=None, cropx_e=None, cropy_s=None, cropy_e=None):
        cxs = cropx_s if cropx_s else 0
        cxe = cropx_e if cropx_e else img.shape[1]
        cys = cropy_s if cropy_s else 0
        cye = cropy_e if cropy_e else img.shape[0]
        img_copy = img.copy()
        return img_copy[cxs:cxe, cys:cye]


    def create_mask(self, img, lower, upper):
        img_copy = img.copy()
        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img_copy, lower, upper)
        mask = cv.threshold(mask, 2, 255, cv.THRESH_BINARY)[1]
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, None, iterations=2)
        if self.debug: self.showimg(mask, "created mask")

        return mask



    def edit_thresh_mask(self, mask, dist_param):
        #Dilating mask to create outer object selection
        threshmask = cv.morphologyEx(mask, cv.MORPH_ERODE, (20, 20), iterations=2)
        threshmask = cv.morphologyEx(threshmask, cv.MORPH_DILATE, None, iterations=dist_param)
        threshmask = cv.morphologyEx(threshmask, cv.MORPH_OPEN, (20, 20), iterations=2)

        #debug
        if self.debug: self.showimg(threshmask, "threshmask")
        return threshmask


    def edit_inv_mask(self, mask, dist_param):
        #Eroding Mask to create inner removal mask
        invmask = cv.morphologyEx(mask, cv.MORPH_ERODE, None, iterations=dist_param)
        invmask = cv.morphologyEx(invmask, cv.MORPH_OPEN, None, iterations=1)
        invmask = cv.bitwise_not(invmask)

        #debug
        if self.debug: self.showimg(invmask, "invmask")
        return invmask

    #Weiß noch nicht ob ich das brauche ???
    def cornersto3d(self, corners):
        if len(corners) == 4:
            pass

        #AUs 6 erkennbaren des Quaders ein Sechseck und daraus seine 3D koordinaten
        if len(corners) == 6:
            # Aus Ecken Kanten machen
            pass

    #sollte in extras sein???
    def depthestimatetest(self, img, realsize, imgsize, f, sensor):
        return ()

    #Kp wofür das ist
    def refine_edges(self):
        pass

    #Kp wofür das ist Pt.2
    def calc_coords_from_edges(self):
        pass

    #Erklärt sich von selbst
    def calc_coords_from_corners(self):
        pass

    #Erstellt Maske für HSV Farbräume
    def createmasks(self, img, colors):
        imgcopy = img.copy()
        outmasks = dict()

        for color in colors:
            mask = self.create_mask(imgcopy, color[0], color[1])
            if cv.hasNonZero(mask):
                outmasks[color[2]] = mask

        return outmasks

    def maskstolist(self, masks):
        out = []
        for key in masks.keys():
            out.append((masks[key], key))
        return out

    #Für alle Farben jeweils seperate Kantenerkennungen und diese kombinieren
    def cons_per_color(self, img, masks):
        copyimg = img.copy()
        outimg = img.copy()
        outpoints = []
        for mask in masks:
            edges = self.get_color_channels(copyimg, mask[0])
            polyn, points = self.lines(edges)
            outpoints.append(points)
            outimg = self.drawcontour(outimg, points, polyn, mask[1])

        #debug
        if self.debug: self.showimg(outimg, "cons_per_color")

        return outimg, outpoints


    #Anzeigen des Bildes
    def showimg(self, img, name):
        copy = img.copy()
        copy = self.rotate(self.rotate(copy))
        # if copy.shape[0] > 1920:
        #     factor = 1920/copy.shape[0]
        #     copy = cv.resize(copy, None, fx=factor, fy=factor, interpolation=cv.INTER_LINEAR)
        # if copy.shape[1] > 1080:
        #     factor = 1080/copy.shape[1]
        #     copy = cv.resize(copy, None, fx=factor, fy=factor, interpolation=cv.INTER_LINEAR)
        cv.imshow(name, copy)
        cv.waitKey(0)
        cv.destroyWindow(name)

    #wtf ist sbm???
    def sbm(self):
        out = np.array()
        while out.empty():
            for i in range(MASK_REP):
                try:
                    pass
                except:
                    pass


    def generate_aruco(self, dest, aruco_dict, num_markers, marker_size, start_id):
        arucodict = cv.aruco.getPredefinedDictionary(aruco_dict)
        id = start_id
        for i in range(num_markers):
            img_marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
            img_marker = cv.aruco.generateImageMarker(arucodict, id, marker_size, img_marker, 1)
            cv.imwrite(dest / f"marker_{id}.jpg", img_marker)
            id+=1
            #debug
            if self.debug: self.showimg(img_marker, "aruco_marker")


    #Eigentlich unnötig, lasse ich vlt für noch irgendwas
    def detect_aruco(self, img, aruco_dict_type):
        aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
        params = aruco.DetectorParameters()
        # params.adaptiveThreshConstant = 3
        # params.aprilTagCriticalRad = 0
        # params.aprilTagMaxNmaxima = 15
        # params.aprilTagMinWhiteBlackDiff = 0
        params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        # params.cornerRefinementMaxIterations = 50
        # params.errorCorrectionRate = 0.9
        # params.polygonalApproxAccuracyRate = 0.1
        # params.useAruco3Detection = True
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 300
        # params.minMarkerPerimeterRate = 0.01
        params.adaptiveThreshConstant = 2
        # params.adaptiveThreshWinSizeStep = 10
        # params.perspectiveRemovePixelPerCell = 10
        # params.perspectiveRemoveIgnoredMarginPerCell = 0.05
        # params.minMarkerPerimeterRate =
        detector = aruco.ArucoDetector(aruco_dict, params)
        imgcopy = img.copy()
        # gray = cv.cvtColor(imgcopy, cv.COLOR_BGR2GRAY)
        marker_corners, marker_ids, rejected_corners = detector.detectMarkers(imgcopy)
        return marker_corners, marker_ids

    def aruco_pose_estimation(self, img, aruco_dict_type, marker_size, matrix_coeffs, distortion_coeffs):
        r_vecs, t_vecs = list(), list()

        markerPoints = np.array([[-marker_size/2, marker_size/2, 0],
                                 [marker_size/2, marker_size/2, 0],
                                 [marker_size/2, -marker_size/2, 0],
                                 [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)

        aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
        params = aruco.DetectorParameters()
        #params.adaptiveThreshConstant = 3
        #params.aprilTagCriticalRad = 0
        #params.aprilTagMaxNmaxima = 15
        #params.aprilTagMinWhiteBlackDiff = 0
        params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        #params.cornerRefinementMaxIterations = 50
        #params.errorCorrectionRate = 0.9
        #params.polygonalApproxAccuracyRate = 0.1
        #params.useAruco3Detection = True
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 300
       # params.minMarkerPerimeterRate = 0.01
        params.adaptiveThreshConstant = 2
        #params.adaptiveThreshWinSizeStep = 10
        #params.perspectiveRemovePixelPerCell = 10
        #params.perspectiveRemoveIgnoredMarginPerCell = 0.05
        #params.minMarkerPerimeterRate =
        detector = aruco.ArucoDetector(aruco_dict, params)
        imgcopy = img.copy()
        #gray = cv.cvtColor(imgcopy, cv.COLOR_BGR2GRAY)
        marker_corners, marker_ids, rejected_corners = detector.detectMarkers(imgcopy)

        print(marker_ids)
        ret = False


        if marker_ids is not None:
            for i in range(len(marker_ids)):
                ret, rvec, tvec = cv.solvePnP(markerPoints, marker_corners[i], matrix_coeffs, distortion_coeffs)
                if ret:

                    r_vecs.append(rvec)
                    t_vecs.append(tvec)
                ret = True

        return ret, r_vecs, t_vecs


    def pose_drawing(self, img, r_vecs, t_vecs, marker_size, mat, dist):
        imgcopy  = img.copy()
        if len(r_vecs) == len(t_vecs):
            for i in range(len(r_vecs)):
                cv.drawFrameAxes(imgcopy, mat, dist, r_vecs[i], t_vecs[i], marker_size)
        return imgcopy


    def trypnp(self, img, objpoints, points, matrix, dist, size):
        ret, rvec, tvec = cv.solvePnP(objpoints, points, matrix, dist)
        return ret, rvec, tvec


    def start_searching(self):
        pass


    def create_charuco(self, checkboardSize, squarelength, markerlength, dictionary, folder):
        arucodict = cv.aruco.getPredefinedDictionary(dictionary)
        charboard = cv.aruco.CharucoBoard(checkboardSize, squarelength, markerlength, arucodict)
        ratio = checkboardSize[1] / checkboardSize[0]
        return charboard



