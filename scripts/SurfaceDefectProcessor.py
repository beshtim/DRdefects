import cv2.cv2 as cv2
import numpy as np
import imutils
import yaml
import math

class SurfaceDefects:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        reference_yaml = cfg["reference_yaml_path"]
        self.thresh = cfg["surface_threshold"]

        with open(reference_yaml, "r", encoding="utf-8") as ymlfile:
            self.cfg = yaml.safe_load(ymlfile)

    def find_defects(self, image: np.ndarray, js):

        yaml_reference = self.cfg['signs']
        reference_shape = yaml_reference[js['class']]

        mask = self.get_mask_from_polygon(image, js['segmentation'][0])

        segs = (js['segmentation'][0])
        x = segs[::2]
        y = segs[1::2]

        defect = False

        if reference_shape == "rectangle":

            pts1 = np.float32(
                [[min(x), min(y)],  # top left
                 [max(x), min(y)],  # top right
                 [min(x), max(y)],  # bottom left
                 [max(x), max(y)]]  # bottom right
            )

            pts2 = np.float32(
                [[100, 100],  # top left
                 [500, 100],  # top right
                 [100, 500],  # bottom left
                 [500, 500]]  # bottom righqt
            )

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(mask, matrix, (700, 700))
            sign_shape, num_verticies = self.detect_shape(result, self.thresh)

            if sign_shape != reference_shape:
                if num_verticies == 4:
                    return str(defect)
                else:
                    defect = True
                    return str(defect)
            else:
                return str(defect)


        if reference_shape == "triangle":
            pts1 = np.float32([[min(x), min(y)],
                               [max(x), min(y)],
                               [min(x), max(y)]])

            pts2 = np.float32([[125, 100],
                               [250, 62],
                               [125, 250]])

            matrix = cv2.getAffineTransform(pts1, pts2)
            result = cv2.warpAffine(mask, matrix, (700, 700))

            sign_shape, num_verticies = self.detect_shape(result, self.thresh)

            if num_verticies != 3:
                defect = True
                return str(defect)

            else:
                return str(defect)

        if reference_shape == "circle":
            pass

    def get_mask_from_polygon(self, image: np.ndarray, segment: list) -> np.ndarray:
        mask = np.zeros(image.shape, dtype=np.uint8)
        polygon = np.array([[segment[i:i + 2] for i in range(0, len(segment), 2)]], dtype=np.int32)
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        new_mask = cv2.fillPoly(mask, polygon, ignore_mask_color)
        # masked_image = cv2.bitwise_and(image, mask)
        # masked_image[(mask == 0).all(-1)] = [255, 255, 255]
        # masked_image = masked_image[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2])]
        return new_mask

    def detect_shape(self, image, tresh):
            resized = imutils.resize(image, width=300)
            ratio = image.shape[0] / float(resized.shape[0])
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            shape = 'undefined'
            for c in cnts:
                M = cv2.moments(c)
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
                shape = self.shape_detector(c,tresh)
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.putText(image, shape[0], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
            return shape

    def shape_detector(self, c, tresh):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        long_verticies_num = []
        for i in range(1, len(approx)):
            x2 = approx[i][0][0]
            x1 = approx[i-1][0][0]
            y2 = approx[i][0][1]
            y1 = approx[i-1][0][1]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist > tresh:
                long_verticies_num.append(dist)
            if i+1 == len(approx):
                x2 = approx[0][0][0]
                x1 = approx[i][0][0]
                y2 = approx[0][0][1]
                y1 = approx[i][0][1]
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if dist > tresh:
                    long_verticies_num.append(dist)

        num_verticies = len(long_verticies_num)
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            shape = "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        return shape, num_verticies


