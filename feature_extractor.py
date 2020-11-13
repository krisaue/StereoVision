import cv2 as cv

AKAZE = 0
KAZE = 1
ORB = 2
SIFT = 3  # SIFT not currently supported by publicly available in OpenCV 4.2.0


class Exctractor():

    def __init__(self, descriptor, **kwargs):
        if descriptor == AKAZE:
            self.descriptor = cv.AKAZE_create()
        elif descriptor == KAZE:
            self.descriptor = cv.KAZE_create()
        elif descriptor == ORB:
            self.descriptor = cv.ORB_create(**kwargs)
        elif descriptor == SIFT:
            self.descriptor = cv.xfeatures2d.SIFT_create()

    def __repr__(self):
        return self.descriptor

    def extract_feature(self, image):

        cimage = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        gimage = cv.cvtColor(cimage, cv.COLOR_RGB2GRAY)

        kp, des = self.descriptor.detectAndCompute(gimage, None)

        feature = []
        for j, point in enumerate(kp):
            temp = (point.pt, point.size, point.angle,
                    point.response, point.octave, point.class_id, des[j])
            feature.append(temp)

        return kp,des,feature
