import cv2


def resize(x, size=64):
    return cv2.resize(x, dsize=(size, size),
                       interpolation=cv2.INTER_CUBIC)

