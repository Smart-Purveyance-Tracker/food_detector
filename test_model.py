from modules.model.utils import detect

from cv2 import cv2


if __name__ == '__main__':

    image = cv2.imread('/home/vadbeg/Data/Docker_mounts/food/_5235025_39542c5132_o.jpg')

    detect(image=image)

