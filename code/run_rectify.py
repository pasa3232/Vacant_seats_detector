import numpy as np
import rectify
from PIL import Image
import time

if __name__ == '__main__':
    # read images
    # img0 = cv2.imread('../data/00/cam0/0001.jpg')
    # img1 = cv2.imread('../data/00/cam1/0001.jpg')
    # img2 = cv2.imread('../data/00/cam2/0001.jpg')
    # img3 = cv2.imread('../data/00/cam3/0001.jpg')

    start = time.time()
    img_rec = Image.open('../data/00/cam0/0001.jpg').convert('RGB')

    igs_rec = np.array(img_rec)
    c_in, c_ref = rectify.set_cor_rec()
    igs_rec = rectify.rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('0001.png')
    end = time.time()
    print(f'total time: {end - start}')