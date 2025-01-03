import cv2
import numpy as np
import scipy.fft
# import lib.phase as phase
# import lib.unwrap as unwrap
# import lib.opd as opd
#

image = cv2.imread("off-axis-data/1.tiff", cv2.IMREAD_GRAYSCALE)
fft = scipy.fft.fft2(image)
print(type(fft))
#
# for_showing = np.abs(fft)
#
# cv2.imshow("this is fft", fft)
# ifft = scipy.fft.ifft2(fft)
# cv2.imshow("this is ifft", ifft)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
