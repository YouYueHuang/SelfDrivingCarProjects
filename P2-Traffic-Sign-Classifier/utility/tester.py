import numpy as np

def test_unit():
	return np.random.uniform()

def image_Normalization(img_batches):
    return ((np.float32(img_batches)-128.0) / 128.0)

def grayscale(img_batches):
    return np.expand_dims(0.21 * img_batches[:,:,:,0] + 0.72 * img_batches[:,:,:,1] + 0.07 * img_batches[:,:,:,2], axis=-1)

def grayscale_cv2(img):
    """
    input is an uint8 type image
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# ref: https://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html