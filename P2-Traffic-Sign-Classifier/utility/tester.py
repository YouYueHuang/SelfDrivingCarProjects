import numpy as np

def test_unit():
	return np.random.uniform()


# ref: https://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html
# class preprocess_pipeline:
#     """
#     Generate batches with stratified Sampling
#     args:
#         class_dist: the balanced label distribution of all samples (test, valid, train)
#         tr_X: train features
#         tr_Y: train labels

#     returns:

#     """
#     def __init__(self, class_dist, tr_X, tr_Y, batchsize = None):
#         self.class_dist = class_dist
#         self.tr_X = tr_X
#         self.tr_Y = tr_Y   