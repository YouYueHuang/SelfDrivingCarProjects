class batch_Generator_StratifiedSampling:
    """
    Generate batches with stratified Sampling
    args:
        class_dist: the balanced label distribution of all samples (test, valid, train)
        tr_X: train features
        tr_Y: train labels
        
    returns:
        batch generator pairs: (train_index, test_index), use train_X[test_index], train_Y[test_index] to get batch for model training.
    """
    def __init__(self, class_dist, tr_X, tr_Y, batchsize = None):
        self.class_dist = class_dist
        self.tr_X = tr_X
        self.tr_Y = tr_Y
        
        # make sure all the classes are in each batch
        # The class distribution of the batch match that of mother 
        self.batchsize = np.sum(np.round(class_dist/min(class_dist))) if (batchsize == None) else batchsize # default is min_batch_size
        
        self.t_size = self.batchsize / tr_X.shape[0]
        self.num_split = math.ceil(tr_X.shape[0]/self.batchsize)
    
    def batches(self):

        # draw sample for 1 epoch
        sss = StratifiedShuffleSplit(n_splits=self.num_split, test_size=self.t_size, random_state=0)
        sss.get_n_splits(self.tr_X, self.tr_Y)
        return sss.split(self.tr_X, self.tr_Y)