def BatchClaheEqualizer(imgs, method, tileSize = 8, clipLim = 2.0):
    sample_num = imgs.shape[0]
    final_imgs = np.zeros((imgs.shape[0],imgs.shape[1],imgs.shape[2]), dtype=np.int)
    
    if method == "clahe":    
        clahe = cv2.createCLAHE(clipLimit=clipLim, tileGridSize=(tileSize,tileSize))
        for i in range(sample_num):
            final_imgs[i] = clahe.apply(cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY))
        return np.expand_dims(final_imgs, axis=-1)
    elif method == "hist":
        for i in range(sample_num):
            final_imgs[i] = cv2.equalizeHist(cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY))
        return np.expand_dims(final_imgs, axis=-1)
    else:    
        for i in range(sample_num):
            final_imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        return imgs
    
X_train_equ = BatchClaheEqualizer(X_train, "clahe")