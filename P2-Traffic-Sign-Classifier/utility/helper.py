# Load pickled data

def plot_label_image(file):
	image_every_class=dict()
	for im,cl in zip(train['features'],train['labels']):
	    image_every_class[cl]=im

	image_every_class=list(image_every_class.values())

	cols=7
	rows=7

	for y in range(0,1+int(len(image_every_class)/cols)):
	    if y==6: #last row
	        img_last=image_every_class[y*cols:(y+1)*cols][0]
	        img_empty=np.zeros((32, 192, 3))
	        img_row=np.concatenate(image_every_class[y*cols:(y+1)*cols]*7,axis=1)
	        img_row[:,32:]=0
	    else:
	        img_row=np.concatenate(image_every_class[y*cols:(y+1)*cols],axis=1)
	        
	    if y==0:
	        image=img_row
	    else:
	        image=np.concatenate((image,img_row),axis=0)        

	for y in range(0,1+int(len(image_every_class)/cols)):
	    for x in range(cols):
	        if(x+y*cols<n_classes):
	            cv2.putText(image,str(x+y*cols),(11+x*32,11+y*32), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,0,0),1,cv2.LINE_AA)

	plt.axis('off');
	plt.imshow(image);