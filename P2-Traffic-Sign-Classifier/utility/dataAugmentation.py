import cv2 
import numpy as np
import matplotlib.gridspec as gridspec
import pickle
from tqdm import tqdm

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img, ang_range, shear_range, trans_range, brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values for affine transformation
    4- trans_range: Range of values for translation

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range) - ang_range/2  # set the angle between -ang_range/2 < x < ang_range/2
    rows,cols,ch = img.shape    
    """
    getRotationMatrix2D(Point2f center, double angle, double scale)
    * center: rotation center of input 
    * angle: rotation angel, the reference is the left top point, if it is positive
    * scale: scale rate, the output is a 2x3 matrix (affine matrix)
    """
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
        img = augment_brightness_camera_images(img)
    return img

# training_file = 'datasets//train.p'
# train = None
# with open(training_file, mode='rb') as f:
#     train = pickle.load(f)

# MINIMAL_IMAGES_COUNT=600
# pbar = tqdm(range(len(train['features'])), desc='Image', unit='images')
# n_augment = 10
# for i in pbar:
#     cl=train['labels'][i]
#     if label_counter[cl] < MINIMAL_IMAGES_COUNT:
#         for i in range(n_augment):
#             img = transform_image(im,20,10,1)
#             img = img.reshape(1,32,32,3)
#             train['features']=np.concatenate((train['features'],img),axis=0)
#             train['labels']=np.concatenate((train['labels'],[cl]))
#         label_counter[cl]=label_counter[cl]+n_augment

# aug_training_file = 'datasets//aug_train.p'
# with open(aug_training_file, 'wb') as handle:
#     pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(argv):
   inputdir = None

   try:
      opts, args = getopt.getopt(argv,"hi:",["idir="])
   except getopt.GetoptError as err:
      print (str(err))
      print ('please follow the syntax: dataAugmentation.py \
         -i <input directory> \
         -o <output directory> \
         -m img_min_num')
      sys.exit(2)

   # argument parsing
   for opt, arg in opts:
      if opt == '-h':
	      print ('please follow the syntax: dataAugmentation.py \
	         -i <input directory> \
	         -o <output directory> \
	         -m img_min_num')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputdir = arg
         if not os.path.exists(inputdir):
            print ('input directory does exist>"')
            sys.exit(2)

   if inputdir is None:
      print ('input file could not be empty')
      sys.exit(2)


   if len(args) == 2:
      width = args[0]
      height = args[1]
   elif len(args) == 0:
      pass
   else:
      print ('incorrect gif size')
      sys.exit(2) 
   try:
      width = int(width)
      height = int(height)
   except ValueError:
      print ('gif size should be a number')
      sys.exit(2)  

   # gif output path
   gifSaved_dir = "gif_output"
   if not os.path.exists(gifSaved_dir):
       os.makedirs(gifSaved_dir)

   # extract only files in input directory
   for f in listdir(inputdir): 
      inputFile = join(inputdir, f)
      if isfile(inputFile) and f.split('.')[-1] == 'mp4':

         # load video and set gif fps
         reader = imageio.get_reader(inputFile)
         fps = reader.get_meta_data()['fps']

         print("gif generating starts")
         
         out_path = join(gifSaved_dir,basename(f).split('.')[0] + '.gif')

         with imageio.get_writer(out_path, mode='I', fps=fps) as writer:
             for im in reader:
                 resized_image = cv2.resize(im, (width, height)) 
                 writer.append_data(resized_image)

         print ('Output file is :', out_path)

if __name__ == "__main__":
   	main(sys.argv[1:])