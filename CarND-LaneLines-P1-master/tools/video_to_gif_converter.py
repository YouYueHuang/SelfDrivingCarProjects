import sys
import os
import getopt
from os.path import join, basename, isfile
import imageio
import cv2
import math
from os import listdir

def main(argv):
   inputdir = None

   try:
      opts, args = getopt.getopt(argv,"hi:",["idir="])
   except getopt.GetoptError as err:
      print (str(err))
      print ('please follow the syntax: video_to_gif_converter.py \
         -i <input directory> \
         width height')
      sys.exit(2)

   # argument parsing
   for opt, arg in opts:
      if opt == '-h':
         print ('valid syntax: video_to_gif_converter.py \
         -i <input directory> \
         width height')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputdir = arg
         if not os.path.exists(inputdir):
            print ('input directory does exist>"')
            sys.exit(2)

   if inputdir is None:
      print ('input file could not be empty')
      sys.exit(2)

   # set gif size
   height = 100
   width = int(height * 1.618)

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