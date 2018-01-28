import sys
import os
import getopt
from os.path import join, basename
from moviepy.editor import VideoFileClip
import math

def main(argv):
   inputfile = None

   try:
      opts, args = getopt.getopt(argv,"hi:",["ifile="])
   except getopt.GetoptError as err:
      print (str(err))
      print ('please follow the syntax: video_clipper.py\
                                       -i <input file name.mp4>\
                                       start_time\
                                        end_time')
      sys.exit(2)

   # argument parsing
   for opt, arg in opts:
      if opt == '-h':
         print ('valid syntax: "video_clipper.py\
                               -i <input file name.mp4>\
                                 start_time end_time"')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
         if inputfile.split('.')[-1] != 'mp4':
            print ('input file is not the format of mp4>"')
            sys.exit(2)
         if not os.path.exists(inputfile):
            sys.exit('video %s was not found!' % inputfile)
            sys.exit(2)

   if inputfile is None:
      print ('input file could not be empty')
      sys.exit(2)

   # load video
   origian_clip = VideoFileClip(inputfile)

   # clipping time 
   start_t = 0
   end_t = origian_clip.duration

   if len(args)>2:
      print ('incorrect clipping time')
      sys.exit(2)  
   else:
      if len(args) == 2:
         start_t = args[0]
         end_t = args[1]
      elif len(args) == 1:
         start_t = args[0]
      else:
         pass
   try:
      start_t = float(start_t)
      end_t = float(end_t)
   except ValueError:
      print ('clipping time should be a number')
      sys.exit(2)        



   ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
   ## To do so add .subclip(start_second,end_second) to the end of the line below
   ## Where start_second and end_second are integer values representing the start and end of the subclip

   if start_t > origian_clip.duration or end_t > origian_clip.duration:
      print ('clipping time exceeds video length:'+ str(origian_clip.duration))
      sys.exit(2)  

   print("clippling starts")
   
   clip1 = origian_clip.subclip(start_t,end_t)
   # processed_clip = clip1.fl_image(process_image) 
   # process_image is a image processing pipeline
   # the output should be color image
   
   # output video
   videoSaved_dir = "clipped_video_output"
   if not os.path.exists(videoSaved_dir):
       os.makedirs(videoSaved_dir)
   
   outputfile = "clipped_" + str(start_t) + "_to_" + str(end_t) + "_" + basename(inputfile)
   out_path = join(videoSaved_dir,outputfile)
   clip1.write_videofile(out_path, audio=False)
   print ('Output file is :', out_path)

if __name__ == "__main__":
   	main(sys.argv[1:])