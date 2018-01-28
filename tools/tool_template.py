import sys
import os
import getopt
# if len(sys.argv) < 2:
#     sys.exit('Usage: %s database-name' % sys.argv[0])

# if not os.path.exists(sys.argv[1]):
#     sys.exit('ERROR: Database %s was not found!' % sys.argv[1])


# if len(sys.argv) < 2:
#     sys.stderr.write('Usage: sys.argv[0] ')
#     sys.stderr.write('%s'% sys.argv)
#     sys.exit(1)


# if not os.path.exists(sys.argv[1]):
#     sys.stderr.write('ERROR: Database sys.argv[1] was not found!')
#     sys.exit(1)

def main(argv):
   inputfile = None
   outputfile = None

   # if len(sys.argv) < 2:
   #     sys.exit('Usage: %s database-name' % sys.argv[0])

   # if not os.path.exists(sys.argv[1]):
   #     sys.exit('ERROR: Database %s was not found!' % sys.argv[1])

   try:
      """
      syntax: getopt.getopt(args, options, [long_options])

      * args − This is the argument list to be parsed.

      * options − This is the string of option letters that the script wants to recognize, 
      with options that require an argument should be followed by a colon (:).

      * long_options − This is optional parameter and if specified, 
      must be a list of strings with the names of the long options, 
      which should be supported. Long options, 
      which require an argument should be followed by an equal sign ('='). 
      To accept only long options, options should be an empty string.

      This method returns value consisting of two elements: 
      the first is a list of (option, value) pairs. 
      The second is the list of program arguments left after the option list was stripped.

      Each option-and-value pair returned has the option as its first element, 
      prefixed with a hyphen for short options (e.g., '-x') 
      or two hyphens for long options (e.g., '--long-option').
      """ 





	   # the second flag (-i) (-o) must be followed by an argument, 
	   # which is the name of the file to read.  
	   # So you tell getopt this by putting a colon after the i in that second parameter to the getopt function.
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
      # The --ifile and --ofile flag must always be followed by an additional argument, just like the -i flag. 
      # This is notated by an equals sign =
   except getopt.GetoptError:
      print ('please follow the syntax: video_clipper.py -i <inputfile> -o <outputfile>')
      sys.exit(2)



   for opt, arg in opts:
      if opt == '-h':
         print ('the syntax: "video_clipper.py -i <input file name> -o <output file name>"')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print (opts)
   # print ('Input file is :', inputfile)
   # print ('Output file is :', outputfile)

if __name__ == "__main__":
   	main(sys.argv[1:])