import tensorflow as tf  

# if you just run the app with python app_flags.py, then the code will use default value as parameters.  
# 若 python app_flags.py --A <value of A> --B <value of B>

FLAGS = tf.app.flags.FLAGS  
  

tf.app.flags.DEFINE_string("author", "Maxim", "your name")  
tf.app.flags.DEFINE_integer("A", 3, "A constant")  
tf.app.flags.DEFINE_integer("B", 13, "B constant")  
  
def main(unused_argv):  
    # using basic constant operations as example

    a = tf.constant(FLAGS.A)
    b = tf.constant(FLAGS.B)
    
    # Launch the default graph.
    with tf.Session() as sess:
        print("Addition with constants: %i" % sess.run(a+b))
        print("Multiplication with constants: %i" % sess.run(a*b))
  
        # sv.saver.save(sess, "/home/yongcai/tmp/")  
  
  
if __name__ == '__main__':  
    tf.app.run()   # parse arguments, 解析命令行参数，调用main 函数 main(sys.argv)  