# Training Parameters
learning_rate = 0.0001
epochs = 20
logs_path = 'tensorboard_logs'

# Network Parameters
n_classes = np.unique(y_train).shape[0] # 43 traffic sign classes/labels there are in the dataset.
dropout = 0.8  # Dropout, probability to keep units


def conv2fc(input_height, input_width, filter_size, stride, layers):
    con2fc_height = input_height
    con2fc_width = input_width
    for i in layers:
        con2fc_height = np.ceil((con2fc_height - filter_size[i] + 1) / stride[i])
        con2fc_width = np.ceil((con2fc_width - filter_size[i] + 1) / stride[i])
    return int(con2fc_height) * int(con2fc_width)

layers = ['conv_1', 'maxpool_1', 'conv_2', 'maxpool_2']

# (img shape: 32*32)
layer_input = {
     'depth': 1
    ,'height': X_train.shape[1]
    ,'width': X_train.shape[2]
}

layer_depth = {
    'conv_1': 32
    ,'conv_2': 128
    ,'fc_1': 1024
    ,'fc_2': 1024
    ,'out': n_classes
}

# a kernel size of 3
filter_size={
    'conv_1': 5
    ,'maxpool_1': 2
    ,'conv_2': 5
    ,'maxpool_2': 2
}

stride = {
    'conv_1': 1
    ,'maxpool_1': 2
    ,'conv_2': 1
    ,'maxpool_2': 2
}

# xavier_initializer
# ref https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer
# tf.truncated_normal(shape=[dims[l-1],dims[l]], mean=mu[l], stddev=std[l], dtype=tf.float64)
    
for k,v in sorted(layer_depth.items()):
    print("Layer %5s depth is %s"%(k,v))

# Tensorboard graph
def conv2d(x, W, b, stride):
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k, s):
    # Tensor input is 4-D: [Batch Size, Height, Width, Feature(Channel)]
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID')

def fullycon(x, w, b):
    return tf.nn.relu(tf.nn.xw_plus_b(tf.contrib.layers.flatten(x), w, b))

def conv_net(x, weights, biases, dropout):
    # Input
    print ("input shape is: " + str(x.shape))
    
    # Layer 1 
    conv1 = conv2d(x, weights['conv1'], biases['conv1'], stride['conv_1'])
    print ("conv1 shape is: " + str(conv1.shape))

    conv1 = maxpool2d(conv1, filter_size['maxpool_1'], stride['maxpool_1'])
    print ("maxpool1 shape is: " + str(conv1.shape))

    # Layer 2 
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'], stride['conv_2'])
    print ("conv2 shape is: " + str(conv2.shape))
    
    conv2 = maxpool2d(conv2, filter_size['maxpool_1'], stride['maxpool_2'])
    print ("maxpool2 shape is: " + str(conv2.shape))
    
    # Layer 3
    fc1 = fullycon(conv2, weights['fc1'], biases['fc1'])
    print ("fully connected layer from conv2 {} to fc1 {}".format(conv2.shape,fc1.shape))
    
    fc1 = tf.nn.dropout(fc1, dropout)
    print ("fully connected layer 1 dropout is: {}".format(fc1.shape))
    
    # Layer 4
    fc2 = fullycon(fc1, weights['fc2'], biases['fc2'])
    print ("fully connected layer from fc1 {} to fc2 {}".format(fc2.shape,fc1.shape))
    
    fc2 = tf.nn.dropout(fc2, dropout)
    print ("fully connected layer 2 dropout is: {}".format(fc2.shape))

    # Layer 5: output
    out = tf.nn.xw_plus_b(fc2, weights['out'], biases['out'])
    print ("output shape is: {}".format(out.shape))

    return out

"""
graph computation model
"""

# ref: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py
# tf Graph input

tf.reset_default_graph() 
# g = tf.Graph()
# with g.as_default():
x = tf.placeholder(tf.float32, [None, layer_input["height"], layer_input["width"], 1], name='InputData')
y = tf.placeholder(tf.float32, [None, n_classes], name='LabelData')
keep_prob = tf.placeholder(tf.float32)

conv2fc_num = conv2fc(layer_input["height"], layer_input["width"], filter_size, stride, layers)
weights = {
    'conv1': tf.get_variable("conv1w", shape=[filter_size['conv_1'], filter_size['conv_1']
                                              , layer_input['depth'], layer_depth['conv_1']]
                             , initializer=tf.contrib.layers.xavier_initializer()),
    'conv2': tf.get_variable("conv2w", shape=[filter_size['conv_2'], filter_size['conv_2']
                                              , layer_depth['conv_1'], layer_depth['conv_2']]
                             , initializer=tf.contrib.layers.xavier_initializer()),
    'fc1':  tf.get_variable("fc1w", shape=[conv2fc_num * layer_depth['conv_2'], layer_depth['fc_1']]
                            , initializer=tf.contrib.layers.xavier_initializer()),
    'fc2':  tf.get_variable("fc2w", shape=[layer_depth['fc_1'], layer_depth['fc_2']]
                            , initializer=tf.contrib.layers.xavier_initializer()),
    'out':  tf.get_variable("out1w", shape=[layer_depth['fc_2'], layer_depth['out']]
                            , initializer=tf.contrib.layers.xavier_initializer())
}

# ref https://www.leiphone.com/news/201703/3qMp45aQtbxTdzmK.html
biases = {
    'conv1': tf.get_variable("conv1b", shape=[layer_depth['conv_1']], initializer=tf.contrib.layers.xavier_initializer()),
    'conv2': tf.get_variable("conv2b", shape=[layer_depth['conv_2']], initializer=tf.contrib.layers.xavier_initializer()),
    'fc1': tf.get_variable("fc1b", shape=[layer_depth['fc_1']], initializer=tf.contrib.layers.xavier_initializer()),
    'fc2': tf.get_variable("fc2b", shape=[layer_depth['fc_2']], initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable("out1b", shape=[layer_depth['out']], initializer=tf.contrib.layers.xavier_initializer())
}

for k,v in sorted(weights.items()):
    print("Weights for %5s: %s"%(k,v.shape))

# with tf.name_scope('Model'):
logits = conv_net(x, weights, biases, keep_prob)

# with tf.name_scope('Cost'):
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# with tf.name_scope('Accuracy'):
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()



# Initializing the variables
init = tf.global_variables_initializer()

tf.summary.FileWriter("logs", g)



# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        
        # get batch generator
        bgss = batch_Generator_StratifiedSampling(y_train_dist, X_train_input, y_train_input)
        get_nextBatch = bgss.batches()
        
        batch_id = 0
        for _, train_index in get_nextBatch:  # for batch_id, batch_x, batch_y in batch_iteration(X_train_input, y_train_input, batch_size):
            batch_id += 1
            batch_x = X_train_input[train_index]
            batch_y = y_train_input[train_index]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            
        valid_acc = sess.run(accuracy, feed_dict={
            x: X_valid_input,
            y: y_valid_input,
            keep_prob: 1.})
        print('Epoch {:>2}/{}, Batch {:>3}/{} - Loss: {:>10.1f} Validation Accuracy: {:.6f}'.format(
            epoch + 1,
            epochs,
            batch_id,
            bgss.num_split,
            loss,
            valid_acc))

        # Calculate Test Accuracy
        test_acc = sess.run(accuracy, feed_dict={
            x: X_test_input,
            y: y_test_input,
            keep_prob: 1.})
        print('Testing Accuracy: {}'.format(test_acc))
    
print("--- %s seconds ---" % (time.time() - start_time))