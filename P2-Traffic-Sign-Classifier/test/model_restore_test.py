batches = []
loss_epoch = []
train_acc_epoch = []
valid_acc_epoch = []
validation_accuracy = 0.0
EPOCHS = 2
BATCH_SIZE = 150

# batch_count = int(len(train['features']) / BATCH_SIZE)
batch_count = 5
# Add ops to save and restore all the variables.

# Start training
with tf.Session() as sess_re:
    new_saver = tf.train.import_meta_graph(os.path.join('model', 'mynet_A', 'valid_0.13.ckpt-100.meta'))
    new_saver.restore(sess_re, tf.train.latest_checkpoint(os.path.join('model'
                                                                       , 'mynet_A')))        
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Run the initializer
    sess_re.run(init)
    # Restoring model and retraining with your own data 
    re_g = tf.get_default_graph()
    
    # op to write logs to Tensorboard
    train_writer = tf.summary.FileWriter(logdir="logs", graph=re_g, filename_suffix="mynet_A_re")
    
#     print ("scope","name", "shape","type")
#     for i in (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)):
#         print (i)

    for epoch in range(EPOCHS): # EPOCHS = 2
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch + 1, EPOCHS), unit='batches')

        for batch_i in batches_pbar:
            batch_start = batch_i * BATCH_SIZE
            batch_features = train['features'][batch_start:batch_start + BATCH_SIZE]
            batch_labels = train['labels'][batch_start:batch_start + BATCH_SIZE]

            _, l, summary = sess_re.run(
                [optimizer, loss, merged_summary_op],
                feed_dict={x: batch_features, y: batch_labels})
            # print (epoch * batch_count + batch_i)
            train_writer.add_summary(summary, epoch * BATCH_SIZE + batch_i)

        training_accuracy = sess_re.run(
            accuracy,
            feed_dict={x: batch_features, y: batch_labels}
        )

        idx = np.random.randint(len(valid['features']), size=int(BATCH_SIZE * .2))

        validation_accuracy = sess_re.run(
            accuracy,
            feed_dict={x: valid['features'][idx,:], y: valid['labels'][idx,:]}
        )

        print('Epoch {:>2}/{}'.format(epoch + 1, EPOCHS))
        print("loss is {}".format(l))
        print("training set accuracy is {}".format(training_accuracy))
        print("validation set accuracy is {}".format(validation_accuracy))


        batches.append(len(batches))
        loss_epoch.append(l)
        train_acc_epoch.append(training_accuracy)
        valid_acc_epoch.append(validation_accuracy)
    
    # Save model weights to disk
    # global_step refer to the number of batches seen by the graph.
    model_path = "model//{0}//valid_{1:.2f}.ckpt".format("mynet_A", validation_accuracy)
    save_path = saver.save(sess_re, model_path, global_step=100)
    print("Model saved in file: %s" % save_path)