# How the current model stored in `model.json` and `model.h5` generated.

## Training data
See the project README.MD for a description of these datasets.
```
train_data = [
          ("data/t1r1/driving_log.csv", "data/t1r1/IMG")
        , ("data/t1r2/driving_log.csv", "data/t1r2/IMG/")
        , ("data/t1r3/driving_log.csv", "data/t1r3/IMG/")
        , ("data/t1r4/driving_log.csv", "data/t1r4/IMG/")
        , ("data/t1r5/driving_log.csv", "data/t1r5/IMG/")

        , ("data/t1b1/driving_log.csv", "data/t1b1/IMG/")

        , ("data/t1w1/driving_log.csv", "data/t1w1/IMG/") 

        , ("data/t1rr1/driving_log.csv", "data/t1rr1/IMG/")

        , ("data/t2r1/driving_log.csv", "data/t2r1/IMG/")
        , ("data/t2r2/driving_log.csv", "data/t2r2/IMG/")
        , ("data/t2r3/driving_log.csv", "data/t2r3/IMG/")
        , ("data/t2r3.1/driving_log.csv", "data/t2r3.1/IMG/")
        , ("data/t2r4/driving_log.csv", "data/t2r4/IMG/")
        , ("data/t2w1/driving_log.csv", "data/t2w1/IMG/")
]
```

## Model
It is a VGG16 based model based on the following configurations in `config.py`
```
model_name = "vgg16_pretrained"
image_size = (80, 80, 3)
processors = {"CenterImage": process.vgg_processor(image_size)}
```

## Training
Training runs for 6 epoches on a GTX 980M, with the following logs
```
python -m sdc.train_regression --train --nb_epoch 6 | tee train.log 
Using TensorFlow backend.
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so locally
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:925] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 0 with properties: 
name: GeForce GTX 980M
major: 5 minor: 2 memoryClockRate (GHz) 1.1265
pciBusID 0000:01:00.0
Total memory: 3.99GiB
Free memory: 3.59GiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:126] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_init.cc:136] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:838] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 980M, pci bus id: 0000:01:00.0)
W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 2.03GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 3.52GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 2.54GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 2.13GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 2.13GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 2.13GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 2.13GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 2.13GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 2.13GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
4026 records have been removed due to speed <= 20
training size 166734, validation size 10000, test size 10000
building vgg16_pretrained model
inspect model:
input_1 (None, 80, 80, 3) (None, 80, 80, 3) False
block1_conv1 (None, 80, 80, 3) (None, 80, 80, 64) False
block1_conv2 (None, 80, 80, 64) (None, 80, 80, 64) False
block1_pool (None, 80, 80, 64) (None, 40, 40, 64) False
block2_conv1 (None, 40, 40, 64) (None, 40, 40, 128) False
block2_conv2 (None, 40, 40, 128) (None, 40, 40, 128) False
block2_pool (None, 40, 40, 128) (None, 20, 20, 128) False
block3_conv1 (None, 20, 20, 128) (None, 20, 20, 256) False
block3_conv2 (None, 20, 20, 256) (None, 20, 20, 256) False
block3_conv3 (None, 20, 20, 256) (None, 20, 20, 256) False
block3_pool (None, 20, 20, 256) (None, 10, 10, 256) False
block4_conv1 (None, 10, 10, 256) (None, 10, 10, 512) False
block4_conv2 (None, 10, 10, 512) (None, 10, 10, 512) False
block4_conv3 (None, 10, 10, 512) (None, 10, 10, 512) False
block4_pool (None, 10, 10, 512) (None, 5, 5, 512) False
block5_conv1 (None, 5, 5, 512) (None, 5, 5, 512) False
block5_conv2 (None, 5, 5, 512) (None, 5, 5, 512) True
block5_conv3 (None, 5, 5, 512) (None, 5, 5, 512) True
averagepooling2d_1 (None, 5, 5, 512) (None, 2, 2, 512) True
dropout_1 (None, 2, 2, 512) (None, 2, 2, 512) True
batchnormalization_1 (None, 2, 2, 512) (None, 2, 2, 512) True
dropout_2 (None, 2, 2, 512) (None, 2, 2, 512) True
flatten_1 (None, 2, 2, 512) (None, 2048) True
dense_1 (None, 2048) (None, 4096) True
dropout_3 (None, 4096) (None, 4096) True
dense_2 (None, 4096) (None, 2048) True
dense_3 (None, 2048) (None, 2048) True
dense_4 (None, 2048) (None, 1) True
Epoch 1/6
 60416/166656 [=========>....................] - ETA: 284s - loss: 69.2436W tensorflow/core/common_runtime/bfc_allocator.cc:213] Ran out of memory trying to allocate 1.06GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
166656/166656 [==============================] - 465s - loss: 44.6383 - val_loss: 0.0587
Epoch 2/6
166656/166656 [==============================] - 457s - loss: 8.7741 - val_loss: 0.0391
Epoch 3/6
166656/166656 [==============================] - 457s - loss: 1.6978 - val_loss: 0.0363
Epoch 4/6
166656/166656 [==============================] - 456s - loss: 0.3307 - val_loss: 0.0361
Epoch 5/6
166656/166656 [==============================] - 455s - loss: 0.0848 - val_loss: 0.0344
Epoch 6/6
166656/166656 [==============================] - 455s - loss: 0.0448 - val_loss: 0.0335
save model to models/model
evaluate on test data
test mse: 0.03336181522120209
save test result for inspection
```

