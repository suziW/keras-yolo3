"""
Retrain the YOLO model for your own dataset.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda, add, multiply, dot, Average
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import multi_gpu_model

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/single_gpu/'
    classes_path = 'model_data/yolo_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    # weights_path='model_data/yolo_weights.h5'
    weights_path='logs/single_gpu/trained_weights_stage_1.h5'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    print('<><><><><><><><><><><><><> anchors, num_classes, class_names', anchors, num_classes, class_names)
    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=weights_path) # make sure you know what you freeze

    model.summary()
    print('detail', model.inputs, model.outputs, len(model.layers))
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 8 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]  # [(?, 13, 13, 3, 85), (?, 26, 26, 3, 85), (?, 52, 52, 3, 85)]
    print('<><><><><><><><><><><><><><><><>y_true', y_true)   

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('??????', model_body.input, model_body.output, len(model_body.layers)) # input: (?, ?, ?， 3), output: [(?, ?, ?, 255), (?, ?, ?, 255), (?, ?, ?, 255)], total 252 layers 
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))  # path: model_data/yolo_weights.h5
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]    # (185, 252)[freze] , 185 refers darknet, 252 refers model_body
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    # model_body = multi_gpu_model(model_body, gpus=2) # modified
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    print('<><><><><><><><><><><><><><><><><>model.input, model.output', model.input, model.output)
    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)

            image, box = get_random_data(annotation_lines[i], input_shape, random=False)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
            # break
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # break
        yield [image_data, *y_true], np.zeros(batch_size)

# def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes): 
#     '''data generator for fit_generator'''
#     n = len(annotation_lines)
#     i = 0 
#     while True: 
#         image_data = [] 
#         box_data = [] 
#         if (i+batch_size > n): np.random.shuffle(annotation_lines)
#         i = 0 
#         output = threadpool.starmap(get_random_data, zip(annotation_lines[i:i+batch_size], itertools.repeat(input_shape, batch_size))) 
#         image_data = list(zip(*output))[0] 
#         box_data = list(zip(*output))[1] 
#         i = i+batch_size 
#         image_data = np.array(image_data) 
#         box_data = np.array(box_data) 
#         y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes) 
#         yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def _get_available_devices():
    return [x.name for x in K.get_session().list_devices()]


def _normalize_device_name(name):
    name = '/' + ':'.join(name.lower().replace('/', '').split(':')[-2:])
    return name

# def multi_gpu_model(model, gpus=None):
#     if K.backend() != 'tensorflow':
#         raise ValueError('`multi_gpu_model` is only available '
#                          'with the TensorFlow backend.')

#     available_devices = _get_available_devices()
#     available_devices = [_normalize_device_name(name) for name in available_devices]
#     if not gpus:
#         # Using all visible GPUs when not specifying `gpus`
#         # e.g. CUDA_VISIBLE_DEVICES=0,2 python3 keras_mgpu.py
#         gpus = len([x for x in available_devices if 'gpu' in x])

#     if isinstance(gpus, (list, tuple)):
#         if len(gpus) <= 1:
#             raise ValueError('For multi-gpu usage to be effective, '
#                              'call `multi_gpu_model` with `len(gpus) >= 2`. '
#                              'Received: `gpus=%s`' % gpus)
#         num_gpus = len(gpus)
#         target_gpu_ids = gpus
#     else:
#         if gpus <= 1:
#             raise ValueError('For multi-gpu usage to be effective, '
#                              'call `multi_gpu_model` with `gpus >= 2`. '
#                              'Received: `gpus=%d`' % gpus)
#         num_gpus = gpus
#         target_gpu_ids = range(num_gpus)

#     import tensorflow as tf

#     target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in target_gpu_ids]
#     for device in target_devices:
#         if device not in available_devices:
#             raise ValueError(
#                 'To call `multi_gpu_model` with `gpus=%d`, '
#                 'we expect the following devices to be available: %s. '
#                 'However this machine only has: %s. '
#                 'Try reducing `gpus`.' % (gpus,
#                                           target_devices,
#                                           available_devices))

#     def get_slice(data, i, parts):
#         shape = tf.shape(data)
#         batch_size = shape[:1]
#         input_shape = shape[1:]
#         step = batch_size // parts
#         if i == num_gpus - 1:
#             size = batch_size - step * i
#         else:
#             size = step
#         size = tf.concat([size, input_shape], axis=0)
#         stride = tf.concat([step, input_shape * 0], axis=0)
#         start = stride * i
#         return tf.slice(data, start, size)

#     all_outputs = []
#     for i in range(len(model.outputs)):
#         all_outputs.append([])

#     # Place a copy of the model on each GPU,
#     # each getting a slice of the inputs.
#     for i, gpu_id in enumerate(target_gpu_ids):
#         with tf.device('/gpu:%d' % gpu_id):
#             with tf.name_scope('replica_%d' % gpu_id):
#                 inputs = []
#                 # Retrieve a slice of the input.
#                 for x in model.inputs:
#                     input_shape = tuple(x.get_shape().as_list())[1:]
#                     slice_i = Lambda(get_slice,
#                                      output_shape=input_shape,
#                                      arguments={'i': i,
#                                                 'parts': num_gpus})(x)
#                     inputs.append(slice_i)

#                 # Apply model on slice
#                 # (creating a model replica on the target device).
#                 outputs = model(inputs)
#                 if not isinstance(outputs, list):
#                     outputs = [outputs]

#                 # Save the outputs for merging back together later.
#                 for o in range(len(outputs)):
#                     all_outputs[o].append(outputs[o])

#     # Merge outputs on CPU.
#     with tf.device('/cpu:0'):
#         merged = []
#         print(model.output_names)
#         print(zip(model.output_names, all_outputs))
#         for name, outputs in zip(model.output_names, all_outputs):
#             merged.append(Average(name=name)(outputs))
#             # merged.append(add(outputs, name=name))
#         return Model(model.inputs, merged)

if __name__ == '__main__':
    _main()
    