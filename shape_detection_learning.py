

#------------------------------------------Single Rectangle dection-------------------------------------------#

## Adapted from https://github.com/jrieke/shape-detection by Jhonathan Pedroso Rigal dos Santos

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os

# Import tensorflow to use GPUs on keras:
import tensorflow as tf

# Set keras with GPUs
import keras
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 12} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# Import keras tools:
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Create images with random rectangles and bounding boxes:
num_imgs = 50000

img_size = 8
min_object_size = 1
max_object_size = 4
num_objects = 1

bboxes = np.zeros((num_imgs, num_objects, 4))
imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 

# Generating random images and bounding boxes:
for i_img in range(num_imgs):
    for i_object in range(num_objects):
       w, h = np.random.randint(min_object_size, max_object_size, size=2) # bbox width (w) and height (h)
       x = np.random.randint(0, img_size - w)			    			  # bbox x lower left corner coordinate 
       y = np.random.randint(0, img_size - h)							  # bbox y lower left corner coordinate
       imgs[i_img, x:x+w, y:y+h] = 1.  									  # set rectangle to 1
       bboxes[i_img, i_object] = [x, y, w, h]							  # store coordinates

# Lets plot one example of generated image:
i = 0
plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
for bbox in bboxes[i]:
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))

## Obs:
# - The transpose was done for using properly both plt functions
# - extent is the size of the image 
# - ec is the color of the border of the bounding box
# - fc is to avoid any coloured background of the bounding box

# Display plot:
# plt.show()

# Reshape (stack rows horizontally) and normalize the image data to mean 0 and std 1:
X = (imgs.reshape(num_imgs, -1) - np.mean(imgs)) / np.std(imgs)
X.shape, np.mean(X), np.std(X)

# Normalize x, y, w, h by img_size, so that all values are between 0 and 1:
# Important: Do not shift to negative values (e.g. by setting to mean 0)
#----------- because the IOU calculation needs positive w and h
y = bboxes.reshape(num_imgs, -1) / img_size
y.shape, np.mean(y), np.std(y)

# Split training and test:
i = int(0.8 * num_imgs)
train_X = X[:i]
test_X = X[i:]
train_y = y[:i]
test_y = y[i:]
test_imgs = imgs[i:]
test_bboxes = bboxes[i:]

# Build the model:
model = Sequential([Dense(200, input_dim=X.shape[-1]), 
			        Activation('relu'), 
			        Dropout(0.2), 
			        Dense(y.shape[-1])])

model.compile('adadelta', 'mse')

# Fit the model:
tic = time.time()
model.fit(train_X, train_y,nb_epoch=30, validation_data=(test_X, test_y), verbose=2)
toc = time.time() - tic
print(toc)

# Predict bounding boxes on the test images:
pred_y = model.predict(test_X)
pred_bboxes = pred_y * img_size
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
pred_bboxes.shape

# Function to define the intersection over the union of the bounding boxes pair:
def IOU(bbox1, bbox2):	
    '''Calculate overlap between two bounding boxes [x, y, w, h]
     		as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0
    else:
    	I = w_I * h_I
    	U = w1 * h1 + w2 * h2 - I
    return I / U

# Show a few images and predicted bounding boxes from the test dataset. 
os.chdir('/workdir/jp2476/repo/diversity-proj/files')
plt.figure(figsize=(12, 3))
for i_subplot in range(1, 5):
    plt.subplot(1, 4, i_subplot)
    i = np.random.randint(len(test_imgs))
    plt.imshow(test_imgs[i].T, cmap='Greys',
     						   interpolation='none', 
     						   origin='lower',
     						   extent=[0, img_size, 0, img_size])
    for pred_bbox, train_bbox in zip(pred_bboxes[i], test_bboxes[i]):
        plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0],
         												  pred_bbox[1]),
         												  pred_bbox[2],
         												  pred_bbox[3],
         												  ec='r', fc='none'))
        plt.annotate('IOU: {:.2f}'.format(IOU(pred_bbox, train_bbox)),
         									 (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.2),
         									  color='r')

# plt.savefig("simple_detection.pdf", dpi=150)
# plt.savefig("simple_detection.png", dpi=150)
plt.show()
plt.clf()

# Calculate the mean IOU (overlap) between the predicted and expected bounding boxes on the test dataset: 
summed_IOU = 0.
for pred_bbox, test_bbox in zip(pred_bboxes.reshape(-1, 4), test_bboxes.reshape(-1, 4)):
    summed_IOU += IOU(pred_bbox, test_bbox)

mean_IOU = summed_IOU / len(pred_bboxes)
mean_IOU


#-------------------------------------------Two Rectangle dection---------------------------------------------#

## Adapted from https://github.com/jrieke/shape-detection by Jhonathan Pedroso Rigal dos Santos

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os

# Import tensorflow to use GPUs on keras:
import tensorflow as tf

# Set keras with GPUs
import keras
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 12} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# Import keras tools:
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Create images with random rectangles and bounding boxes: 
num_imgs = 50000

# Image parameters for simulation:
img_size = 8
min_rect_size = 1
max_rect_size = 4
num_objects = 2

# Initialize objects:
bboxes = np.zeros((num_imgs, num_objects, 4))
imgs = np.zeros((num_imgs, img_size, img_size))

# Generate images and bounding boxes:
for i_img in range(num_imgs):
    for i_object in range(num_objects):
        w, h = np.random.randint(min_rect_size, max_rect_size, size=2)
        x = np.random.randint(0, img_size - w)
        y = np.random.randint(0, img_size - h)
        imgs[i_img, x:x+w, y:y+h] = 1.
        bboxes[i_img, i_object] = [x, y, w, h]

# Get shapes:
imgs.shape, bboxes.shape

# Plot one example of generated images:
i = 0
plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
for bbox in bboxes[i]:
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))

# plt.show()

# Reshape and normalize the data to mean 0 and std 1:
X = (imgs.reshape(num_imgs, -1) - np.mean(imgs)) / np.std(imgs)
X.shape, np.mean(X), np.std(X)

# Normalize x, y, w, h by img_size, so that all values are between 0 and 1:
# Important: Do not shift to negative values (e.g. by setting to mean 0),
#----------  because the IOU calculation needs positive w and h
y = bboxes.reshape(num_imgs, -1) / img_size
y.shape, np.mean(y), np.std(y)

# Function to define the intersection over the union of the bounding boxes pair:
def IOU(bbox1, bbox2):  
    '''Calculate overlap between two bounding boxes [x, y, w, h]
            as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0
    else:
        I = w_I * h_I
        U = w1 * h1 + w2 * h2 - I
    return I / U

# Split training and test.
i = int(0.8 * num_imgs)
train_X = X[:i]
test_X = X[i:]
train_y = y[:i]
test_y = y[i:]
test_imgs = imgs[i:]
test_bboxes = bboxes[i:]

# Build the model.
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
model = Sequential([
        Dense(256, input_dim=X.shape[-1]), 
        Activation('relu'), 
        Dropout(0.4), 
        Dense(y.shape[-1])
    ])
model.compile('adadelta', 'mse')

# Flip bboxes during training:
# Note: The validation loss is always quite big here because we don't flip the bounding boxes for 
#------ the validation data

# Define the distance between the two bounding boxes:
def distance(bbox1, bbox2):
    return np.sqrt(np.sum(np.square(bbox1[:2] - bbox2[:2])))

# Parameters to fit the model:
num_epochs = 50
flipped = np.zeros((len(train_y), num_epochs))
ious_epoch = np.zeros((len(train_y), num_epochs))
dists_epoch = np.zeros((len(train_y), num_epochs))
mses_epoch = np.zeros((len(train_y), num_epochs))

# Training the model:
for epoch in range(num_epochs):    
    # Print the current epoch:
    print('Epoch', epoch)
    # Fit the model:
    model.fit(train_X, train_y, nb_epoch=1, validation_data=(test_X, test_y), verbose=2)
    # Get the output from the neural net:
    hat_y = model.predict(train_X)
    for i, (hat_bboxes, train_bboxes) in enumerate(zip(hat_y, train_y)):
        # Flip the training data:   
        flipped_train_bboxes = np.concatenate([train_bboxes[4:], train_bboxes[:4]])
        # Compute the mean-squared error for non-flipped and flipped data points:
        mse = np.mean(np.square(hat_bboxes - train_bboxes))
        mse_flipped = np.mean(np.square(hat_bboxes - flipped_train_bboxes))
        # Compute the IOU for each variation:
        iou = IOU(hat_bboxes[:4], train_bboxes[:4]) + IOU(hat_bboxes[4:], train_bboxes[4:])
        iou_flipped = IOU(hat_bboxes[:4], flipped_train_bboxes[:4]) + IOU(hat_bboxes[4:], flipped_train_bboxes[4:])
        # Compute the distance for each variation:
        dist = distance(hat_bboxes[:4], train_bboxes[:4]) + distance(hat_bboxes[4:], train_bboxes[4:])
        dist_flipped = distance(hat_bboxes[:4], flipped_train_bboxes[:4]) + distance(hat_bboxes[4:], flipped_train_bboxes[4:])
        # Store stats:
        if mse_flipped < mse:  # you can also use iou or dist here
            train_y[i] = flipped_train_bboxes
            flipped[i, epoch] = 1
            mses_epoch[i, epoch] = mse_flipped
            ious_epoch[i, epoch] = iou_flipped / 2.
            dists_epoch[i, epoch] = dist_flipped / 2.
        else:
            mses_epoch[i, epoch] = mse
            ious_epoch[i, epoch] = iou / 2.
            dists_epoch[i, epoch] = dist / 2.
    print('Flipped {} training samples ({} %)'.format(np.sum(flipped[:, epoch]), np.mean(flipped[:, epoch]) * 100.))
    print('Mean IOU: {}'.format(np.mean(ious_epoch[:, epoch])))
    print('Mean dist: {}'.format(np.mean(dists_epoch[:, epoch])))
    print('Mean mse: {}'.format(np.mean(mses_epoch[:, epoch])))

# Show flippings for a few training samples:
plt.figure(figsize=(12, 12))
plt.pcolor(flipped[:300], cmap='Greys')
plt.xlabel('Epoch')
plt.ylabel('Training sample')
plt.show()

# Plot metrics on the training data:
mean_ious_epoch = np.mean(ious_epoch, axis=0)
mean_dists_epoch = np.mean(dists_epoch, axis=0)
mean_mses_epoch = np.mean(mses_epoch, axis=0)
plt.plot(mean_ious_epoch, label='Mean IoU')  # between predicted and assigned true bboxes
plt.plot(mean_dists_epoch, label='Mean distance')  # relative to image size
plt.plot(mean_mses_epoch, label='Mean MSE')
plt.annotate(np.round(np.max(mean_ious_epoch), 3), (len(mean_ious_epoch)-1, mean_ious_epoch[-1]+0.03),
             horizontalalignment='right',
             color='b')
plt.annotate(np.round(np.min(mean_dists_epoch), 3), (len(mean_dists_epoch)-1, mean_dists_epoch[-1]+0.03),
             horizontalalignment='right', color='g')
plt.annotate(np.round(np.min(mean_mses_epoch), 3), (len(mean_mses_epoch)-1, mean_mses_epoch[-1]+0.03),
             horizontalalignment='right', color='r')
plt.ylabel('IoU')
plt.xlabel('Epoch')
plt.legend()
plt.ylim(0, 1)
plt.show()

# Predict bounding boxes on the test images.
pred_y = model.predict(test_X)
pred_bboxes = pred_y * img_size
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
pred_bboxes.shape

# Show a few images and predicted bounding boxes from the test dataset:
plt.figure(figsize=(12, 3))
for i_subplot in range(1, 5):
    plt.subplot(1, 4, i_subplot)
    i = np.random.randint(len(test_X))
    plt.imshow(test_imgs[i].T, cmap='Greys',
                               interpolation='none',
                               origin='lower',
                               extent=[0, img_size, 0, img_size])
    for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i]):
        plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]),
                                                          pred_bbox[2], pred_bbox[3],
                                                         ec='r', fc='none'))

plt.show()

#--------------------------------Multiple rectangles or triangles---------------------------------------------#

## Adapted from https://github.com/jrieke/shape-detection by Jhonathan Pedroso Rigal dos Santos

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os

# Import tensorflow to use GPUs on keras:
import tensorflow as tf

# Set keras with GPUs
import keras
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 12} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

# Import keras tools:
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Create images with random rectangles and bounding boxes: 
num_imgs = 50000

# Image parameters for simulation:
img_size = 16
min_rect_size = 3
max_rect_size = 8
num_objects = 2

# Initialize objects:
bboxes = np.zeros((num_imgs, num_objects, 4))
imgs = np.zeros((num_imgs, img_size, img_size))
shapes = np.zeros((num_imgs, num_objects, 1))

# Generate images and bounding boxes:
for i_img in range(num_imgs):
    for i_object in range(num_objects):
        if np.random.choice([True, False]):
            width, height = np.random.randint(min_rect_size, max_rect_size, size=2)
            x = np.random.randint(0, img_size - width)
            y = np.random.randint(0, img_size - height)
            imgs[i_img, x:x+width, y:y+height] = 1.
            bboxes[i_img, i_object] = [x, y, width, height]
            shapes[i_img, i_object] = [0]
        else:
            size = np.random.randint(min_rect_size, max_rect_size)
            x, y = np.random.randint(0, img_size - size, size=2)
            mask = np.tril_indices(size)
            imgs[i_img, x + mask[0], y + mask[1]] = 1.
            bboxes[i_img, i_object] = [x, y, size, size]
            shapes[i_img, i_object] = [1]

# Get shapes:
imgs.shape, bboxes.shape

i = 1
# TODO: Why does the array have to be transposed?
plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
for bbox, shape in zip(bboxes[i], shapes[i]):
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                                     ec='r' if shape[0] == 0 else 'y',
                                                     fc='none'))

plt.show()

# Standalize the input volume:
X = (imgs.reshape(num_imgs, -1) - np.mean(imgs)) / np.std(imgs)
X.shape, np.mean(X), np.std(X)

# Rescale the target vector and add binary for the shape:
y = np.concatenate([bboxes / img_size, shapes], axis=-1).reshape(num_imgs, -1)
y.shape

# Set train and test sets:
i = int(0.8 * num_imgs)
train_X = X[:i]
test_X = X[i:]
train_y = y[:i]
test_y = y[i:]
test_imgs = imgs[i:]
test_bboxes = bboxes[i:]

# Compile the neural net:
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
model = Sequential([
        Dense(256, input_dim=X.shape[-1]), 
        Activation('relu'), 
        Dropout(0.4), 
        Dense(y.shape[-1])
    ])
model.compile('adadelta', 'mse')

# Flip bboxes during training:
# Note: The validation loss is always quite big here
#------ because we don't flip the bounding boxes for the validation data

# Function to compute the intersection over the union:
def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]  # TODO: Check if its more performant if tensor elements are accessed directly below.
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0
    I = w_I * h_I
    U = w1 * h1 + w2 * h2 - I
    return I / U

# Function to compute the distances between the boxes:
def dist(bbox1, bbox2):
    return np.sqrt(np.sum(np.square(bbox1[:2] - bbox2[:2])))

# Hyperparameters of the neural net and initialize objects:
num_epochs_flipping = 50
num_epochs_no_flipping = 0  # has no significant effect
flipped = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
ious_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
dists_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
mses_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))

# TODO: Calculate ious directly for all samples (using slices of the array hat_y for x, y, w, h).
for epoch in range(num_epochs_flipping):
    # Print current epoch:
    print('Epoch', epoch)
    # Fit the model:
    model.fit(train_X, train_y, nb_epoch=1, validation_data=(test_X, test_y), verbose=2)
    # Compute the output layer from the neural net:
    hat_y = model.predict(train_X)
    # Train the model:
    for sample, (hat, train) in enumerate(zip(hat_y, train_y)):      
        # TODO: Make this simpler.
        hat = hat.reshape(num_objects, -1)
        train = train.reshape(num_objects, -1)
        hat_bboxes = hat[:, :4]
        train_bboxes = train[:, :4]
        # TODO: Try flipping array and see if results differ.
        ious = np.zeros((num_objects, num_objects))
        dists = np.zeros((num_objects, num_objects))
        mses = np.zeros((num_objects, num_objects))
        for i, train_bbox in enumerate(train_bboxes):
            for j, hat_bbox in enumerate(hat_bboxes):
                ious[i, j] = IOU(train_bbox, hat_bbox)
                dists[i, j] = dist(train_bbox, hat_bbox)
                mses[i, j] = np.mean(np.square(train_bbox - hat_bbox))
        new_order = np.zeros(num_objects, dtype=int)
        for i in range(num_objects):
            # Find hat and train bbox with maximum iou and assign them to each other (i.e. switch the positions of the train bboxes in y).
            ind_train_bbox, ind_hat_bbox = np.unravel_index(mses.argmin(), mses.shape)
            ious_epoch[sample, epoch] += ious[ind_train_bbox, ind_hat_bbox]
            dists_epoch[sample, epoch] += dists[ind_train_bbox, ind_hat_bbox]
            mses_epoch[sample, epoch] += mses[ind_train_bbox, ind_hat_bbox]
            mses[ind_train_bbox] = 1000000#-1  # set iou of assigned bboxes to -1, so they don't get assigned again
            mses[:, ind_hat_bbox] = 10000000#-1
            new_order[ind_hat_bbox] = ind_train_bbox
        # Flatten the object:
        train_y[sample] = train[new_order].flatten()
        # Store outputs:
        flipped[sample, epoch] = 1. - np.mean(new_order == np.arange(num_objects, dtype=int))#np.array_equal(new_order, np.arange(num_objects, dtype=int))  # TODO: Change this to reflect the number of flips.
        ious_epoch[sample, epoch] /= num_objects
        dists_epoch[sample, epoch] /= num_objects
        mses_epoch[sample, epoch] /= num_objects
    # Print results:        
    print('Flipped {} % of all elements'.format(np.mean(flipped[:, epoch]) * 100.))
    print('Mean IOU: {}'.format(np.mean(ious_epoch[:, epoch])))
    print('Mean dist: {}'.format(np.mean(dists_epoch[:, epoch])))
    print('Mean mse: {}'.format(np.mean(mses_epoch[:, epoch])))


# Show flippings for a few training samples:
plt.figure(figsize=(12, 12))
plt.pcolor(flipped[:1000], cmap='Greys', vmax=1.)
# plt.axvline(num_epochs_flipping, c='r')
plt.xlabel('Epoch')
plt.ylabel('Training sample')
plt.show()

# Plot metrics on the training data:
mean_ious_epoch = np.mean(ious_epoch, axis=0)
mean_dists_epoch = np.mean(dists_epoch, axis=0)
mean_mses_epoch = np.mean(mses_epoch, axis=0)
plt.plot(mean_ious_epoch, label='Mean IoU')  # between predicted and assigned true bboxes
plt.plot(mean_dists_epoch, label='Mean distance')  # relative to image size
plt.plot(mean_mses_epoch, label='Mean MSE')
plt.annotate(np.round(np.max(mean_ious_epoch), 3), (len(mean_ious_epoch)-1, mean_ious_epoch[-1]+0.03),
             horizontalalignment='right',
             color='b')
plt.annotate(np.round(np.min(mean_dists_epoch), 3), (len(mean_dists_epoch)-1, mean_dists_epoch[-1]+0.03),
             horizontalalignment='right', color='g')
plt.annotate(np.round(np.min(mean_mses_epoch), 3), (len(mean_mses_epoch)-1, mean_mses_epoch[-1]+0.03),
             horizontalalignment='right', color='r')
plt.ylabel('IoU')
plt.xlabel('Epoch')
plt.legend()
plt.ylim(0, 1)
plt.show()

# Predict bounding boxes on the test images.
pred_y = model.predict(test_X)
pred_y = pred_y.reshape(len(pred_y), num_objects, -1)
pred_bboxes = pred_y[..., :4] * img_size
pred_shapes = pred_y[..., 4:5]
pred_bboxes.shape, pred_shapes.shape

# Show a few images and predicted bounding boxes from the test dataset:
plt.figure(figsize=(16, 8))
for i_subplot in range(1, 9):
    plt.subplot(2, 4, i_subplot)
    i = np.random.randint(len(test_X))
    plt.imshow(test_imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    for pred_bbox, exp_bbox, pred_shape in zip(pred_bboxes[i], test_bboxes[i], pred_shapes[i]):
        plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r' if pred_shape[0] <= 0.5 else 'y', fc='none'))
        plt.annotate('IOU: {:.2f}'.format(IOU(pred_bbox, exp_bbox)), (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.4), color='r')

plt.show()
