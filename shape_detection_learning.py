

#------------------------------------------Single Rectangle dection-------------------------------------------#

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
#  , because the IOU calculation needs positive w and h
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
model.fit(train_X, train_y, nb_epoch=30, validation_data=(test_X, test_y), verbose=2)

# Predict bounding boxes on the test images:
pred_y = model.predict(test_X)
pred_bboxes = pred_y * img_size
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
pred_bboxes.shape

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
plt.figure(figsize=(12, 3))
for i_subplot in range(1, 5):
    plt.subplot(1, 4, i_subplot)
    i = np.random.randint(len(test_imgs))
    plt.imshow(test_imgs[i].T, cmap='Greys',
     						   interpolation='none', 
     						   origin='lower',
     						   extent=[0, img_size, 0, img_size])
    for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i]):
        plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0],
         												  pred_bbox[1]),
         												  pred_bbox[2],
         												  pred_bbox[3],
         												  ec='r', fc='none'))
        plt.annotate('IOU: {:.2f}'.format(IOU(pred_bbox, exp_bbox)),
         									 (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.2),
         									  color='r')



