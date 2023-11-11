import sys
import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
from model_continue_train import ResNet50

if len(sys.argv) < 2:
    print("Image name missing; \n usage: python predict.py <image name>")
    sys.exit()
img_name = sys.argv[1]


img = cv2.imread(img_name)
img_float = cv2.resize(img, (224,224)).astype(np.float32)
img_float -= 128
img_in = np.expand_dims(img_float, axis=0)
model = ResNet50(include_top=False, load_weight=True, weights='models/run1_0.01_weights.14-5.09.hdf5',
                input_shape=(224,224,3))
pred = model.predict(img_in)

bboxes = utils.get_boxes(pred[0], cutoff=0.1)
bboxes = utils.nonmax_suppression(bboxes, iou_cutoff = 0.05)
draw = utils.draw_boxes(img, bboxes, color=(0, 0, 255), thick=3, draw_dot=True, radius=3)
draw = draw.astype(np.uint8)

plt.imshow(draw[...,::-1])
plt.show()

# import sys
# import numpy as np
# import cv2
# import utils
# import matplotlib.pyplot as plt
# from model_continue_train import ResNet50

# if len(sys.argv) < 2:
#     print("Image name missing; \n usage: python predict.py <image name>")
#     sys.exit()

# img_name = sys.argv[1]

# img = cv2.imread(img_name)
# img_float = cv2.resize(img, (224, 224)).astype(np.float32)
# img_float -= 128

# img_in = np.expand_dims(img_float, axis=0)

# model = ResNet50(include_top=False, load_weight=True, weights='models/rerun9_0.01_weights.02-2.05.hdf5',
#                 input_shape=(224, 224, 3))
# pred = model.predict(img_in)

# # Assuming class index 0 is for weapons
# class_index = 0

# # Extract relevant information from predictions
# bboxes = utils.get_boxes(pred[0], cutoff=0.1)

# # Filter detections only for the weapon class (class_index)
# filtered_bboxes = [box for box in bboxes if box[2] == class_index]

# # Apply confidence threshold
# confidence_threshold = 0.1
# filtered_bboxes = [box for box in filtered_bboxes if box[2] >= confidence_threshold]

# # Draw bounding boxes on the image
# draw = img.copy()
# for box in filtered_bboxes:
#     (x1, y1), (x2, y2), _ = box
#     draw = utils.draw_boxes(draw, [(x1, y1, x2, y2)], color=(0, 0, 255), thick=3, draw_dot=True, radius=3)

# plt.imshow(draw[...,::-1])
# plt.show()
