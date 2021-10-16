# load necessary libraries
import cv2
import numpy as np


# define path to model
path_to_model = 'C:/Users/user/Downloads/instance_segmentation/frozen_inference_graph_coco.pb'
path_to_config = 'C:/Users/user/Downloads/instance_segmentation/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
image_path = 'G:/rauf/STEPBYSTEP/Projects2/vision/Instance Segmentation/Maks RCNN/dog.jpg'
colors = np.random.randint(125, 255, (80, 3))

# define the model parameters
model = cv2.dnn.readNetFromTensorflow(path_to_model,path_to_config)
img = cv2.imread(image_path)
img = cv2.resize(img, (650, 650))
height, width, _ = img.shape
black_image = np.zeros((height, width, 3), np.uint8)
black_image[:] = (0, 0, 0)


# preprocessing and detecting object
blob = cv2.dnn.blobFromImage(img, swapRB=True)
model.setInput(blob)
boxes, masks = model.forward(['detection_out_final', 'detection_masks'])
no_of_objects = boxes.shape[2]

# fetching object
for i in range(no_of_objects):
    box = boxes[0, 0, i]
    class_id = box[1]
    score = box[2]
    if score < 0.5:
        continue

    # getting cordinates
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)

    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)


    # build masks
    roi = black_image[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape
    mask = masks[i, int(class_id)]
    mask = cv2.resize(mask, (roi_width, roi_height))
    _, maks = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)


    # find contorus and draw them
    contorus, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = colors[int(class_id)]
    for cnt in contorus:
        cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))

# visualize the result
cv2.imshow("Final", np.hstack([img, black_image]))
overlay_frame = ((0.6*black_image)+(0.4*img)).astype("uint8")
cv2.imshow("Overlay frame: ", overlay_frame)
cv2.waitKey(0)