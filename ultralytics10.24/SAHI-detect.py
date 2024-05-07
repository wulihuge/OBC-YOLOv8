from ultralytics import YOLO
from sahi.utils.yolov8 import (
    download_yolov8s_model,
)
 
# Import required functions and classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.prediction import visualize_object_predictions
from IPython.display import Image
from numpy import asarray
import cv2



from ultralytics.utils.plotting import Annotator, colors, save_one_box



model = YOLO('yolov8n.pt')

model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = [0, 1, 5]  # perform detection on only several classes

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model=model)




result = get_sliced_prediction(
    "ultralytics/data/sahi/0000364_01373_d_0000780.jpg",
    detection_model,
    slice_height = 100,
    slice_width = 100,
    overlap_height_ratio = 0.05,
    overlap_width_ratio = 0.05
)



img = cv2.imread("ultralytics/data/sahi/0000364_01373_d_0000780.jpg", cv2.IMREAD_UNCHANGED)
img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
numpydata = asarray(img_converted)
visualize_object_predictions(
    numpydata, 
    object_prediction_list = result.object_prediction_list,
    hide_labels = None, 
    output_dir='runs/detect/SAHI-detect',

    file_name = 'result',
    export_format = 'jpg'
)
Image('demo_data/result.png')

