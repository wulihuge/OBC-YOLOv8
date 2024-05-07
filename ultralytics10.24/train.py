from ultralytics import YOLO

# 忽略一些警告
import warnings    
warnings.filterwarnings('ignore')


model = YOLO("ultralytics/cfg/models/v8/yolov8-CA.yaml")  # build a new model from scratch

 
# Train the model
results = model.train(data="RDDChina.yaml", epochs=400, imgsz=640)


