from ultralytics import YOLO
 
 
model = YOLO('/home/lzh/ultralytics10.24/runs/detect/train52/weights/best.pt')

results = model('/home/lzh/ultralytics10.24/ultralytics/assets', save=True)