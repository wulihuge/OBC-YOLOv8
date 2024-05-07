
from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train40/weights/best.pt")  # load a pretrained model (recommended for training)

# Use the model

metrics = model.val(data="pavement.yaml")  # evaluate model performance on the validation set
