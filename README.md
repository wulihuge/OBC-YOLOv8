# OBC-YOLOv8
# Here is the code you want!
Effective and efficient detection of pavement distress is very important for the normal use and maintenance of roads. To achieve this goal, a new road damage detection method based on YOLOv8 is proposed in this article. Firstly, omni-dimensional dynamic convolution (ODConv) block is employed to better grasp the complex and diverse features of damage objects by making dynamic adjustment according to the features of input images. Secondly, to extract the global and local feature information simultaneously to better improve the feature extraction ability of the model, BoTNet is added to the end of the backbone, which can combine the advantages of convolutional neural network (CNN) and Transformer. Finally, the coordinate attention mechanism (CA) is incorporated into the Neck section to make more accurate speculations and enhance detection accuracy further which can effectively mitigate irrelevant feature interference. The new proposed model is named OBC-YOLOv8 and the experimental results on the RDD2022-China dataset demonstrate its superiority compared with baselines, with 1.8% and 1.6% increases in mean average precision 50 (mAP@0.5) and F1-score, respectively.
Dataset Information：
The dataset is available at figshare: Arya, Deeksha; Maeda, Hiroya; Sekimoto, Yoshihide; Omata, Hiroshi; Ghosh, Sanjay Kumar; Toshniwal, Durga; et al. (2022). RDD2022-The multi-national Road Damage Dataset released through CRDDC'2022. figshare. Dataset. https://doi.org/10.6084/m9.figshare.21431547.v1.
Code Information:
The code is based on YOLOv8 and the three models of this paper is following:
ODConv：OBC-YOLOv8\ultralytics10.24\ultralytics\nn\ODConv.py
BotNet：OBC-YOLOv8\ultralytics10.24\ultralytics\nn\BoTNet.py
CoordAtt：OBC-YOLOv8\ultralytics10.24\ultralytics\nn\CA_Attention.py
Usage Instructions:
Find the Training file and modify the path of your own. Also, you can change the structure of model by modifying the Configuration file.
Configuration file: OBC-YOLOv8\ultralytics10.24\ultralytics\cfg\models\v8\yolov8-CA.yaml
Training file: OBC-YOLOv8\ultralytics10.24\train.py
Weight file for training: OBC-YOLOv8\ultralytics10.24\yolov8n.pt
Requirments：
OBC-YOLOv8\ultralytics10.24\requirements.txt
