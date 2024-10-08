import pandas as pd
import matplotlib.pyplot as plt
 
# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\s+', '_', regex=True)
 

original_results = pd.read_csv("/home/lzh/ultralytics10.24/runs/detect/train52/results.csv")

improved_results = pd.read_csv("/home/lzh/ultralytics10.24/runs/detect/train64/results.csv")
 
# Clean column names
clean_column_names(original_results)
clean_column_names(improved_results)
 
# Plot mAP@0.5 curves
plt.figure()

plt.plot(original_results['metrics/mAP50(B)'], label="YOLOv8n")
plt.plot(improved_results['metrics/mAP50(B)'], '-.', label="OBC-YOLOv8")
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5")
plt.legend()
plt.title("mAP@0.5 Comparison")
plt.savefig("mAP_0.5_comparison_02.png")
 
# Plot mAP@0.5:0.95 curves
plt.figure()
plt.plot(original_results['metrics/mAP50-95(B)'], label="YOLOv8n")
plt.plot(improved_results['metrics/mAP50-95(B)'], '-.', label="OBC-YOLOv8")
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5:0.95")
plt.legend()

plt.title("mAP@0.5:0.95 Comparison")

plt.savefig("mAP_0.5_0.95_comparison_02.png")
 
