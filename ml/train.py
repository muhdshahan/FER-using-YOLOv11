from ultralytics import YOLO

# Setup
project_root = 'FER-using-YOLOv11'
DATASET_ROOT_DIR = '../dataset'

# Training Parameters
EPOCHS = 50
IMAGE_SIZE = 48   # setting all images to 48x48 pixel size as it is the pre-registered resolution of FER-2013

MODEL_NAME = "yolo11n-cls.pt"
RUN_NAME = 'facial_emotion_fer2013_v1'  # The current run name

# 1. Load the pre-trained YOLOv11 Classification model
print(f"Loading model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)

# 2. Start Classification Training
print("\n--- Starting YOLOv11 Classification Fine-Tuning ---")
# Pass the path to data.yaml file to 'data' argument.
# Let's explicitly set task to 'classify' even though the model is classification-specific.
results = model.train(
    data=DATASET_ROOT_DIR,
    task='classify',
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    name=RUN_NAME,
    batch=128     # Increased batch size, its good for classification on small images
)

print("\n--- Training Complete ---")
print(f"Final model weights saved to runs/classify/{RUN_NAME}/weights/best.pt")