from ultralytics import YOLO

# Load a pretrained YOLO26 segment model
model = YOLO("yolo26l-seg.pt")

# Train the model
results = model.train(
    data="/pasteur/appa/homes/ctuna/dataset.yaml",
    epochs=200,
    imgsz=1600,
    multi_scale=0.5,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.001,
    batch=4,
)    
