import os
import torch
from ultralytics import YOLO

# ABSOLUTE PFADE
PATH_DATASET = "/Users/lukas/Documents/GitHub/BrickByBrick/Woodblock-Dataset-Yolov11/data.yaml"
OUTPUT_DIR = "/Users/lukas/Documents/GitHub/BrickByBrick/Yolov11_640_2"


EXPERIMENTS = [
    # {"model": "yolo11n-seg.pt", "name": "Nano_Optimized", "batch": 16, "opt": True},
    # {"model": "yolo11s-seg.pt", "name": "Small_Optimized", "batch": 8,  "opt": True},
    {"model": "yolo11m-seg.pt", "name": "Medium_Standard", "batch": 4,  "opt": False},
    {"model": "yolo11m-seg.pt", "name": "Medium_Standard", "batch": 4,  "opt": True}
]

for exp in EXPERIMENTS:
    print(f"\nSTARTE JETZT: {exp['name']}")
    
    # Modell frisch laden
    model = YOLO(exp['model'])
    
    # Parameter-Logik (Optimized bekommt längere Close-Mosaic Phase)
    close_val = 20 if exp['opt'] else 10
    
    # Training (Native YOLO-Aufruf ohne Schnickschnack)
    model.train(
        data=PATH_DATASET,
        epochs=80,           # Auf 80 reduziert, damit beide bis morgen früh fertig werden
        imgsz=640,
        batch=exp['batch'],
        device='mps',
        project=OUTPUT_DIR,
        name=exp['name'],
        close_mosaic=close_val,
        mask_ratio=1,        
        workers=0,           # Verhindert MPS-Deadlocks und spart RAM
        cache=False,         # Schont den RAM massiv
        exist_ok=True,       # Falls Ordner existiert, überschreiben/fortsetzen
        plots=True           # YOLO erstellt results.png AUTOMATISCH
    )
    
    # Speicher-Cleanup nach jedem Run
    del model
    torch.mps.empty_cache()
    print(f"✅ FERTIG MIT: {exp['name']}")

print("\n ALLE MODELLE FERTIG. SCHAU IN DEN ORDNER 'FINAL_MODELS'.")