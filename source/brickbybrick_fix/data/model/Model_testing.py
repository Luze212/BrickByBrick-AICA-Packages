import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

# ==========================================
# KONFIGURATION
# ==========================================
MODELS_ROOT = "./Models"
DATASET_TEST_IMAGES = "./Woodblock-Dataset-Yolov11/test/images"
DATASET_TEST_LABELS = "./Woodblock-Dataset-Yolov11/test/labels"
LOG_BASE_DIR = "./Logs"

CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
IMG_SIZE = 640

# ==========================================
# HILFSFUNKTIONEN
# ==========================================

def get_ground_truth_boxes(label_path, img_w, img_h):
    """
    Liest Label-Datei.
    Unterstützt YOLO SEGMENTATION Format (Polygone) und wandelt sie in Bounding-Boxen um.
    """
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5: continue
                
                # Alles nach der Klasse sind Koordinaten
                coords = parts[1:] 
                
                # SEGMENTATION CHECK (Polygon > 4 Werte)
                if len(coords) > 4:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    x1, y1 = int(x_min * img_w), int(y_min * img_h)
                    x2, y2 = int(x_max * img_w), int(y_max * img_h)
                else:
                    # KLASSISCHE BOX (5 Werte)
                    xc, yc, w, h = coords[0], coords[1], coords[2], coords[3]
                    x1 = int((xc - w / 2) * img_w)
                    y1 = int((yc - h / 2) * img_h)
                    x2 = int((xc + w / 2) * img_w)
                    y2 = int((yc + h / 2) * img_h)
                    
                boxes.append([x1, y1, x2, y2])
    return boxes

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ==========================================
# HAUPTPROGRAMM
# ==========================================

def run_benchmark():
    os.makedirs(LOG_BASE_DIR, exist_ok=True)
    
    model_files = glob.glob(os.path.join(MODELS_ROOT, "**", "*.pt"), recursive=True)
    
    if not model_files:
        print(f" Keine Modelle in {MODELS_ROOT} gefunden!")
        return

    print(f" Starte Vollen Benchmark für {len(model_files)} Modelle...")
    all_models_summary = []

    for model_path in model_files:
        # Namen generieren
        parent_dir = os.path.dirname(os.path.dirname(model_path))
        if os.path.basename(os.path.dirname(model_path)) != "weights":
             parent_dir = os.path.dirname(model_path)
        subfolder_name = os.path.basename(parent_dir)
        filename = os.path.basename(model_path).replace(".pt", "")
        run_name = f"{subfolder_name}_{filename}"
        
        print(f"\n Bearbeite: {run_name}")

        current_log_dir = os.path.join(LOG_BASE_DIR, run_name)
        results_img_dir = os.path.join(current_log_dir, "Ergebnisse")
        os.makedirs(results_img_dir, exist_ok=True)

        try:
            model = YOLO(model_path)
        except Exception as e:
            continue

        test_images = glob.glob(os.path.join(DATASET_TEST_IMAGES, "*.*"))
        test_images = [f for f in test_images if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        stats = {
            "Model": run_name, "Images_Count": 0, "Misses": 0, "Ghosts": 0,
            "Avg_IoU": [], "Avg_Conf": [], "Time_Inf": [], "Time_Total": []
        }

        for img_path in tqdm(test_images, desc=f"Testing {run_name}"):
            stats["Images_Count"] += 1
            img_name = os.path.basename(img_path)
            label_path = os.path.join(DATASET_TEST_LABELS, os.path.splitext(img_name)[0] + ".txt")

            img_origin = cv2.imread(img_path)
            if img_origin is None: continue
            h, w, _ = img_origin.shape
            
            # Predict
            results = model.predict(img_path, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)[0]
            
            # Zeiten
            t_inf = results.speed['inference']
            t_tot = results.speed['preprocess'] + results.speed['inference'] + results.speed['postprocess']
            stats["Time_Inf"].append(t_inf)
            stats["Time_Total"].append(t_tot)

            # Ground Truth & Matching
            gt_boxes = get_ground_truth_boxes(label_path, w, h)
            pred_boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            current_misses = 0
            current_ghosts = 0
            image_ious = []

            # Misses check
            for gt in gt_boxes:
                matched = False
                for pred in pred_boxes:
                    if calculate_iou(gt, pred) > IOU_THRESHOLD:
                        matched = True
                        image_ious.append(calculate_iou(gt, pred))
                        break
                if not matched:
                    current_misses += 1
            
            # Ghosts check
            for pred in pred_boxes:
                matched = False
                for gt in gt_boxes:
                    if calculate_iou(gt, pred) > IOU_THRESHOLD:
                        matched = True
                        break
                if not matched:
                    current_ghosts += 1

            stats["Misses"] += current_misses
            stats["Ghosts"] += current_ghosts
            if len(confs) > 0: stats["Avg_Conf"].extend(confs)
            stats["Avg_IoU"].extend(image_ious)

            # Visualisierung
            img_left = img_origin.copy()
            if not gt_boxes:
                cv2.putText(img_left, "KEIN LABEL", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                for box in gt_boxes:
                    cv2.rectangle(img_left, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            img_right = results.plot(labels=False, conf=False)
            avg_iou_val = np.mean(image_ious) if image_ious else 0.0
            
            cv2.putText(img_right, f"Conf: {np.mean(confs):.2f}" if confs.size > 0 else "Nichts", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_right, f"IoU: {avg_iou_val:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if current_misses > 0:
                 cv2.putText(img_right, "MISS!", (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            combined = cv2.hconcat([img_left, img_right])
            cv2.imwrite(os.path.join(results_img_dir, f"res_{img_name}"), combined)

        # Statistik
        avg_iou = np.mean(stats["Avg_IoU"]) if stats["Avg_IoU"] else 0
        avg_conf = np.mean(stats["Avg_Conf"]) if stats["Avg_Conf"] else 0
        avg_time = np.mean(stats["Time_Inf"])

        model_summary = {
            "Model_Name": run_name,
            "Images_Tested": stats["Images_Count"],
            "Misses (False Neg)": stats["Misses"],
            "Ghosts (False Pos)": stats["Ghosts"],
            "Avg_Accuracy_IoU": round(avg_iou, 4),
            "Avg_Confidence": round(avg_conf, 4),
            "Avg_Inference_ms": round(avg_time, 2),
            "Total_Benchmark_Time_s": round(np.sum(stats["Time_Total"]) / 1000, 2)
        }
        all_models_summary.append(model_summary)
        
        with open(os.path.join(current_log_dir, f"Stats_{run_name}.txt"), "w") as f:
            for k, v in model_summary.items(): f.write(f"{k}: {v}\n")

    # --- GRAPHEN GENERIERUNG (WICHTIG!) ---
    if all_models_summary:
        print("\n Erstelle ALLE Graphen und Master-Log...")
        df = pd.DataFrame(all_models_summary)
        df.to_csv(os.path.join(LOG_BASE_DIR, "Benchmark_Master_Summary.csv"), index=False)

        # 1. Graph: Genauigkeit (IoU)
        plt.figure(figsize=(12, 6))
        plt.bar(df["Model_Name"], df["Avg_Accuracy_IoU"], color='lightgreen')
        plt.title("Genauigkeit: Durchschnittliche IoU (Masken-Passgenauigkeit)")
        plt.ylabel("IoU (0.0 - 1.0)")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_BASE_DIR, "Graph_Genauigkeit_IoU.png"))

        # 2. Graph: Geschwindigkeit (Inferenzzeit)
        plt.figure(figsize=(12, 6))
        plt.bar(df["Model_Name"], df["Avg_Inference_ms"], color='skyblue')
        plt.title("Geschwindigkeit: Inferenzzeit pro Bild")
        plt.ylabel("Millisekunden (ms)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_BASE_DIR, "Graph_Geschwindigkeit_Ms.png"))

        # 3. Graph: Fehleranalyse (Gestapelt oder Nebeneinander)
        if "Misses (False Neg)" in df.columns and "Ghosts (False Pos)" in df.columns:
            plt.figure(figsize=(12, 6))
            x = np.arange(len(df["Model_Name"]))
            width = 0.35
            
            plt.bar(x - width/2, df["Misses (False Neg)"], width, label='Misses (Übersehen)', color='orange')
            plt.bar(x + width/2, df["Ghosts (False Pos)"], width, label='Ghosts (Falsch Erkannt)', color='red')
            
            plt.xlabel('Modell')
            plt.ylabel('Anzahl Fehler')
            plt.title('Fehleranalyse: Misses vs Ghosts')
            plt.xticks(x, df["Model_Name"], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_BASE_DIR, "Graph_Fehleranalyse.png"))

        print("\n Fertig! Alle 3 Graphen, CSV und Bilder sind erstellt.")

if __name__ == "__main__":
    run_benchmark()