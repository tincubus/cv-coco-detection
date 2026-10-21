"""
03_train_coco.py
================
ขั้นตอนที่ 3: การเทรน (Fine-Tuning) โมเดล Object Detection ด้วยชุดข้อมูล COCO
โครงสร้างไดเรกทอรี:
- Base Pretrained Model: /workspace/artifacts/pretrained/yolo11n.pt
- Dataset: /workspace/data/coco128
- Training Runs (ผลการเทรน): /workspace/artifacts/runs/coco_train_demo
"""

import os
import torch
from ultralytics import YOLO

# กำหนด path ให้เป็นไปตามโครงสร้างมาตรฐาน
PRETRAINED_DIR = "/workspace/artifacts/pretrained"
RUNS_DIR = "/workspace/artifacts/runs"
DATA_DIR = "/workspace/data"

os.makedirs(PRETRAINED_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

BASE_MODEL_PATH = os.path.join(PRETRAINED_DIR, "yolo11n.pt")

def main():
    print("=" * 70)
    print("🏋️‍♂️ เริ่มต้นการฝึกสอนโมเดล (Training Object Detection)")
    print(f"📦 Pretrained Model: {BASE_MODEL_PATH}")
    print(f"📂 Dataset Directory: {DATA_DIR}")
    print(f"🎯 Output Runs Directory: {RUNS_DIR}")
    print("=" * 70)

    # ตรวจสอบสถานะ GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"⚡ อุปกรณ์ที่ใช้ประมวลผล: {device_name} (Device: {device})")

    # โหลด Base Model จาก /workspace/artifacts/pretrained/
    if not os.path.exists(BASE_MODEL_PATH):
        print("กำลังเตรียม Base Weights ...")
        model = YOLO("yolo11n.pt")
    else:
        model = YOLO(BASE_MODEL_PATH)

    # เริ่มกระบวนการเทรน
    print("\n🚀 กำลังเริ่มกระบวนการเทรน...")
    results = model.train(
        data="coco128.yaml",
        epochs=5,
        imgsz=640,
        batch=16,
        device=device,
        project=RUNS_DIR,
        name="coco_train_demo",
        exist_ok=True,
        plots=True
    )

    print("\n" + "=" * 70)
    print("✅ การเทรนเสร็จสมบูรณ์เรียบร้อยแล้ว!")
    print("=" * 70)
    
    save_dir = str(results.save_dir)
    best_weight = os.path.join(save_dir, "weights", "best.pt")
    last_weight = os.path.join(save_dir, "weights", "last.pt")
    
    print(f"📁 โมเดลที่เทรนเสร็จถูกบันทึกไว้ใน:")
    print(f"   • Best Weights: {best_weight}")
    print(f"   • Last Weights: {last_weight}")
    print(f"📈 ภาพกราฟผลการเทรน (Loss & Metrics): {os.path.join(save_dir, 'results.png')}")

if __name__ == "__main__":
    main()
