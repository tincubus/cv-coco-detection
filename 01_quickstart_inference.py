"""
01_quickstart_inference.py
==========================
ขั้นตอนที่ 1: สำหรับมือใหม่ - ลองใช้งาน Model ที่เทรนบน COCO สำเร็จรูปทันที (Pre-trained Inference)
โครงสร้างไดเรกทอรี:
- โมเดล Pretrained: /workspace/artifacts/pretrained
- ผลลัพธ์ Predictions: /workspace/artifacts/predictions
"""

import os
from ultralytics import YOLO

# กำหนด path ให้เป็นไปตามโครงสร้างมาตรฐาน
PRETRAINED_DIR = "/workspace/artifacts/pretrained"
PREDICTIONS_DIR = "/workspace/artifacts/predictions"
os.makedirs(PRETRAINED_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(PRETRAINED_DIR, "yolo11n.pt")

def main():
    print("=" * 60)
    print("🚀 โหลดโมเดล YOLO11 ที่ผ่านการเทรนบนชุดข้อมูล COCO (80 Classes)")
    print(f"📁 Pretrained Model Path: {MODEL_PATH}")
    print("=" * 60)

    # หากยังไม่มีไฟล์โมเดลในโฟลเดอร์ pretrained จะดาวน์โหลดและเก็บไว้ที่นี่
    if not os.path.exists(MODEL_PATH):
        print("กำลังดาวน์โหลด Pretrained Weights เข้าสู่ /workspace/artifacts/pretrained ...")
        temp_model = YOLO("yolo11n.pt")
        # ย้าย weight เข้าสู่ pretrained folder ถ้าถูกโหลดมาที่อื่น
        if os.path.exists("yolo11n.pt"):
            os.rename("yolo11n.pt", MODEL_PATH)

    model = YOLO(MODEL_PATH)

    # กำหนดภาพทดสอบ (ใช้ภาพ sample bus.jpg ในโฟลเดอร์โปรเจกต์)
    sample_image = "bus.jpg" if os.path.exists("bus.jpg") else "https://ultralytics.com/images/bus.jpg"
    print(f"\n📸 กำลังทำการตรวจจับวัตถุบนภาพ: {sample_image}")

    # รัน Object Detection
    # บันทึกภาพพร้อมกรอบลงใน /workspace/artifacts/predictions/quickstart_demo
    results = model.predict(
        source=sample_image,
        conf=0.5,
        save=True,
        project=PREDICTIONS_DIR,
        name="quickstart_demo",
        exist_ok=True
    )

    result = results[0]

    print("\n" + "=" * 60)
    print("🎯 สรุปผลการตรวจจับวัตถุ (Detections):")
    print("=" * 60)

    # วนลูปอ่านกล่อง Bounding Box แต่ละกล่องที่โมเดลตรวจพบ
    for idx, box in enumerate(result.boxes, 1):
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = float(box.conf[0].item()) * 100
        coords = [round(c, 1) for c in box.xyxy[0].tolist()]

        print(f"[{idx}] เจอวัตถุ: '{class_name}' (ID: {class_id}) | ความมั่นใจ: {confidence:.1f}% | พิกัดกล่อง: {coords}")

    saved_path = os.path.join(PREDICTIONS_DIR, "quickstart_demo")
    print(f"\n💾 บันทึกรูปภาพผลลัพธ์ไว้ที่: {saved_path}")

if __name__ == "__main__":
    main()
