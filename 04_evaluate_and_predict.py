"""
04_evaluate_and_predict.py
==========================
ขั้นตอนที่ 4: การประเมินผล (Evaluation) และนำโมเดลไปใช้งานตรวจจับภาพจริง (Inference)
โครงสร้างไดเรกทอรี:
- Model Weights: /workspace/artifacts/runs/coco_train_demo/weights/best.pt
- Dataset: /workspace/data/coco128
- Prediction Results: /workspace/artifacts/predictions/evaluation_demo
"""

import os
from ultralytics import YOLO

RUNS_DIR = "/workspace/artifacts/runs"
PREDICTIONS_DIR = "/workspace/artifacts/predictions"
PRETRAINED_DIR = "/workspace/artifacts/pretrained"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("📊 1. การประเมินผลความแม่นยำของโมเดล (Model Validation)")
    print("=" * 70)

    # ค้นหาโมเดลที่เทรนเสร็จแล้วจาก /workspace/artifacts/runs
    custom_model_path = os.path.join(RUNS_DIR, "coco_train_demo", "weights", "best.pt")
    fallback_model_path = os.path.join(PRETRAINED_DIR, "yolo11n.pt")

    if os.path.exists(custom_model_path):
        print(f"✅ โหลดโมเดลที่เทรนเสร็จจาก: {custom_model_path}")
        model = YOLO(custom_model_path)
    elif os.path.exists(fallback_model_path):
        print(f"ℹ️ โหลดโมเดล Pretrained จาก: {fallback_model_path}")
        model = YOLO(fallback_model_path)
    else:
        print("ℹ️ ดาวน์โหลด Base Model สำหรับทดสอบ")
        model = YOLO("yolo11n.pt")

    # รัน validation บนชุดข้อมูล coco128 ใน /workspace/data
    metrics = model.val(data="coco128.yaml", split="val", project=RUNS_DIR, name="val")

    print("\n" + "-" * 70)
    print("📈 ค่าสถิติความแม่นยำ (Metrics):")
    print("-" * 70)
    print(f"• mAP50      : {metrics.box.map50 * 100:.2f}% (ความแม่นยำเมื่อ IoU >= 50%)")
    print(f"• mAP50-95   : {metrics.box.map * 100:.2f}% (ค่าเฉลี่ยความแม่นยำมาตรฐาน COCO)")
    print(f"• Precision  : {metrics.box.mp * 100:.2f}% (ความแม่นยำของการทายถูก)")
    print(f"• Recall     : {metrics.box.mr * 100:.2f}% (อัตราการตรวจจับวัตถุเจอ)")

    print("\n" + "=" * 70)
    print("🎯 2. นำโมเดลไปทำนายภาพใหม่ (Inference on New Image)")
    print("=" * 70)

    # ทดสอบ Predict บนภาพ sample zidane.jpg ในโฟลเดอร์โปรเจกต์
    test_image = "zidane.jpg" if os.path.exists("zidane.jpg") else "https://ultralytics.com/images/zidane.jpg"
    print(f"กำลังทดสอบตรวจจับภาพ: {test_image}")

    results = model.predict(
        source=test_image,
        conf=0.4,
        save=True,
        project=PREDICTIONS_DIR,
        name="evaluation_demo",
        exist_ok=True
    )

    saved_path = os.path.join(PREDICTIONS_DIR, "evaluation_demo")
    print(f"\n💾 บันทึกผลลัพธ์ภาพที่ตีกรอบแล้วที่: {saved_path}")

if __name__ == "__main__":
    main()
