"""
02_explore_coco_data.py
=======================
ขั้นตอนที่ 2: ทำความเข้าใจโครงสร้าง COCO Dataset สำหรับงาน Detection
โครงสร้างไดเรกทอรี:
- Datasets: /workspace/data/coco128
"""

import os
from ultralytics.data.utils import check_det_dataset

DATA_DIR = "/workspace/data"

def main():
    print("=" * 70)
    print("📂 กำลังตรวจสอบและเตรียมชุดข้อมูล COCO128 ใน /workspace/data ...")
    print("=" * 70)

    # Ultralytics จะดาวน์โหลดและเก็บ dataset ลงใน datasets_dir (/workspace/data)
    data_dict = check_det_dataset("coco128.yaml")

    dataset_path = data_dict["path"]
    classes = data_dict["names"]

    print(f"\n✅ ตำแหน่งที่เก็บชุดข้อมูล: {dataset_path}")
    print(f"📊 จำนวนคลาสทั้งหมด: {len(classes)} คลาส")

    print("\n" + "=" * 70)
    print("📋 รายชื่อ 80 Classes ใน COCO Dataset:")
    print("=" * 70)
    class_names = [f"[{k}] {v}" for k, v in classes.items()]
    col_width = 20
    for i in range(0, len(class_names), 4):
        row = class_names[i:i+4]
        print("".join(c.ljust(col_width) for c in row))

    print("\n" + "=" * 70)
    print("🔍 โครงสร้างไฟล์ Label สำหรับ Object Detection (YOLO format)")
    print("=" * 70)
    
    labels_dir = os.path.join(dataset_path, "labels", "train2017")
    if os.path.exists(labels_dir):
        label_files = sorted(os.listdir(labels_dir))
        sample_label = label_files[0]
        label_file_path = os.path.join(labels_dir, sample_label)

        print(f"ตัวอย่างไฟล์ Label: {sample_label} ({label_file_path})")
        print("-" * 50)
        with open(label_file_path, "r") as f:
            lines = f.readlines()

        print(f"มีวัตถุในภาพนี้ทั้งหมด {len(lines)} ชิ้น:")
        for line in lines[:5]:
            parts = line.strip().split()
            cls_id = int(parts[0])
            cls_name = classes[cls_id]
            x_c, y_c, w, h = [float(p) for p in parts[1:]]
            print(f"  • Class {cls_id} ('{cls_name}') => จุดกึ่งกลาง: ({x_c:.3f}, {y_c:.3f}), กว้าง: {w:.3f}, สูง: {h:.3f}")

        print("\n💡 คำอธิบาย Bounding Box Format:")
        print("  1. พิกัดทั้งหมดถูก Normalize ให้อยู่ในช่วง 0.0 ถึง 1.0 (เทียบกับความกว้าง-ยาวของรูปภาพ)")
        print("  2. แต่ละบรรทัดคือ: <class_id> <center_x> <center_y> <width> <height>")

    print("\n🎉 ชุดข้อมูลพร้อมใช้งานใน /workspace/data เรียบร้อย!")

if __name__ == "__main__":
    main()
