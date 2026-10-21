# Computer Vision Object Detection กับ COCO Dataset (ฉบับมือใหม่)

โปรเจกต์เรียนรู้ **Computer Vision: Object Detection** ด้วยชุดข้อมูลมาตรฐานโลก **COCO (Common Objects in Context)** และโมเดล **YOLO (You Only Look Once)**

---

## โครงสร้างไดเรกทอรีมาตรฐาน (Directory Structure)

ทุกไฟล์ถูกจัดระเบียบตามโครงสร้างระบบอย่างเป็นสัดส่วน ดังนี้:

```text
/workspace/
├── artifacts/
│   ├── pretrained/                 # ที่เก็บโมเดลเริ่มต้น (Pretrained Weights เช่น yolo11n.pt)
│   ├── runs/                       # ผลลัพธ์จากการเทรนและ validation (best.pt, charts, loss)
│   └── predictions/                # ผลลัพธ์ภาพที่ผ่านการตรวจจับและตีกรอบ (Inference Detections)
│
├── data/                           # ที่เก็บชุดข้อมูล (Datasets)
│   ├── coco128/                    # ชุดข้อมูลทดลอง 128 ภาพ ครบ 80 คลาส
│   ├── coco8/                      # ชุดข้อมูลขนาดจิ๋ว 8 ภาพ
│   └── coco/                       # ชุดข้อมูลเต็ม 2017 (~20GB+) เมื่อดาวน์โหลด
│
└── cv-coco-detection/              # โค้ดและภาพตัวอย่างสำหรับเรียนรู้
    ├── bus.jpg                     # ภาพตัวอย่างสำหรับทดสอบ Quickstart
    ├── zidane.jpg                  # ภาพตัวอย่างสำหรับทดสอบ Evaluation
    ├── 01_quickstart_inference.py  # ทดสอบรัน Detection ทันทีด้วย Pretrained Model
    ├── 02_explore_coco_data.py     # สำรวจ 80 คลาส และโครงสร้าง Label Bounding Box
    ├── 03_train_coco.py            # ฝึกสอนโมเดล (Fine-Tuning) บน GPU
    ├── 04_evaluate_and_predict.py  # วัดค่าความแม่นยำ (mAP) และรันตรวจจับภาพใหม่
    ├── tutorial_coco_detection.ipynb # สมุด Interactive Notebook สำหรับรันทีละขั้นตอน
    ├── download_full_coco.sh       # สคริปต์ดาวน์โหลด COCO 2017 ตัวเต็ม
    └── README.md                   # คู่มืออธิบายการใช้งาน
```

---

## COCO Dataset คืออะไร? (อธิบายแบบมือใหม่)

* **COCO (Common Objects in Context)** เป็นชุดข้อมูลมาตรฐานโลก มี 80 คลาส เช่น คน (`person`), รถ (`car`), แมว (`cat`), จอทีวี (`tv`), โน้ตบุ๊ก (`laptop`)
* **จุดเริ่มต้นสำหรับผู้เริ่มต้น**: แนะนำให้เริ่มฝึกด้วย **COCO128** ในโฟลเดอร์ `/workspace/data/coco128` ซึ่งมีรูปภาพตัวอย่าง 128 ภาพ ครบทั้ง 80 คลาส โหลดและเทรนได้รวดเร็ว ช่วยให้เข้าใจ pipeline ได้ทันทีโดยไม่ต้องรอโหลด 20GB

---

## ลำดับขั้นตอนการเรียนรู้ (Step-by-Step Tutorial)

### ขั้นที่ 1: ตรวจจับวัตถุได้ทันทีไม่ต้องรอเทรน (Pre-trained Inference)
รันคำสั่ง:
```bash
python3 01_quickstart_inference.py
```
* **การทำงาน**: โหลด `yolo11n.pt` จาก `/workspace/artifacts/pretrained` มาตรวจจับภาพ
* **ผลลัพธ์**: ภาพที่ตีกรอบพร้อมระบุชื่อวัตถุและความมั่นใจ (%) จะถูกบันทึกไว้ใน `/workspace/artifacts/predictions/quickstart_demo`

---

### ขั้นที่ 2: สำรวจ Dataset ใน `/workspace/data`
รันคำสั่ง:
```bash
python3 02_explore_coco_data.py
```
* **สิ่งที่ได้เรียนรู้**:
  * รายชื่อ 80 Classes ของ COCO
  * โครงสร้างพิกัดของกรอบ Bounding Box: `<class_id> <x_center> <y_center> <width> <height>` (ถูก Normalize เป็น `0.0 - 1.0` เสมอ)

---

### ขั้นที่ 3: ฝึกสอนโมเดล (Fine-Tuning on GPU)
รันคำสั่ง:
```bash
python3 03_train_coco.py
```
* **การทำงาน**: นำโมเดลจาก `/workspace/artifacts/pretrained/yolo11n.pt` มาเทรนต่อบนข้อมูลใน `/workspace/data/coco128`
* **ผลลัพธ์**: ไฟล์น้ำหนักโมเดล (`best.pt`, `last.pt`) และกราฟผลการเทรนจะถูกบันทึกไว้ใน `/workspace/artifacts/runs/coco_train_demo`

---

### ขั้นที่ 4: วัดความแม่นยำและการนำไปใช้งาน
รันคำสั่ง:
```bash
python3 04_evaluate_and_predict.py
```
* **การอ่านค่าประเมินผล (Evaluation Metrics)**:
  * **Precision**: ทายว่าใช่ แล้วถูกต้องจริงกี่ % (ป้องกันการทายมั่ว)
  * **Recall**: จากวัตถุจริงทั้งหมดในภาพ ตรวจจับเจอได้กี่ % (ป้องกันการมองข้าม)
  * **mAP50**: ความแม่นยำเฉลี่ยเมื่อกรอบทับซ้อน (IoU) อย่างน้อย 50%
  * **mAP50-95**: เกณฑ์มาตรฐานสากลของ COCO ที่คิดเฉลี่ยตั้งแต่เกณฑ์ 50% ถึง 95%
* **ผลลัพธ์**: บันทึกภาพตรวจจับใน `/workspace/artifacts/predictions/evaluation_demo`

---

## ใช้งานผ่าน Interactive Notebook
คุณสามารถเปิดไฟล์ [`tutorial_coco_detection.ipynb`](file:///workspace/cv-coco-detection/tutorial_coco_detection.ipynb) ใน VS Code เพื่อรันและเรียนรู้ทีละขั้นตอนได้อย่างสะดวกสบาย!
