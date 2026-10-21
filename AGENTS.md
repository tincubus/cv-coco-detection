# AI Agent Guidelines: COCO Computer Vision Detection Tutorial

เอกสารนี้มีไว้สำหรับ AI Coding Agent ที่เปิดเข้ามาทำงานใน Workspace นี้ โดยสรุปบริบท กฎเกณฑ์ โครงสร้างไดเรกทอรี และเวิร์กโฟลว์ทั้งหมด เพื่อให้ Agent สามารถเริ่มงานและให้คำแนะนำผู้ใช้ได้อย่างถูกต้องทันทีแม้ไม่มีประวัติแชตเดิม

---

## 1. จุดประสงค์ของโปรเจกต์ (Project Identity)
* โปรเจกต์นี้จัดทำขึ้นเพื่อเป็น **Tutorial สอนทำ Computer Vision: Object Detection ฉบับมือใหม่**
* ใช้โมเดลตระกูล **YOLO (You Only Look Once)** โดย Ultralytics ร่วมกับชุดข้อมูลมาตรฐานโลก **COCO (Common Objects in Context)**
* เน้นการอธิบายคอนเซปต์แบบเข้าใจง่าย เป็นขั้นเป็นตอน และสามารถรันได้จริงบน GPU (NVIDIA CUDA)

---

## 2. โครงสร้างไดเรกทอรีและกฎการจัดเก็บไฟล์ (Strict File Layout)

โปรเจกต์นี้มีกฎการแยกไฟล์อย่างเคร่งครัด **ห้ามบันทึกไฟล์สุ่มสี่สุ่มห้าในโฟลเดอร์โค้ด**:

```text
/workspace/
├── artifacts/
│   ├── pretrained/                 # โมเดล Pretrained ทั้งหมด (เช่น yolo11n.pt, yolo26n.pt)
│   ├── runs/                       # ผลลัพธ์จากการเทรนและ validation (best.pt, charts, loss)
│   └── predictions/                # ผลลัพธ์ภาพที่ตรวจจับและตีกรอบแล้ว (quickstart_demo, evaluation_demo)
│
├── data/                           # ชุดข้อมูลทั้งหมด (Datasets)
│   ├── coco128/                    # ชุดข้อมูลทดลอง 128 ภาพ ครบ 80 คลาส
│   ├── coco8/                      # ชุดข้อมูลขนาดเล็ก 8 ภาพ
│   └── coco/                       # ชุดข้อมูลเต็ม COCO 2017 (~20GB+)
│
└── cv-coco-detection/              # โฟลเดอร์โค้ด เอกสาร และภาพตัวอย่างสำหรับทดสอบ
    ├── bus.jpg                     # ภาพตัวอย่างสำหรับทดสอบ Pretrained Inference
    ├── zidane.jpg                  # ภาพตัวอย่างสำหรับทดสอบ Evaluation & Prediction
    ├── 01_quickstart_inference.py  # สคริปต์รันตรวจจับภาพทันทีด้วย Pretrained Model
    ├── 02_explore_coco_data.py     # สคริปต์สำรวจ 80 คลาส และโครงสร้าง Label Bounding Box
    ├── 03_train_coco.py            # สคริปต์ฝึกสอนโมเดล (Fine-Tuning) บน GPU
    ├── 04_evaluate_and_predict.py  # สคริปต์วัดค่า mAP และนำโมเดลไปทดสอบภาพใหม่
    ├── tutorial_coco_detection.ipynb # Jupyter Notebook สำหรับรันทีละขั้นตอน
    ├── download_full_coco.sh       # สคริปต์ดาวน์โหลด COCO 2017 ตัวเต็ม
    ├── README.md                   # คู่มือสำหรับ User
    └── AGENTS.md                   # คู่มือสำหรับ Agent ฉบับนี้
```

### กฎเหล็กที่ Agent ต้องปฏิบัติตามเสมอ:
1. **ห้ามบันทึก weights หรือผลการเทรนตกค้างไว้ใน `/workspace/cv-coco-detection`**:
   - Pretrained model ต้องชี้ไปที่ `/workspace/artifacts/pretrained/`
   - ผลการเทรนต้องบันทึกใน `/workspace/artifacts/runs/`
   - ผลการทำนายภาพต้องบันทึกใน `/workspace/artifacts/predictions/`
   - ชุดข้อมูลต้องเก็บไว้ใน `/workspace/data/` (เช่น `/workspace/data/coco128/`)
   - ภาพทดสอบ `bus.jpg` และ `zidane.jpg` ประจำอยู่ในโฟลเดอร์โปรเจกต์สำหรับรันเทสต์
2. **รักษาความสะอาดของ Code Directory**: ห้ามมีไฟล์ที่ไม่เกี่ยวข้องมาปะปนในโฟลเดอร์นี้
3. **ห้ามใช้ Emoticon หรือ Emoji โดยเด็ดขาด**: ห้ามใส่อีโมจิ ไอคอน หรือ emoticon ใดๆ ในเอกสาร โค้ด ข้อความ log และคำตอบที่สื่อสารกับผู้ใช้ เพื่อความเป็นมืออาชีพและเรียบร้อย

---

## 3. ลำดับขั้นตอนการสอน (Tutorial Learning Path)

เมื่อ User ขอให้รัน แนะนำ หรือต่อยอดบทเรียน ให้เรียงลำดับตามขั้นตอนดังนี้:

### Step 1: Pre-trained Inference (ลองตรวจจับทันที)
* **ไฟล์**: `python3 01_quickstart_inference.py`
* **แนวคิด**: โหลด weights จาก `/workspace/artifacts/pretrained/yolo11n.pt` และส่งภาพ `bus.jpg` เข้าไปตรวจจับทันที
* **ผลลัพธ์**: ได้รายการวัตถุ, ความมั่นใจ (Confidence Score), พิกัดกรอบ (Bounding Box `[x1, y1, x2, y2]`) และบันทึกภาพพร้อมกรอบลงใน `/workspace/artifacts/predictions/quickstart_demo`

### Step 2: Explore COCO Dataset (ทำความรู้จักโครงสร้างข้อมูล)
* **ไฟล์**: `python3 02_explore_coco_data.py`
* **แนวคิด**: 
  * COCO มี 80 คลาสสากล
  * Label ในฟอร์แมต YOLO จะเป็น: `<class_id> <x_center> <y_center> <width> <height>` ทุกค่าถูก Normalize ให้อยู่ระหว่าง `0.0 - 1.0` เสมอ

### Step 3: Train & Fine-Tune Model (การฝึกสอนโมเดล)
* **ไฟล์**: `python3 03_train_coco.py`
* **แนวคิด**:
  * ใช้ **Transfer Learning** เริ่มต้นจาก Base Weights เพื่อประหยัดเวลาและพลังงาน
  * Hyperparameters สำคัญ: `data='coco128.yaml'`, `epochs=5`, `imgsz=640`, `batch=16`, `device=0` (GPU)
  * ผลลัพธ์ Weights ที่ดีที่สุด (`best.pt`) จะถูกเซฟไว้ใน `/workspace/artifacts/runs/coco_train_demo/weights/best.pt`

### Step 4: Evaluate & Predict (การวัดผลและนำไปใช้)
* **ไฟล์**: `python3 04_evaluate_and_predict.py`
* **แนวคิด**:
  * **Precision**: ความแม่นยำในการทายถูก (กันการทายมั่ว)
  * **Recall**: ตรวจจับวัตถุจริงในภาพเจอได้กี่ % (กันการหลุดรอด)
  * **mAP50**: Mean Average Precision ที่เกณฑ์ IoU >= 50%
  * **mAP50-95**: เกณฑ์มาตรฐานสากลของ COCO ที่ประเมินตั้งแต่ IoU 50% ถึง 95%
  * ทดสอบทำนายภาพ `zidane.jpg` และบันทึกลง `/workspace/artifacts/predictions/evaluation_demo`

---

## 4. สภาพแวดล้อมระบบ (Environment)
* **OS**: Linux (Containerized)
* **Python**: 3.12+
* **CUDA / GPU**: NVIDIA GPU รองรับ CUDA เต็มรูปแบบ (`device=0`)
* **Libraries หลัก**: `ultralytics`, `torch`, `torchvision`, `pycocotools`, `nbformat`

---

## 5. แนวทางการตอบ User (Agent Persona & Tone)
* ใช้ภาษาไทยที่สุภาพ เข้าใจง่าย เป็นมิตรกับมือใหม่ (Beginner-Friendly)
* เมื่ออธิบาย Concept ทาง Computer Vision (เช่น Bounding Box, IoU, Confidence, mAP) ให้ยกตัวอย่างเปรียบเทียบในชีวิตประจำวัน
* แนบลิงก์ไปยังไฟล์ที่เกี่ยวข้องเสมอ เช่น [`01_quickstart_inference.py`](file:///workspace/cv-coco-detection/01_quickstart_inference.py) ด้วยรูปแบบ Markdown Link `[file](file:///absolute/path)`
* ห้ามใช้ Emoticon หรือ Emoji ในคำตอบและเอกสารทุกประเภท ให้ใช้ข้อความที่ชัดเจนและเรียบร้อย
