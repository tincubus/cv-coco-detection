#!/usr/bin/env bash
# ==============================================================================
# download_full_coco.sh
# ==============================================================================
# สคริปต์นี้สำหรับดาวน์โหลด COCO 2017 Dataset ตัวเต็ม (Full Dataset ~20GB+)
# ปลายทางจัดเก็บ: /workspace/data/coco
# ==============================================================================

set -e

DATA_DIR="/workspace/data/coco"
mkdir -p "$DATA_DIR"

echo "======================================================================"
echo "เริ่มต้นดาวน์โหลด COCO 2017 Dataset เข้าสู่ /workspace/data/coco ..."
echo "คำเตือน: ต้องใช้พื้นที่ว่างประมาณ 25GB และเวลาดาวน์โหลดขึ้นกับอินเทอร์เน็ต"
echo "======================================================================"

python3 -c "
from ultralytics.data.utils import check_det_dataset
print('กำลังตรวจสอบและดาวน์โหลด COCO 2017 เข้าสู่ /workspace/data ...')
check_det_dataset('coco.yaml')
"

echo "======================================================================"
echo "ดาวน์โหลด COCO 2017 สำเร็จแล้วที่: $DATA_DIR"
echo "คุณสามารถเริ่มเทรนโมเดลเต็มรูปแบบได้ด้วยคำสั่ง:"
echo "python3 -c \"from ultralytics import YOLO; model = YOLO('/workspace/artifacts/pretrained/yolo11n.pt'); model.train(data='coco.yaml', epochs=100, imgsz=640, device=0, project='/workspace/artifacts/runs', name='coco_full')\""
echo "======================================================================"
