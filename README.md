# S.I.S Stage 1 Training Module (RadIR + PhoBERT)

Mã nguồn trong thư mục này thực hiện bài toán **Tiền huấn luyện không điều kiện (Unconditional Image-Text Retrieval - Stage 1)** dành cho tập dữ liệu S.I.S Cần Thơ dựa trên framework [RadIR](file:///c:/Users/DUONG/Desktop/RadIR).

---

## 1. Cấu trúc thư mục

```
RadIR/S.I.S/stage1/
├── __init__.py           # Package initializer
├── dataset.py            # SISMRIDataset (Đọc ảnh 3D Volume DWI/ADC & Text)
├── model.py              # Xây dựng mô hình (CTViT 3D + PhoBERT + RADIR)
├── train.py              # Script chạy huấn luyện chính
├── evaluate.py           # Script tính toán chỉ số Recall@1, 5, 10
├── config.yaml           # Tệp cấu hình siêu tham số và đường dẫn
└── README.md             # Tài liệu hướng dẫn sử dụng
```

---

## 2. Yêu cầu môi trường & Cài đặt

Mã nguồn phụ thuộc vào thư viện RadIR và `transformer_maskgit`:

```bash
# Kích hoạt môi trường RadIR
conda activate RadIR

# Cài đặt PhoBERT và các thư viện hỗ trợ
pip install transformers sentencepiece openpyxl pyyaml
```

---

## 3. Hướng dẫn chạy Huấn luyện (Training)

Chạy huấn luyện mô hình Stage 1 với các tham số mặc định trong `config.yaml`:

```bash
cd c:\Users\DUONG\Desktop\RadIR\S.I.S\stage1
python train.py --config config.yaml
```

Hoặc ghi đè các tham số từ dòng lệnh:

```bash
python train.py --batch_size 4 --lr 0.0001 --epochs 30
```

### Các thông số chính trong `config.yaml`:
*   `report_path`: Đường dẫn đến tệp báo cáo Excel (`report_filtered.xlsx` hoặc `report.xlsx`).
*   `images_dir`: Đường dẫn đến thư mục chứa ảnh (`S.I.S/images`).
*   `text_column`: Cột văn bản dùng để huấn luyện (mặc định `'KETLUAN'`).
*   `sequences`: Chuỗi xung được sử dụng (mặc định `['DWI']`, có thể chọn `['DWI', 'ADC']`).
*   `output_dir`: Thư mục lưu vết các checkpoint (`stage1/checkpoints`).

---

## 4. Hướng dẫn Đánh giá (Evaluation)

Tính toán chỉ số **Recall@1**, **Recall@5**, **Recall@10** trên tập kiểm thử:

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_stage1_model.pt
```

---

## 5. Tùy chỉnh dữ liệu Text trong tương lai

Khi bạn tiến hành hiệu chỉnh xong file text báo cáo y khoa:
1. Bạn có thể lưu thành file `.xlsx` hoặc `.csv` mới (ví dụ: `report_custom.xlsx`).
2. Mở file `config.yaml` và cập nhật hai trường:
   ```yaml
   report_path: "c:/Users/DUONG/Desktop/RadIR/S.I.S/report_custom.xlsx"
   text_column: "TEN_COT_TEXT_CUA_BAN"
   ```
3. Chạy lại `python train.py` mà không cần chỉnh sửa bất kỳ dòng code Python nào.
