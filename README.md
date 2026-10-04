# Xây dựng pipeline hồi quy với tiền xử lý và giảm chiều PCA

Bài tập cá nhân — Nhận dạng mẫu
**Họ và tên:** Lưu Ngọc Anh — **MSSV:** B26CHKH006

## Mô tả

Pipeline hồi quy hoàn chỉnh (tiền xử lý → mở rộng đặc trưng đa thức → chuẩn hóa → PCA
tùy chọn → mô hình) trên bộ dữ liệu **Diabetes** (scikit-learn, 442 mẫu, 10 đặc trưng
gốc, target liên tục). So sánh baseline với 7 mô hình hồi quy (Linear Regression, Ridge,
Lasso, KNN, Random Forest, Gradient Boosting, SVR) bằng 5-fold cross-validation (MAE,
RMSE, R², độ ổn định), đồng thời kiểm tra tác động của PCA lên từng mô hình.

## Cấu trúc repo

```
.
├── PCA_Regression_Pipeline.ipynb   # Notebook chính — đã chạy sẵn, đầy đủ kết quả/biểu đồ/kết luận
├── build_notebook.py               # Script sinh notebook (thực thi từng cell, nhúng output, xuất hình)
├── extract_results.py              # Script tái tạo pipeline, xuất bảng kết quả full-precision
├── report_assets/                  # Kết quả xuất ra dùng cho báo cáo
│   ├── all_results.csv, no_pca.csv, pca.csv, pca_impact.csv, summary.json
│   └── figures/                    # Hình (PNG) dùng trong báo cáo Word
├── BaoCao_PCA_Regression_LuuNgocAnh_B26CHKH006.docx   # Báo cáo Word hoàn chỉnh
└── README.md
```

## Kết quả chính (5-fold CV trên tập train)

| Mô hình | PCA | MAE | RMSE | R² | Std RMSE | Std R² |
|---|---|---|---|---|---|---|
| Baseline (mean) | — | 66.8673 | 78.1336 | -0.0273 | 5.6383 | 0.0244 |
| **Ridge** | **Có** | **46.0476** | **57.2646** | **0.4405** | **2.4022** | **0.0823** |
| Lasso | Có | 46.0539 | 57.2800 | 0.4401 | 2.4187 | 0.0825 |
| Linear Regression | Có | 46.0550 | 57.2825 | 0.4401 | 2.4295 | 0.0826 |
| Random Forest | Không | 48.9097 | 59.3248 | 0.4041 | 2.6425 | 0.0473 |

Mô hình tốt nhất qua cross-validation: **Ridge + PCA** (giữ ≥95% phương sai, 33/65
thành phần). Ba mô hình tuyến tính + PCA chênh nhau chưa tới 0.02 RMSE (nhỏ hơn nhiều so
với std ≈ 2.4), nên thực chất là tương đương. Sau khi tinh chỉnh `alpha=50` bằng
GridSearchCV và đánh giá trên tập test giữ riêng (20%): MAE = 44.7379, RMSE = 55.3385,
R² = 0.4220 (baseline: MAE = 64.0065, RMSE = 73.2225, R² = -0.0120), tức giảm 24.4% RMSE.

## Tác động của PCA (thay đổi RMSE khi thêm PCA)

| Linear Regression | Lasso | Ridge | Gradient Boosting | SVR (RBF) | KNN | Random Forest |
|---|---|---|---|---|---|---|
| -8.36% | -7.88% | -5.29% | -3.69% | -0.75% | +0.43% | +3.93% |

PCA cải thiện rõ các mô hình tuyến tính (loại bỏ đa cộng tuyến giữa các đặc trưng đa
thức) và cả Gradient Boosting, gần như không ảnh hưởng tới KNN và SVR, nhưng làm giảm
hiệu quả của Random Forest. Phân tích chi tiết và hạn chế xem trong notebook (Bước 14)
và báo cáo Word.

## Chạy lại

Đã kiểm tra với Python 3.13, scikit-learn 1.9.1, pandas 3.0.6, seaborn 0.13.2.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python build_notebook.py     # sinh lại notebook với output đã chạy + report_assets/figures/
python extract_results.py    # tái tạo bảng kết quả full-precision (report_assets/*.csv)
```
