# Phase 2.4 — Phân tích feature HPG Research

Nguồn: [notebook 02](../notebooks/02_feature_engineering_labeling.ipynb), chạy trên `data/processed/HPG_features_labeled_research.parquet`, SHA-256 `c764983413e04163934bdd6c14b0e2afaf8907807e236a8aece4eba7add573e2`. Dữ liệu VCI Research có gate `PASS_WITH_LIMITATIONS`, không phải dữ liệu validated toàn diện. Số liệu và hình dưới đây được tạo từ lần chạy này, không phải dataset cũ.

## 1. Phạm vi và phương pháp

Master có 2,702 hàng, 37 cột, gồm 5 OHLCV, 26 features và 6 cột forward/target; index từ 2016-01-04 đến 2026-10-30, duy nhất và tăng. Dtype nhãn nullable Int8, không có Inf trong features. Điều này kiểm tra cấu trúc, không chứng minh chất lượng nguồn hay tính point-in-time.

Train 2016-01-04–2020-12-31 có 1251 phiên; validation 2021-01-04–2021-12-31 có 250; test không phân tích nhãn 2022-01-04–2026-10-30 có 1201. Sau warm-up và purge, RQ1 còn 1051/249 và RQ2 1048/246 hàng train/validation. Số bị purge theo ranh giới lần lượt RQ1 1/1, RQ2 4/4. Không random shuffle; tính theo phiên thực, không phải ngày lịch.

| Giai đoạn | Đầu | Cuối | Rows | Đủ feature | RQ1 eligible | RQ2 eligible | Purge RQ1 | Purge RQ2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train | 2016-01-04 | 2020-12-31 | 1251 | 1052 | 1051 | 1048 | 1 | 4 |
| Validation | 2021-01-04 | 2021-12-31 | 250 | 250 | 249 | 246 | 1 | 4 |
| Reserved test | 2022-01-04 | 2026-10-30 | 1201 | không đánh giá | kín | kín | — | — |

RQ1 dùng return Close(t) đến Close(t+1); RQ2 là proxy mua Open(t+1), bán Open(t+4), sau phí. Purge theo thời điểm kết thúc nhãn; không áp dụng embargo riêng, chưa thực hiện walk-forward. Chỉ phân tích target train/validation; EDA mô tả trước đây đã dùng cả test, nên test không hoàn toàn kín.

Dataset lưu trước thay đổi cấu hình phí: `target_return_above_cost_proxy` vẫn dùng ngưỡng 0,45% cũ và không được phân tích trong notebook. RQ2 dùng các thành phần phí hiện tại (tổng gần đúng 0,50%). Giá adjusted không được coi là giá thực thi.

## 2. Chất lượng và phân phối

Trong train+validation, sma_ratio_200 thiếu nhiều nhất: 199 hàng (13.26%), chủ yếu do warm-up; tổng missing sau lần hợp lệ đầu là 0. Hai phiên 04–05/10/2021 có volume 45,357,500 và 26,571,500, thay đổi -0.41; đây là các quan sát trong dataset Research, không tự kết luận lỗi. Giá adjusted và volume cần đối chiếu nguồn và corporate actions. Logistic Regression nhạy scale và cực trị; các phép biến đổi cần được fit trên train.

Returns: log_return median 0.00000, P1/P99 -0.0533/0.0533, skew -0.07, excess kurtosis 1.39; đuôi và bất đối xứng là mô tả mẫu. Trend: sma_ratio_20 median 0.0150, min/max -0.178/0.249; horizon dài có thể phản ứng chậm hơn. Momentum: RSI median 55.4, P25–P75 45.0–65.4; std momentum_5/21 = 0.045/0.100. Volatility: volatility_21 median/max 0.278/0.639, đỉnh ngày 2020-04-07; volatility_63 median/max 0.286/0.503. Volume: volume_change_1 max 5.627, P99 2.436; volume_zscore_20 max 3.995. Biểu đồ cắt phạm vi hiển thị, không có nghĩa các cực trị đã bị loại khỏi thống kê.

Histogram và boxplot chỉ hiển thị P1–P99; thống kê dùng toàn bộ quan sát train, không cắt dữ liệu.

## 3. Tương quan giữa feature

Có 19 cặp đạt |Pearson| hoặc |Spearman| ≥ 0,90 trên train. Cặp đầu theo |Pearson|: sma_ratio_10–ema_ratio_12, Pearson 0.970, Spearman 0.962. volatility_21–volatility_63: Pearson 0.730, Spearman 0.687. SMA/EMA và momentum dùng lịch sử giá chồng lấn nên dễ đồng biến. Tương quan không phải predictive power; đa cộng tuyến có thể làm hệ số Logistic Regression thiếu ổn định, còn cây thường phân chia tầm quan trọng giữa biến tương tự. Không loại feature ở đây.

| A | B | Pearson | Spearman | Nhận xét |
| --- | --- | --- | --- | --- |
| sma_ratio_10 | ema_ratio_12 | 0.96951 | 0.96189 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | ema_ratio_26 | 0.96490 | 0.95981 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | ema_ratio_12 | 0.96183 | 0.95735 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_50 | macd_ratio | 0.96121 | 0.97073 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| volume_zscore_20 | volume_ratio_20 | 0.95351 | 0.98394 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| ema_ratio_26 | rsi_14 | 0.94824 | 0.97587 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_50 | ema_ratio_26 | 0.94027 | 0.93091 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| ema_ratio_12 | ema_ratio_26 | 0.93243 | 0.92077 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | momentum_10 | 0.92819 | 0.91993 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_10 | momentum_5 | 0.92784 | 0.91560 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| ema_ratio_26 | momentum_21 | 0.91619 | 0.90908 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| volatility_21 | atr_ratio_14 | 0.91090 | 0.90237 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | rsi_14 | 0.90823 | 0.93516 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_50 | momentum_21 | 0.90624 | 0.89462 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| macd_ratio | momentum_21 | 0.90544 | 0.89529 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| rsi_14 | bollinger_percent_b_20 | 0.90102 | 0.91020 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| ema_ratio_12 | bollinger_percent_b_20 | 0.89228 | 0.94422 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_50 | rsi_14 | 0.88844 | 0.91498 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | bollinger_percent_b_20 | 0.88405 | 0.94280 | Có thể trùng thông tin; cần kiểm tra bổ sung |

## 4. Nhãn và thông tin dự báo mô tả

Tỷ lệ tăng RQ1 train/validation: 49.38%/52.61%; tỷ lệ dương RQ2: 48.38%/55.28%. Baseline lớp đa số trong từng mẫu RQ1 là 50.62%/52.61%; không dùng lớp đa số validation để fit mô hình. Hai nhãn đo các khoảng giá khác nhau, nên accuracy cần so với baseline và không suy ra lợi nhuận chiến lược.

| period | target | n | positive | negative | positive_pct | negative_pct | majority_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Train | RQ1 | 1051 | 519 | 532 | 49.38154 | 50.61846 | 50.61846 |
| Validation | RQ1 | 249 | 131 | 118 | 52.61044 | 47.38956 | 52.61044 |
| Train | RQ2 | 1048 | 507 | 541 | 48.37786 | 51.62214 | 51.62214 |
| Validation | RQ2 | 246 | 136 | 110 | 55.28455 | 44.71545 | 55.28455 |

Ba |IC| lớn nhất trên train: return_lag_2 -0.0623 (n=1051); volume_ratio_20 +0.0618 (n=1051); volume_change_1 +0.0575 (n=1051). Đây là thứ tự mô tả, không phải các feature đã chứng minh dự báo: 26 phép thử, phụ thuộc thời gian và nhiễu chọn mẫu chưa được hiệu chỉnh. Chỉ 5 feature đầu theo train được đóng băng để đối chiếu validation. Không dùng test target.

| feature | group | n | pearson | spearman | abs_ic |
| --- | --- | --- | --- | --- | --- |
| return_lag_2 | Returns | 1051 | -0.06384 | -0.06229 | 0.06229 |
| volume_ratio_20 | Volume | 1051 | 0.04721 | 0.06182 | 0.06182 |
| volume_change_1 | Volume | 1051 | 0.04236 | 0.05750 | 0.05750 |
| momentum_21 | Momentum | 1051 | 0.04945 | 0.05586 | 0.05586 |
| macd_ratio | Trend | 1051 | 0.04383 | 0.05533 | 0.05533 |
| volume_zscore_20 | Volume | 1051 | 0.04351 | 0.05280 | 0.05280 |
| rsi_14 | Momentum | 1051 | 0.04974 | 0.04919 | 0.04919 |
| sma_ratio_200 | Trend | 1051 | 0.03993 | 0.04883 | 0.04883 |
| ema_ratio_26 | Trend | 1051 | 0.04163 | 0.04849 | 0.04849 |
| sma_ratio_50 | Trend | 1051 | 0.04171 | 0.04838 | 0.04838 |
| sma_ratio_20 | Trend | 1051 | 0.04004 | 0.04582 | 0.04582 |
| return_lag_21 | Returns | 1051 | 0.04548 | 0.04575 | 0.04575 |
| bollinger_percent_b_20 | Volatility | 1051 | 0.04966 | 0.04435 | 0.04435 |
| return_lag_10 | Returns | 1051 | 0.04188 | 0.03618 | 0.03618 |
| ema_ratio_12 | Trend | 1051 | 0.03329 | 0.03502 | 0.03502 |
| return_lag_3 | Returns | 1051 | 0.06209 | 0.02822 | 0.02822 |
| sma_ratio_10 | Trend | 1051 | 0.02591 | 0.02689 | 0.02689 |
| return_lag_1 | Returns | 1051 | 0.04815 | 0.02374 | 0.02374 |
| momentum_10 | Momentum | 1051 | 0.02712 | 0.02337 | 0.02337 |
| bollinger_width_20 | Volatility | 1051 | 0.01992 | 0.01705 | 0.01705 |
| momentum_5 | Momentum | 1051 | 0.01930 | 0.01525 | 0.01525 |
| volatility_63 | Volatility | 1051 | 0.01270 | 0.01463 | 0.01463 |
| atr_ratio_14 | Volatility | 1051 | 0.02814 | 0.01384 | 0.01384 |
| log_return | Returns | 1051 | -0.00292 | -0.00841 | 0.00841 |
| return_lag_5 | Returns | 1051 | -0.00387 | -0.00620 | 0.00620 |
| volatility_21 | Volatility | 1051 | 0.01528 | 0.00253 | 0.00253 |

IC là Spearman time-series của một mã; không phải bằng chứng significance hoặc alpha. Chọn 5 feature theo |IC| train trước khi xem validation.

| feature | train_ic | validation_ic | train_n | validation_n | same_sign |
| --- | --- | --- | --- | --- | --- |
| return_lag_2 | -0.06229 | -0.00442 | 1051 | 249 | True |
| volume_ratio_20 | 0.06182 | 0.04750 | 1051 | 249 | True |
| volume_change_1 | 0.05750 | -0.01579 | 1051 | 249 | False |
| momentum_21 | 0.05586 | 0.03198 | 1051 | 249 | True |
| macd_ratio | 0.05533 | 0.04351 | 1051 | 249 | True |

## 5. Phân tích conditional và ổn định

log_return: Q thấp/cao mean +0.00285/+0.00170, positive 55.5%/48.6%, không đơn điệu; sma_ratio_20: Q thấp/cao mean +0.00135/+0.00294, positive 47.9%/53.3%, không đơn điệu; momentum_5: Q thấp/cao mean +0.00130/+0.00165, positive 53.6%/49.5%, không đơn điệu; volatility_21: Q thấp/cao mean +0.00177/+0.00316, positive 49.8%/54.3%, không đơn điệu; volume_zscore_20: Q thấp/cao mean +0.00115/+0.00290, positive 46.9%/54.3%, không đơn điệu. Đây là khác biệt mô tả của các bin, có thể do nhiễu hoặc outlier; không suy ra quan hệ phi tuyến đã được kiểm chứng hay đặt ngưỡng giao dịch.

| feature | bin | n | mean_return | median_return | positive_rate |
| --- | --- | --- | --- | --- | --- |
| log_return | (-0.0734, -0.012] | 211 | 0.00285 | 0.00347 | 0.55450 |
| log_return | (-0.012, -0.00228] | 210 | -0.00042 | 0.00000 | 0.47619 |
| log_return | (-0.00228, 0.00428] | 210 | -0.00075 | 0.00000 | 0.43810 |
| log_return | (0.00428, 0.0157] | 210 | 0.00270 | 0.00170 | 0.51429 |
| log_return | (0.0157, 0.067] | 210 | 0.00170 | 0.00000 | 0.48571 |
| sma_ratio_20 | (-0.179, -0.0304] | 211 | 0.00135 | 0.00000 | 0.47867 |
| sma_ratio_20 | (-0.0304, -0.00206] | 210 | -0.00106 | 0.00000 | 0.43810 |
| sma_ratio_20 | (-0.00206, 0.0229] | 210 | 0.00030 | 0.00000 | 0.49524 |
| sma_ratio_20 | (0.0229, 0.0519] | 210 | 0.00256 | 0.00144 | 0.52381 |
| sma_ratio_20 | (0.0519, 0.249] | 210 | 0.00294 | 0.00208 | 0.53333 |
| momentum_5 | (-0.164, -0.029] | 211 | 0.00130 | 0.00250 | 0.53555 |
| momentum_5 | (-0.029, -0.00386] | 210 | -0.00057 | 0.00000 | 0.45238 |
| momentum_5 | (-0.00386, 0.0158] | 210 | 0.00026 | 0.00000 | 0.46190 |
| momentum_5 | (0.0158, 0.0388] | 210 | 0.00345 | 0.00177 | 0.52381 |
| momentum_5 | (0.0388, 0.208] | 210 | 0.00165 | 0.00000 | 0.49524 |
| volatility_21 | (0.138, 0.213] | 211 | 0.00177 | 0.00000 | 0.49763 |
| volatility_21 | (0.213, 0.246] | 210 | 0.00040 | 0.00053 | 0.50000 |
| volatility_21 | (0.246, 0.308] | 210 | 0.00138 | 0.00000 | 0.49048 |
| volatility_21 | (0.308, 0.386] | 210 | -0.00063 | 0.00000 | 0.43810 |
| volatility_21 | (0.386, 0.639] | 210 | 0.00316 | 0.00352 | 0.54286 |
| volume_zscore_20 | (-1.825, -0.841] | 211 | 0.00115 | 0.00000 | 0.46919 |
| volume_zscore_20 | (-0.841, -0.383] | 210 | -0.00004 | 0.00000 | 0.45238 |
| volume_zscore_20 | (-0.383, 0.118] | 210 | 0.00065 | 0.00000 | 0.46667 |
| volume_zscore_20 | (0.118, 0.801] | 210 | 0.00143 | 0.00266 | 0.53810 |
| volume_zscore_20 | (0.801, 3.995] | 210 | 0.00290 | 0.00202 | 0.54286 |

return_lag_2: train -0.062, validation -0.004, đảo dấu giữa năm 0 lần; volume_ratio_20: train +0.062, validation +0.047, đảo dấu giữa năm 2 lần; volume_change_1: train +0.058, validation -0.016, đảo dấu giữa năm 1 lần; momentum_21: train +0.056, validation +0.032, đảo dấu giữa năm 3 lần; macd_ratio: train +0.055, validation +0.044, đảo dấu giữa năm 4 lần. Đổi dấu hoặc một năm chi phối là khả năng phụ thuộc chế độ; một validation năm không chứng minh hiệu quả 2022–2026.

| year | feature | n | ic |
| --- | --- | --- | --- |
| 2016 | return_lag_2 | 52 | -0.16502 |
| 2016 | volume_ratio_20 | 52 | -0.03511 |
| 2016 | volume_change_1 | 52 | -0.08917 |
| 2016 | momentum_21 | 52 | -0.10709 |
| 2016 | macd_ratio | 52 | 0.00009 |
| 2017 | return_lag_2 | 250 | -0.03916 |
| 2017 | volume_ratio_20 | 250 | 0.01921 |
| 2017 | volume_change_1 | 250 | 0.06866 |
| 2017 | momentum_21 | 250 | -0.00723 |
| 2017 | macd_ratio | 250 | -0.06714 |
| 2018 | return_lag_2 | 248 | -0.06188 |
| 2018 | volume_ratio_20 | 248 | 0.15573 |
| 2018 | volume_change_1 | 248 | 0.09246 |
| 2018 | momentum_21 | 248 | 0.04025 |
| 2018 | macd_ratio | 248 | 0.03899 |
| 2019 | return_lag_2 | 250 | -0.09317 |
| 2019 | volume_ratio_20 | 250 | 0.05846 |
| 2019 | volume_change_1 | 250 | 0.06230 |
| 2019 | momentum_21 | 250 | -0.02091 |
| 2019 | macd_ratio | 250 | -0.03045 |
| 2020 | return_lag_2 | 251 | -0.08022 |
| 2020 | volume_ratio_20 | 251 | -0.01886 |
| 2020 | volume_change_1 | 251 | 0.02264 |
| 2020 | momentum_21 | 251 | 0.04537 |
| 2020 | macd_ratio | 251 | 0.03846 |

## 6. Độ nhạy và proxy giao dịch

Trong 5 feature được kiểm tra, |Δ Spearman IC| lớn nhất sau loại hai đuôi train là volume_ratio_20: +0.0618 → +0.0532, n 1051 → 1029. Pearson có thể thay đổi khác Spearman. Kết quả không chứng minh điểm cực đoan là lỗi; cần thử robust preprocessing fit riêng trên train trong Phase 3.

| feature | p1 | p99 | n_full | n_trim | pearson_full | pearson_trim | ic_full | ic_trim | delta_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volume_change_1 | -0.65691 | 2.31100 | 1051 | 1029 | 0.04236 | 0.04685 | 0.05750 | 0.06167 | 0.00417 |
| volume_zscore_20 | -1.57750 | 3.10328 | 1051 | 1029 | 0.04351 | 0.04365 | 0.05280 | 0.04647 | -0.00633 |
| volume_ratio_20 | 0.35706 | 2.54425 | 1051 | 1029 | 0.04721 | 0.03923 | 0.06182 | 0.05323 | -0.00859 |
| log_return | -0.05516 | 0.05329 | 1051 | 1029 | -0.00292 | 0.00641 | -0.00841 | -0.00532 | 0.00308 |
| volatility_21 | 0.15197 | 0.58771 | 1051 | 1029 | 0.01528 | 0.01241 | 0.00253 | 0.00438 | 0.00185 |

Train net proxy trung bình -0.00064, median -0.00085; validation trung bình +0.00044. |RQ2 IC| lớn nhất mô tả thuộc momentum_21: +0.0779; không dùng ranking này để xây chiến lược. Close–Close một phiên và Open–Open nhiều phiên có rủi ro, chi phí và horizon khác; return chồng lấn không phải PnL thực hiện.

| index | gross_mean | net_mean | net_median |
| --- | --- | --- | --- |
| Train | 0.00437 | -0.00064 | -0.00085 |
| Validation | 0.00546 | 0.00044 | 0.00364 |

Các proxy chồng lấn và không phải chuỗi PnL danh mục. Không suy ra Sharpe hoặc lợi nhuận thực tế.

## 7. Giới hạn và bước tiếp theo

Baseline giữ đủ 26 features. Đã kiểm tra validation cho 5 giả thuyết chọn từ train, 4/5 giữ cùng dấu; đó không phải kết luận về significance hay test. Cần train-only scaling, robust preprocessing có kiểm soát, walk-forward, purge theo label, calibration xác suất và so sánh baseline; kiểm tra nguồn volume và giá adjusted trước khi suy diễn kết quả.

Chỉ có một mã và một năm validation; nhiều phép thử, serial dependence và non-stationarity chưa được xử lý về mặt thống kê. Calendar chính thức, corporate actions và giá point-in-time chưa được xác minh độc lập đầy đủ; hai ngày bị loại chưa được đối chiếu lịch HOSE chính thức. Không tự sửa hoặc xóa volume bất thường mà chưa điều tra. Phí, thuế và slippage là giả định.

Phase 3 cần xác định target, dùng baseline đơn giản, chia theo thời gian, purge theo horizon và fit preprocessing chỉ trên train. Không sử dụng kết quả này để kết luận alpha hay khả năng thực thi.
