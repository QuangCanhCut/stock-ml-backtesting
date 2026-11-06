# Phase 2.4 — Báo cáo phân tích financial features HPG

Nguồn: notebook `notebooks/02_feature_engineering_labeling.ipynb` đã chạy, cùng processed Parquet. Số liệu dưới đây trích từ outputs cuối cùng; IC là time-series một mã, không phải kiểm định alpha.

## 1. Executive Summary

Master có 2,639 hàng, 37 cột, gồm 5 OHLCV, 26 features và 6 cột forward/target; index từ 2016-01-04 đến 2026-10-30, duy nhất và tăng. Dtype nhãn nullable Int8, không có Inf trong features. Điều này kiểm tra cấu trúc, không chứng minh chất lượng nguồn hay tính point-in-time.

Ba |IC| lớn nhất trên train: momentum_21 +0.0597 (n=1023); volume_zscore_20 +0.0546 (n=1023); macd_ratio +0.0541 (n=1023). Đây là thứ tự mô tả, không phải các feature đã chứng minh dự báo: 26 phép thử, phụ thuộc thời gian và nhiễu chọn mẫu chưa được hiệu chỉnh. Chỉ 5 feature đầu theo train được đóng băng để đối chiếu validation. Không dùng test target.

momentum_21: train +0.060, validation +0.032, đảo dấu giữa năm 3 lần; volume_zscore_20: train +0.055, validation +0.010, đảo dấu giữa năm 2 lần; macd_ratio: train +0.054, validation +0.041, đảo dấu giữa năm 3 lần; volume_ratio_20: train +0.053, validation +0.004, đảo dấu giữa năm 2 lần; sma_ratio_200: train +0.050, validation +0.023, đảo dấu giữa năm 3 lần. Đổi dấu hoặc một năm chi phối là khả năng phụ thuộc chế độ; một validation năm không chứng minh hiệu quả 2022–2026.

Kết quả là mô tả, chưa ước lượng uncertainty có xét phụ thuộc, chưa đánh giá ý nghĩa thống kê; không chứng minh hiệu quả giao dịch.

## 2. Data & Methodology

Train 2016-01-04–2020-12-31 có 1223 phiên; validation 2021-01-04–2021-12-31 có 224; test giữ kín 2022-01-04–2026-10-30 có 1192. Sau warm-up và purge, RQ1 còn 1023/223 và RQ2 1020/220 hàng train/validation. Số bị purge theo ranh giới lần lượt RQ1 1/1, RQ2 4/4. Không random shuffle; tính theo phiên thực, không phải ngày lịch.

| Giai đoạn | Đầu | Cuối | Rows | Đủ feature | RQ1 eligible | RQ2 eligible | Purge RQ1 | Purge RQ2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Train | 2016-01-04 | 2020-12-31 | 1223 | 1024 | 1023 | 1020 | 1 | 4 |
| Validation | 2021-01-04 | 2021-12-31 | 224 | 224 | 223 | 220 | 1 | 4 |
| Reserved test | 2022-01-04 | 2026-10-30 | 1192 | không đánh giá | kín | kín | — | — |

Schema từ metadata được đối chiếu config; tính missing và phân phối feature trên các giai đoạn được phép, phân tích correlation, Pearson và Spearman với return tương lai, quantile và sensitivity trên train. Validation chỉ kiểm tra 5 feature đã chọn từ train. RQ1 purge nhãn kết thúc từ phiên validation/test đầu trở đi; RQ2 purge theo t+4. Không áp dụng thêm embargo 5 phiên; purge và embargo là hai khái niệm khác nhau. Không xem hay tổng hợp giá trị target test. Không thực hiện production walk-forward.

## 3. Data Quality Findings

Trong train+validation, sma_ratio_200 thiếu nhiều nhất: 199 hàng (13.75%), chủ yếu do warm-up; tổng missing sau lần hợp lệ đầu là 0. Volume 04/10/2021 = 7,610, phiên sau = 50,156,363, volume_change_1 = 6,589.85 lần, do mẫu số rất nhỏ. Đây là quan sát validation, không dùng để điều chỉnh giả thuyết train. Quan sát có thể ảnh hưởng rolling volume đến khi rời cửa sổ và Pearson; cần đối chiếu corporate actions/nguồn, không tự kết luận lỗi hoặc xóa. Giá điều chỉnh hồi tố có thể không point-in-time. Logistic Regression nhạy scale/cực trị; cây cũng có thể tạo split dựa trên số ít điểm.

Đầu vào phải có OHLC dương và hình học hợp lệ, volume không âm; chỉ kiểm tra Inf trên các feature và nhãn trong phần nghiên cứu được phép. Dữ liệu adjusted và quy ước volume cần xác minh từ nguồn. Warm-up không đồng nghĩa lỗi; tỷ lệ không xác định có thể phát sinh khi mẫu số bằng 0.

## 4. Feature Distribution Findings

Returns: log_return median 0.00110, P1/P99 -0.0527/0.0529, skew 0.04, excess kurtosis 1.93; đuôi và bất đối xứng là mô tả mẫu. Trend: sma_ratio_20 median 0.0155, min/max -0.178/0.233; horizon dài có thể phản ứng chậm hơn. Momentum: RSI median 55.5, P25–P75 45.0–65.4; std momentum_5/21 = 0.045/0.101. Volatility: volatility_21 median/max 0.278/0.641, đỉnh ngày 2020-04-07; volatility_63 median/max 0.287/0.553. Volume: volume_change_1 max 68.977, P99 2.637; volume_zscore_20 max 4.246. Biểu đồ cắt phạm vi hiển thị, không có nghĩa các cực trị đã bị loại khỏi thống kê.

Biểu đồ histogram và boxplot chỉ hiển thị P1–P99 để đọc phần trung tâm; tất cả thống kê bảng dùng dữ liệu chưa cắt. Trường hợp volume lớn ở validation không được dùng để điều chỉnh quy tắc train.

## 5. Correlation & Redundancy

Có 19 cặp đạt |Pearson| hoặc |Spearman| ≥ 0,90 trên train. Cặp đầu theo |Pearson|: sma_ratio_10–ema_ratio_12, Pearson 0.969, Spearman 0.961. volatility_21–volatility_63: Pearson 0.740, Spearman 0.680. SMA/EMA và momentum dùng lịch sử giá chồng lấn nên dễ đồng biến. Tương quan không phải predictive power; đa cộng tuyến có thể làm hệ số Logistic Regression thiếu ổn định, còn cây thường phân chia tầm quan trọng giữa biến tương tự. Không loại feature ở đây.

Các cặp ≥ 0,90 trong ít nhất một trong hai thước đo:

| A | B | Pearson | Spearman | Nhận xét |
| --- | --- | --- | --- | --- |
| sma_ratio_10 | ema_ratio_12 | 0.96943 | 0.96090 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | ema_ratio_26 | 0.96475 | 0.95884 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | ema_ratio_12 | 0.96140 | 0.95645 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_50 | macd_ratio | 0.96119 | 0.97089 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| ema_ratio_26 | rsi_14 | 0.94799 | 0.97567 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_50 | ema_ratio_26 | 0.93927 | 0.93026 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| ema_ratio_12 | ema_ratio_26 | 0.93241 | 0.91936 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | momentum_10 | 0.92941 | 0.91983 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_10 | momentum_5 | 0.92864 | 0.91566 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| ema_ratio_26 | momentum_21 | 0.91536 | 0.90673 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | rsi_14 | 0.90736 | 0.93535 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_50 | momentum_21 | 0.90605 | 0.89215 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| macd_ratio | momentum_21 | 0.90471 | 0.89286 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| rsi_14 | bollinger_percent_b_20 | 0.90173 | 0.91080 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| ema_ratio_12 | momentum_10 | 0.90137 | 0.89073 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| ema_ratio_12 | bollinger_percent_b_20 | 0.89038 | 0.94425 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_50 | rsi_14 | 0.88812 | 0.91332 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| sma_ratio_20 | bollinger_percent_b_20 | 0.88114 | 0.94232 | Có thể trùng thông tin; cần kiểm tra bổ sung |
| volume_zscore_20 | volume_ratio_20 | 0.75270 | 0.96523 | Có thể trùng thông tin; cần kiểm tra bổ sung |

## 6. Information Coefficient Results

Ba |IC| lớn nhất trên train: momentum_21 +0.0597 (n=1023); volume_zscore_20 +0.0546 (n=1023); macd_ratio +0.0541 (n=1023). Đây là thứ tự mô tả, không phải các feature đã chứng minh dự báo: 26 phép thử, phụ thuộc thời gian và nhiễu chọn mẫu chưa được hiệu chỉnh. Chỉ 5 feature đầu theo train được đóng băng để đối chiếu validation. Không dùng test target.

Bảng xếp theo |train IC|; validation chỉ tính cho năm feature đã chọn trên train, dấu — nghĩa là không khảo sát, không phải IC bằng 0. Tất cả dùng complete-feature RQ1 và purge theo thời điểm kết thúc nhãn.

| Feature | Nhóm | N train | Train IC | Validation IC | N validation |
| --- | --- | --- | --- | --- | --- |
| momentum_21 | Momentum | 1023 | 0.05968 | 0.03226 | 223.00000 |
| volume_zscore_20 | Volume | 1023 | 0.05461 | 0.00968 | 223.00000 |
| macd_ratio | Trend | 1023 | 0.05405 | 0.04075 | 223.00000 |
| volume_ratio_20 | Volume | 1023 | 0.05346 | 0.00447 | 223.00000 |
| sma_ratio_200 | Trend | 1023 | 0.05035 | 0.02342 | 223.00000 |
| ema_ratio_26 | Trend | 1023 | 0.04993 | — | — |
| rsi_14 | Momentum | 1023 | 0.04981 | — | — |
| bollinger_percent_b_20 | Volatility | 1023 | 0.04957 | — | — |
| sma_ratio_50 | Trend | 1023 | 0.04888 | — | — |
| return_lag_2 | Returns | 1023 | -0.04885 | — | — |
| sma_ratio_20 | Trend | 1023 | 0.04642 | — | — |
| return_lag_1 | Returns | 1023 | 0.03915 | — | — |
| ema_ratio_12 | Trend | 1023 | 0.03658 | — | — |
| volume_change_1 | Volume | 1023 | 0.03641 | — | — |
| sma_ratio_10 | Trend | 1023 | 0.03222 | — | — |
| return_lag_10 | Returns | 1023 | 0.02868 | — | — |
| momentum_5 | Momentum | 1023 | 0.02547 | — | — |
| return_lag_21 | Returns | 1023 | 0.02252 | — | — |
| return_lag_3 | Returns | 1023 | 0.02029 | — | — |
| bollinger_width_20 | Volatility | 1023 | 0.02005 | — | — |
| momentum_10 | Momentum | 1023 | 0.01772 | — | — |
| volatility_63 | Volatility | 1023 | 0.01759 | — | — |
| log_return | Returns | 1023 | -0.01610 | — | — |
| atr_ratio_14 | Volatility | 1023 | 0.01336 | — | — |
| return_lag_5 | Returns | 1023 | -0.01204 | — | — |
| volatility_21 | Volatility | 1023 | 0.00090 | — | — |

## 7. Conditional Return Analysis

log_return: Q thấp/cao mean +0.00279/+0.00133, positive 57.6%/47.3%, không đơn điệu; sma_ratio_20: Q thấp/cao mean +0.00067/+0.00304, positive 48.3%/52.2%, không đơn điệu; momentum_5: Q thấp/cao mean +0.00062/+0.00156, positive 51.7%/49.3%, không đơn điệu; volatility_21: Q thấp/cao mean +0.00142/+0.00275, positive 50.2%/52.7%, không đơn điệu; volume_zscore_20: Q thấp/cao mean +0.00066/+0.00325, positive 46.3%/55.6%, không đơn điệu. Đây là khác biệt mô tả của các bin, có thể do nhiễu hoặc outlier; không suy ra quan hệ phi tuyến đã được kiểm chứng hay đặt ngưỡng giao dịch.

Các quantile được tính riêng từng feature, `qcut` cho phép giảm số bin nếu các cạnh trùng. Bảng đầy đủ mean, median, positive rate và n được lưu trong output notebook. Không đưa ra trading threshold.

## 8. Temporal Stability

momentum_21: train +0.060, validation +0.032, đảo dấu giữa năm 3 lần; volume_zscore_20: train +0.055, validation +0.010, đảo dấu giữa năm 2 lần; macd_ratio: train +0.054, validation +0.041, đảo dấu giữa năm 3 lần; volume_ratio_20: train +0.053, validation +0.004, đảo dấu giữa năm 2 lần; sma_ratio_200: train +0.050, validation +0.023, đảo dấu giữa năm 3 lần. Đổi dấu hoặc một năm chi phối là khả năng phụ thuộc chế độ; một validation năm không chứng minh hiệu quả 2022–2026.

IC theo năm trên train:

| feature | macd_ratio | momentum_21 | sma_ratio_200 | volume_ratio_20 | volume_zscore_20 |
| --- | --- | --- | --- | --- | --- |
| 2016 | -0.02090 | -0.13030 | -0.35320 | -0.04550 | -0.08230 |
| 2017 | -0.06730 | -0.00780 | -0.03990 | -0.01530 | 0.03000 |
| 2018 | 0.04830 | 0.05480 | 0.04820 | 0.17070 | 0.16190 |
| 2019 | -0.03040 | -0.02010 | -0.06350 | 0.07260 | 0.05920 |
| 2020 | 0.02360 | 0.03790 | 0.01570 | -0.03840 | -0.04700 |

Năm 2016 chỉ còn 51 hàng đủ 26 feature, do đó ước lượng năm này rất nhiễu. Validation 2021 đã được dùng để kiểm tra giả thuyết, không được tiếp tục khai thác như test độc lập.

## 9. Outlier Sensitivity

Trong 5 feature được kiểm tra, |Δ Spearman IC| lớn nhất sau loại hai đuôi train là volume_zscore_20: +0.0546 → +0.0443, n 1023 → 1001. Pearson có thể thay đổi khác Spearman. Kết quả không chứng minh điểm cực đoan là lỗi; cần thử robust preprocessing fit riêng trên train trong Phase 3.

| feature | p1 | p99 | n_full | n_trim | pearson_full | pearson_trim | ic_full | ic_trim | delta_ic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volume_change_1 | -0.65864 | 2.57774 | 1023 | 1001 | 0.01453 | 0.02011 | 0.03641 | 0.03702 | 0.00061 |
| volume_zscore_20 | -1.57229 | 3.16146 | 1023 | 1001 | 0.04109 | 0.03698 | 0.05461 | 0.04431 | -0.01030 |
| volume_ratio_20 | 0.15337 | 2.62959 | 1023 | 1001 | 0.03125 | 0.03867 | 0.05346 | 0.05653 | 0.00307 |
| log_return | -0.05538 | 0.05280 | 1023 | 1001 | -0.01190 | 0.00945 | -0.01610 | -0.00947 | 0.00663 |
| volatility_21 | 0.15550 | 0.57730 | 1023 | 1001 | 0.00808 | 0.00641 | 0.00090 | 0.00606 | 0.00515 |

Cắt P1/P99 là kịch bản trên bản sao train, không sửa dataset và không phải preprocessing được chấp thuận. Nó thay đổi cả mẫu, vì thế không chứng minh outlier gây ra association.

## 10. RQ1 vs RQ2 Research Observations

Tỷ lệ tăng RQ1 train/validation: 49.66%/51.57%; tỷ lệ dương RQ2: 48.24%/55.45%. Baseline lớp đa số trong từng mẫu RQ1 là 50.34%/51.57%; không dùng lớp đa số validation để fit mô hình. Hai nhãn đo các khoảng giá khác nhau, nên accuracy cần so với baseline và không suy ra lợi nhuận chiến lược.

Train net proxy trung bình -0.00048, median -0.00118; validation trung bình +0.00119. |RQ2 IC| lớn nhất mô tả thuộc sma_ratio_200: +0.0664; không dùng ranking này để xây chiến lược. Close–Close một phiên và Open–Open nhiều phiên có rủi ro, chi phí và horizon khác; return chồng lấn không phải PnL thực hiện.

Tín hiệu sau Close(t), entry giả định Open(t+1), exit Open(t+4) với holding=3; net đã tính phí theo cấu hình. Các nhãn chồng lấn, không là chuỗi PnL của danh mục và không dùng để tính Sharpe chiến lược.

## 11. Research Limitations

Một cổ phiếu, mẫu và validation hữu hạn; multiple testing, data snooping, serial dependence, non-stationarity và IC không có khoảng tin cậy phù hợp. Giá adjusted có thể hồi tố, corporate actions và volume bất thường cần đối chiếu nguồn. Phí, slippage và thuế là giả định; proxy fixed-hold bỏ qua khả năng thực thi, thanh khoản, và nhãn chồng lấn. Prefix invariance hay unit test không bảo đảm dữ liệu point-in-time hoặc loại bỏ mọi leakage.

## 12. Recommendations for Phase 3

Baseline giữ đủ 26 features. Đã kiểm tra validation cho 5 giả thuyết chọn từ train, 5/5 giữ cùng dấu; đó không phải kết luận về significance hay test. Cần train-only scaling, robust preprocessing có kiểm soát, walk-forward, purge theo label, calibration xác suất và so sánh baseline; kiểm tra nguồn volume và giá adjusted trước khi suy diễn kết quả.

Giữ baseline 26 feature; so sánh với majority-class và các baseline đơn giản được fit trên train. Thiết kế walk-forward có label-aware purge, chỉ fit scaler/robust transform trên train mỗi fold, kiểm tra calibration xác suất, metric dự báo và stability ngoài mẫu. Không điều chỉnh dựa trên test; xử lý volume chỉ sau khi điều tra và xác lập quy trình train-only.

## 13. Phase 2.4 Completion Checklist

- [x] Notebook tạo và chạy hai lượt từ đầu; không exception trong cells.
- [x] 12 biểu đồ mới được lưu và xem tổng thể; biểu đồ phân phối ghi rõ giới hạn P1–P99.
- [x] Schema, thống kê feature, ranh giới train/validation và purge theo từng horizon được kiểm tra.
- [x] Target-based test được giữ ngoài phân tích; không global preprocessing.
- [x] Markdown diễn giải được chèn sau outputs và kiểm tra lại qua lần chạy thứ hai.
- [x] Báo cáo này lấy số liệu từ notebook chạy cuối.
- [ ] Chưa kiểm định significance có xét serial dependence hoặc multiple testing; chưa xác thực độc lập corporate actions/điều chỉnh giá và volume.

## 14. Final Readiness

**READY cho Phase 3 với giới hạn nêu trên.** Hoàn tất phân tích mô tả và bảo vệ target test trong phạm vi notebook; độ lớn IC nhỏ hoặc không ổn định không ngăn việc chạy kiểm định nghiêm túc ngoài mẫu. Trạng thái này không bảo đảm khả năng dự báo, chất lượng dữ liệu point-in-time hoặc lợi nhuận.
