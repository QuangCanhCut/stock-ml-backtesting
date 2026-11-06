# HƯỚNG DẪN CHI TIẾT FEATURE ENGINEERING CHO DỰ ÁN DỰ BÁO GIÁ CỔ PHIẾU HPG

Financial Time Series — Technical Indicators — Machine Learning Features

Stock Movement Prediction & ML-Based Trading Strategy Backtesting  
Phase 2.1 — Feature Engineering  
Target Role: AI Quantitative Researcher  
Language: Vietnamese

Tài liệu tự học dựa trên mã nguồn và cấu hình hiện tại. Các ví dụ số là minh họa, trừ khi nêu rõ là dữ liệu HPG. Feature mô tả thị trường không tự chứng minh khả năng dự báo hay lợi nhuận.

## Mục lục

1. Feature Engineering trong Financial Machine Learning

2. Tổng quan 26 features

3. Returns và Lagged Returns

4. Trend Features

5. Momentum Features

6. Volatility Features

7. Volume Features

8. So sánh các features

9. NaN, warm-up và numerical stability

10. Data leakage và causal feature design

11. Features trong Machine Learning

12. Cách diễn giải giá trị feature

13. Câu hỏi ôn tập

Mục lục chương được liệt kê cố định. Trong VS Code, mở Outline hoặc Markdown Preview để duyệt các chương và feature.

## 1. Feature Engineering trong Financial Machine Learning

### 1.1. Feature, dữ liệu và mục tiêu

OHLCV thô gồm giá mở cửa, cao nhất, thấp nhất, đóng cửa và khối lượng. Engineered feature là phép biến đổi có quy tắc của lịch sử này. Target label là biến cần dự báo trong tương lai. Ma trận X chứa các feature của từng phiên; vector y chứa nhãn tương ứng; prediction là đầu ra mô hình, chẳng hạn xác suất tăng, không phải chính nhãn thực tế.

### 1.2. Tại sao không chỉ dùng Close?

Giá tuyệt đối phụ thuộc thang giá và có thể bị điều chỉnh bởi corporate actions. Return thể hiện thay đổi tương đối; chỉ báo tóm tắt lịch sử giá hoặc volume; volatility mô tả trạng thái dao động. Những mô tả hữu ích không đảm bảo predictive power: cần kiểm tra ngoài mẫu, tính ổn định và chi phí.

### 1.3. Thời điểm thông tin

Close(t−1) → Close(t) → Prediction(t) → Future(t+1). X_t chỉ được xác định sau khi phiên t kết thúc. Direction label dùng Close(t+1), chưa biết tại t. Một backtest phải xác định riêng thời điểm, giá thực thi, chi phí và tính khả thi; không mặc nhiên giao dịch tại Close(t) sau khi biết Close(t).

### 1.4. Năm nhóm

Returns và lag mô tả thay đổi từng phiên và lịch sử trễ; trend mô tả vị trí so với đường trung bình; momentum tóm tắt thay đổi tích lũy hoặc cân bằng tăng giảm; volatility đo biên độ rủi ro; volume mô tả hoạt động giao dịch. Mỗi nhóm phản ứng khác nhau, có độ trễ, nhiễu và có thể trùng thông tin.

## 2. Tổng quan 26 features

Ký hiệu: O_t, H_t, L_t, C_t, V_t là Open, High, Low, Close, Volume tại phiên t; r_t là log return ngày; n là số phiên trong cửa sổ; σ là độ lệch chuẩn; t−k là k quan sát giao dịch trước, không nhất thiết là k ngày lịch. Return ngày mô tả một khoảng; return tích lũy mô tả nhiều khoảng. Volatility năm hóa là quy đổi quy ước, không phải return kỳ vọng.

| **Feature** | **Nhóm** | **Công thức** | **Phiên** | **Đơn vị** | **Mục đích** |
| --- | --- | --- | --- | --- | --- |
| log_return | Return | ln(C_t/C_t−1) | 1 | tỷ lệ log | Thay đổi phiên |
| return_lag_1 | Return | r(t−1) | 1 | tỷ lệ log | Lịch sử |
| return_lag_2 | Return | r(t−2) | 2 | tỷ lệ log | Lịch sử |
| return_lag_3 | Return | r(t−3) | 3 | tỷ lệ log | Lịch sử |
| return_lag_5 | Return | r(t−5) | 5 | tỷ lệ log | Lịch sử |
| return_lag_10 | Return | r(t−10) | 10 | tỷ lệ log | Lịch sử |
| return_lag_21 | Return | r(t−21) | 21 | tỷ lệ log | Lịch sử |
| sma_ratio_10 | Trend | C/SMA−1 | 10 | tỷ lệ | Vị trí giá |
| sma_ratio_20 | Trend | C/SMA−1 | 20 | tỷ lệ | Vị trí giá |
| sma_ratio_50 | Trend | C/SMA−1 | 50 | tỷ lệ | Vị trí giá |
| sma_ratio_200 | Trend | C/SMA−1 | 200 | tỷ lệ | Vị trí giá |
| ema_ratio_12 | Trend | C/EMA−1 | 12 | tỷ lệ | Vị trí giá |
| ema_ratio_26 | Trend | C/EMA−1 | 26 | tỷ lệ | Vị trí giá |
| macd_ratio | Trend | (EMA12−EMA26)/C | 12,26 | tỷ lệ | Chênh lệch xu hướng |
| momentum_5 | Momentum | ln(C_t/C_t−5) | 5 | tỷ lệ log | Thay đổi tích lũy |
| momentum_10 | Momentum | ln(C_t/C_t−10) | 10 | tỷ lệ log | Thay đổi tích lũy |
| momentum_21 | Momentum | ln(C_t/C_t−21) | 21 | tỷ lệ log | Thay đổi tích lũy |
| rsi_14 | Momentum | 100G/(G+L) | 14 | 0–100 | Cân bằng tăng giảm |
| volatility_21 | Volatility | std(r)√252 | 21 | tỷ lệ/năm | Dao động |
| volatility_63 | Volatility | std(r)√252 | 63 | tỷ lệ/năm | Dao động |
| atr_ratio_14 | Volatility | ATR/C | 14 | tỷ lệ | Biên độ |
| bollinger_width_20 | Volatility | (U−L)/M | 20 | tỷ lệ | Độ rộng dải |
| bollinger_percent_b_20 | Volatility | (C−L)/(U−L) | 20 | tỷ lệ | Vị trí trong dải |
| volume_zscore_20 | Volume | (V−μ)/s | 20 | độ lệch chuẩn | Bất thường |
| volume_change_1 | Volume | V/Vtrước−1 | 1 | tỷ lệ | Thay đổi |
| volume_ratio_20 | Volume | V/μ | 20 | lần | Tương đối |

Inventory trùng expected_feature_columns() theo config hiện tại. Tên thay đổi nếu đổi cửa sổ; momentum_windows có mặc định [5, 10, 21] nếu thiếu. Mã dùng 252 và hệ số Bollinger 2 theo config; test dùng cửa sổ nhỏ riêng để kiểm thử.

## 3. Returns và Lagged Returns

Simple return = C_t/C_t−1 − 1; log return = ln(1 + simple return). Với 25.000 → 25.500: simple return = 0,02 = 2%; log return = ln(1,02) ≈ 0,019803. Log return cộng được qua thời gian: tổng r bằng ln(C cuối/C đầu).

#### log_return

**Tên feature:** log_return

**Nhóm:** Return

**Công thức:** r_t = ln(C_t/C_t−1)

**Ký hiệu:** C là Close của phiên tương ứng.

**Cách tính:** Chia Close hiện tại cho Close trước rồi lấy log tự nhiên.

**Ví dụ số:** 25.500/25.000=1,02; ln(1,02)=0,019803.

**Ý nghĩa tài chính:** Tốc độ thay đổi log giữa hai phiên.

**Dấu và mức 0:** Dương: tăng; âm: giảm; 0: không đổi.

**Giới hạn và hiểu nhầm:** 0,019803 không chính xác là simple return 1,9803%; simple return ở đây là 2%.

**Tham chiếu Python:** add_return_features: np.log(out["close"] / out["close"].shift(1))

Bảng thời gian minh họa các return đã tính:

| **Phiên** | **t−5** | **t−4** | **t−3** | **t−2** | **t−1** | **t** |
| --- | --- | --- | --- | --- | --- | --- |
| r | 0,01 | −0,02 | 0,03 | 0,00 | 0,01 | 0,02 |

#### return_lag_1

**Tên feature:** return_lag_1

**Nhóm:** Return

**Công thức:** r_(t−1)

**Ký hiệu:** r là log return một phiên; 1 là số phiên dịch lùi.

**Cách tính:** Tính r rồi lấy giá trị tại t−1 bằng shift(1).

**Ví dụ số:** Nếu r_(t−1)=0,01 thì feature = 0,01.

**Ý nghĩa tài chính:** Quan sát biến động một phiên cách đây 1 phiên, có thể nghiên cứu liên hệ với momentum hoặc mean reversion.

**Dấu và mức 0:** Dương/âm/0 biểu thị tăng/giảm/đứng tại phiên trễ.

**Giới hạn và hiểu nhầm:** Không phải tổng return của 1 phiên; không đảm bảo dự báo.

**Tham chiếu Python:** add_return_features: out["log_return"].shift(1)

#### return_lag_2

**Tên feature:** return_lag_2

**Nhóm:** Return

**Công thức:** r_(t−2)

**Ký hiệu:** r là log return một phiên; 2 là số phiên dịch lùi.

**Cách tính:** Tính r rồi lấy giá trị tại t−2 bằng shift(2).

**Ví dụ số:** Nếu r_(t−2)=0,01 thì feature = 0,01.

**Ý nghĩa tài chính:** Quan sát biến động một phiên cách đây 2 phiên, có thể nghiên cứu liên hệ với momentum hoặc mean reversion.

**Dấu và mức 0:** Dương/âm/0 biểu thị tăng/giảm/đứng tại phiên trễ.

**Giới hạn và hiểu nhầm:** Không phải tổng return của 2 phiên; không đảm bảo dự báo.

**Tham chiếu Python:** add_return_features: out["log_return"].shift(2)

#### return_lag_3

**Tên feature:** return_lag_3

**Nhóm:** Return

**Công thức:** r_(t−3)

**Ký hiệu:** r là log return một phiên; 3 là số phiên dịch lùi.

**Cách tính:** Tính r rồi lấy giá trị tại t−3 bằng shift(3).

**Ví dụ số:** Nếu r_(t−3)=0,01 thì feature = 0,01.

**Ý nghĩa tài chính:** Quan sát biến động một phiên cách đây 3 phiên, có thể nghiên cứu liên hệ với momentum hoặc mean reversion.

**Dấu và mức 0:** Dương/âm/0 biểu thị tăng/giảm/đứng tại phiên trễ.

**Giới hạn và hiểu nhầm:** Không phải tổng return của 3 phiên; không đảm bảo dự báo.

**Tham chiếu Python:** add_return_features: out["log_return"].shift(3)

#### return_lag_5

**Tên feature:** return_lag_5

**Nhóm:** Return

**Công thức:** r_(t−5)

**Ký hiệu:** r là log return một phiên; 5 là số phiên dịch lùi.

**Cách tính:** Tính r rồi lấy giá trị tại t−5 bằng shift(5).

**Ví dụ số:** Nếu r_(t−5)=0,01 thì feature = 0,01.

**Ý nghĩa tài chính:** Quan sát biến động một phiên cách đây 5 phiên, có thể nghiên cứu liên hệ với momentum hoặc mean reversion.

**Dấu và mức 0:** Dương/âm/0 biểu thị tăng/giảm/đứng tại phiên trễ.

**Giới hạn và hiểu nhầm:** Không phải tổng return của 5 phiên; không đảm bảo dự báo.

**Tham chiếu Python:** add_return_features: out["log_return"].shift(5)

#### return_lag_10

**Tên feature:** return_lag_10

**Nhóm:** Return

**Công thức:** r_(t−10)

**Ký hiệu:** r là log return một phiên; 10 là số phiên dịch lùi.

**Cách tính:** Tính r rồi lấy giá trị tại t−10 bằng shift(10).

**Ví dụ số:** Nếu r_(t−10)=0,01 thì feature = 0,01.

**Ý nghĩa tài chính:** Quan sát biến động một phiên cách đây 10 phiên, có thể nghiên cứu liên hệ với momentum hoặc mean reversion.

**Dấu và mức 0:** Dương/âm/0 biểu thị tăng/giảm/đứng tại phiên trễ.

**Giới hạn và hiểu nhầm:** Không phải tổng return của 10 phiên; không đảm bảo dự báo.

**Tham chiếu Python:** add_return_features: out["log_return"].shift(10)

#### return_lag_21

**Tên feature:** return_lag_21

**Nhóm:** Return

**Công thức:** r_(t−21)

**Ký hiệu:** r là log return một phiên; 21 là số phiên dịch lùi.

**Cách tính:** Tính r rồi lấy giá trị tại t−21 bằng shift(21).

**Ví dụ số:** Nếu r_(t−21)=0,01 thì feature = 0,01.

**Ý nghĩa tài chính:** Quan sát biến động một phiên cách đây 21 phiên, có thể nghiên cứu liên hệ với momentum hoặc mean reversion.

**Dấu và mức 0:** Dương/âm/0 biểu thị tăng/giảm/đứng tại phiên trễ.

**Giới hạn và hiểu nhầm:** Không phải tổng return của 21 phiên; không đảm bảo dự báo.

**Tham chiếu Python:** add_return_features: out["log_return"].shift(21)

## 4. Trend Features

SMA_n(t) = tổng n Close gần nhất, gồm t, chia n. Tỷ lệ C/SMA−1 đưa khoảng cách về thang tương đối, giúp so sánh các mức giá khác nhau. SMA10 ngắn và nhạy; SMA20 khoảng tháng; SMA50 trung hạn; SMA200 dài hạn. Horizons là số phiên, không phải ngày lịch.

#### sma_ratio_10

**Tên feature:** sma_ratio_10

**Nhóm:** Trend

**Công thức:** C_t / SMA_10(t) − 1

**Ký hiệu:** SMA là trung bình 10 Close kết thúc tại t.

**Cách tính:** Lấy trung bình 10 Close, chia C_t cho trung bình và trừ 1.

**Ví dụ số:** C=26.000, SMA=25.000: 26.000/25.000−1=0,04.

**Ý nghĩa tài chính:** Close cao hơn trung bình 4%; cửa sổ 10 phiên phản ánh horizon tương ứng.

**Dấu và mức 0:** Dương: trên SMA; 0: bằng; âm: dưới.

**Giới hạn và hiểu nhầm:** Đường dài chậm và các đường có thể trùng thông tin; không phải tín hiệu chắc chắn.

**Tham chiếu Python:** add_trend_features: close / close.rolling(10, min_periods=10).mean() − 1

#### sma_ratio_20

**Tên feature:** sma_ratio_20

**Nhóm:** Trend

**Công thức:** C_t / SMA_20(t) − 1

**Ký hiệu:** SMA là trung bình 20 Close kết thúc tại t.

**Cách tính:** Lấy trung bình 20 Close, chia C_t cho trung bình và trừ 1.

**Ví dụ số:** C=26.000, SMA=25.000: 26.000/25.000−1=0,04.

**Ý nghĩa tài chính:** Close cao hơn trung bình 4%; cửa sổ 20 phiên phản ánh horizon tương ứng.

**Dấu và mức 0:** Dương: trên SMA; 0: bằng; âm: dưới.

**Giới hạn và hiểu nhầm:** Đường dài chậm và các đường có thể trùng thông tin; không phải tín hiệu chắc chắn.

**Tham chiếu Python:** add_trend_features: close / close.rolling(20, min_periods=20).mean() − 1

#### sma_ratio_50

**Tên feature:** sma_ratio_50

**Nhóm:** Trend

**Công thức:** C_t / SMA_50(t) − 1

**Ký hiệu:** SMA là trung bình 50 Close kết thúc tại t.

**Cách tính:** Lấy trung bình 50 Close, chia C_t cho trung bình và trừ 1.

**Ví dụ số:** C=26.000, SMA=25.000: 26.000/25.000−1=0,04.

**Ý nghĩa tài chính:** Close cao hơn trung bình 4%; cửa sổ 50 phiên phản ánh horizon tương ứng.

**Dấu và mức 0:** Dương: trên SMA; 0: bằng; âm: dưới.

**Giới hạn và hiểu nhầm:** Đường dài chậm và các đường có thể trùng thông tin; không phải tín hiệu chắc chắn.

**Tham chiếu Python:** add_trend_features: close / close.rolling(50, min_periods=50).mean() − 1

#### sma_ratio_200

**Tên feature:** sma_ratio_200

**Nhóm:** Trend

**Công thức:** C_t / SMA_200(t) − 1

**Ký hiệu:** SMA là trung bình 200 Close kết thúc tại t.

**Cách tính:** Lấy trung bình 200 Close, chia C_t cho trung bình và trừ 1.

**Ví dụ số:** C=26.000, SMA=25.000: 26.000/25.000−1=0,04.

**Ý nghĩa tài chính:** Close cao hơn trung bình 4%; cửa sổ 200 phiên phản ánh horizon tương ứng.

**Dấu và mức 0:** Dương: trên SMA; 0: bằng; âm: dưới.

**Giới hạn và hiểu nhầm:** Đường dài chậm và các đường có thể trùng thông tin; không phải tín hiệu chắc chắn.

**Tham chiếu Python:** add_trend_features: close / close.rolling(200, min_periods=200).mean() − 1

EMA đệ quy với α=2/(n+1): E_0=C_0; E_t=αC_t+(1−α)E_(t−1). Pandas ewm(span=n, adjust=False, min_periods=n) bắt đầu đệ quy từ Close đầu tiên, không seed bằng SMA; min_periods chỉ che kết quả đến khi đủ n quan sát. EMA đặt trọng số lớn hơn vào giá gần đây và thường nhạy hơn SMA cùng span; độ nhạy có thể làm tăng nhiễu.

#### ema_ratio_12

**Tên feature:** ema_ratio_12

**Nhóm:** Trend

**Công thức:** C_t / EMA_12(t) − 1

**Ký hiệu:** EMA có α=2/(12+1), seed từ Close đầu.

**Cách tính:** Cập nhật EMA đệ quy, sau đủ 12 giá chia C cho EMA và trừ 1.

**Ví dụ số:** Nếu EMA=25.000, C=26.000: tỷ lệ = 0,04.

**Ý nghĩa tài chính:** Vị trí Close so với EMA 12; cửa sổ ngắn 12 phản ứng nhanh hơn 26.

**Dấu và mức 0:** Dương/âm/0 lần lượt là trên/dưới/bằng EMA.

**Giới hạn và hiểu nhầm:** EMA không chỉ dùng đúng n giá gần nhất; quá khứ cũ giảm trọng số theo hàm mũ.

**Tham chiếu Python:** add_trend_features: close.ewm(span=12, adjust=False, min_periods=12).mean()

#### ema_ratio_26

**Tên feature:** ema_ratio_26

**Nhóm:** Trend

**Công thức:** C_t / EMA_26(t) − 1

**Ký hiệu:** EMA có α=2/(26+1), seed từ Close đầu.

**Cách tính:** Cập nhật EMA đệ quy, sau đủ 26 giá chia C cho EMA và trừ 1.

**Ví dụ số:** Nếu EMA=25.000, C=26.000: tỷ lệ = 0,04.

**Ý nghĩa tài chính:** Vị trí Close so với EMA 26; cửa sổ ngắn 12 phản ứng nhanh hơn 26.

**Dấu và mức 0:** Dương/âm/0 lần lượt là trên/dưới/bằng EMA.

**Giới hạn và hiểu nhầm:** EMA không chỉ dùng đúng n giá gần nhất; quá khứ cũ giảm trọng số theo hàm mũ.

**Tham chiếu Python:** add_trend_features: close.ewm(span=26, adjust=False, min_periods=26).mean()

#### macd_ratio

**Tên feature:** macd_ratio

**Nhóm:** Trend

**Công thức:** (EMA12 − EMA26) / C_t

**Ký hiệu:** Hai EMA được khởi tạo như trên; C_t là Close.

**Cách tính:** Tính hai EMA, lấy chênh và chia Close.

**Ví dụ số:** EMA12=26.000; EMA26=25.000; C=25.000: 1.000/25.000=0,04.

**Ý nghĩa tài chính:** Khoảng cách chuẩn hóa giữa xu hướng nhanh và chậm.

**Dấu và mức 0:** Dương: EMA nhanh cao hơn chậm; âm: thấp hơn; gần 0: hội tụ.

**Giới hạn và hiểu nhầm:** MACD truyền thống là hiệu giá tuyệt đối; module không tạo signal line hay histogram.

**Tham chiếu Python:** add_trend_features: (ema_values[emas[0]] − ema_values[emas[1]]) / close

## 5. Momentum Features

Momentum_k = ln(C_t/C_(t−k)) = tổng k log return từ t−k+1 đến t. Ví dụ bảy Close tại t−6…t là 100, 102, 101, 103, 104, 102, 105: momentum_5=ln(105/102)=0,028988; return_lag_5=ln(102/100)=0,019803. Một chỉ tiêu tích lũy năm khoảng, một chỉ tiêu là return của phiên t−5.

#### momentum_5

**Tên feature:** momentum_5

**Nhóm:** Momentum

**Công thức:** ln(C_t/C_(t−5))

**Ký hiệu:** C là Close; 5 là số khoảng phiên.

**Cách tính:** Chia Close hiện tại cho Close cách 5 phiên và lấy log.

**Ví dụ số:** Nếu C_t=110 và C_(t−5)=100: ln(1,1)=0,095310.

**Ý nghĩa tài chính:** Thay đổi tích lũy 5 phiên; horizon khác nhau cho góc nhìn khác nhau.

**Dấu và mức 0:** Dương/âm/0: giá cuối cao hơn/thấp hơn/bằng giá đầu.

**Giới hạn và hiểu nhầm:** Không nhầm với return_lag; không đảm bảo xu hướng tiếp diễn.

**Tham chiếu Python:** add_momentum_features: np.log(close / close.shift(5))

#### momentum_10

**Tên feature:** momentum_10

**Nhóm:** Momentum

**Công thức:** ln(C_t/C_(t−10))

**Ký hiệu:** C là Close; 10 là số khoảng phiên.

**Cách tính:** Chia Close hiện tại cho Close cách 10 phiên và lấy log.

**Ví dụ số:** Nếu C_t=110 và C_(t−10)=100: ln(1,1)=0,095310.

**Ý nghĩa tài chính:** Thay đổi tích lũy 10 phiên; horizon khác nhau cho góc nhìn khác nhau.

**Dấu và mức 0:** Dương/âm/0: giá cuối cao hơn/thấp hơn/bằng giá đầu.

**Giới hạn và hiểu nhầm:** Không nhầm với return_lag; không đảm bảo xu hướng tiếp diễn.

**Tham chiếu Python:** add_momentum_features: np.log(close / close.shift(10))

#### momentum_21

**Tên feature:** momentum_21

**Nhóm:** Momentum

**Công thức:** ln(C_t/C_(t−21))

**Ký hiệu:** C là Close; 21 là số khoảng phiên.

**Cách tính:** Chia Close hiện tại cho Close cách 21 phiên và lấy log.

**Ví dụ số:** Nếu C_t=110 và C_(t−21)=100: ln(1,1)=0,095310.

**Ý nghĩa tài chính:** Thay đổi tích lũy 21 phiên; horizon khác nhau cho góc nhìn khác nhau.

**Dấu và mức 0:** Dương/âm/0: giá cuối cao hơn/thấp hơn/bằng giá đầu.

**Giới hạn và hiểu nhầm:** Không nhầm với return_lag; không đảm bảo xu hướng tiếp diễn.

**Tham chiếu Python:** add_momentum_features: np.log(close / close.shift(21))

RSI: ΔC=C_t−C_t−1; gain=max(ΔC,0); loss=max(−ΔC,0). Với n=14, seed G và L bằng trung bình 14 thay đổi đầu; sau đó G_t=((13G_trước)+gain_t)/14, tương tự L. RSI=100G/(G+L), tương đương 100−100/(1+G/L) khi L>0. Mức 30/70 chỉ là ngưỡng truyền thống, không bắt buộc hay chắc chắn cho giao dịch.

#### rsi_14

**Tên feature:** rsi_14

**Nhóm:** Momentum

**Công thức:** 100G/(G+L); nếu G+L=0 thì 50

**Ký hiệu:** G, L là gain và loss đã Wilder smoothing với n=14.

**Cách tính:** Tính 14 ΔC, lấy gain/loss trung bình làm seed, rồi cập nhật Wilder; tính tỷ lệ.

**Ví dụ số:** Ví dụ 14 thay đổi gồm tám lần +2, sáu lần −1: G=16/14, L=6/14, RSI=100×16/22=72,727.

**Ý nghĩa tài chính:** Đo tỷ trọng mức tăng đã làm mượt trong tổng tăng và giảm.

**Dấu và mức 0:** Trong [0,100]: 100 khi chỉ tăng, 0 khi chỉ giảm, 50 nếu cả hai bằng 0.

**Giới hạn và hiểu nhầm:** RSI 75 hay 80 không bắt buộc bán; xu hướng mạnh có thể duy trì. Thiếu 14 thay đổi đầu cho NaN.

**Tham chiếu Python:** add_momentum_features và _wilder: delta.clip; 100 * gain / denominator; mask(denominator == 0, 50)

## 6. Volatility Features

Rolling volatility dùng độ lệch chuẩn mẫu ddof=1 của log return trong cửa sổ kết thúc tại t, nhân √252 theo giả định quy đổi căn thời gian. Ví dụ daily std=0,01 thì năm hóa 0,01√252≈0,158745 hay 15,87%. Đây là quy ước so sánh, không bảo đảm tính độc lập hay ổn định của return.

#### volatility_21

**Tên feature:** volatility_21

**Nhóm:** Volatility

**Công thức:** std_ddof=1(r_(t−21+1),…,r_t) × √252

**Ký hiệu:** r là daily log return; 21 return trong cửa sổ.

**Cách tính:** Tính 21 return, độ lệch chuẩn mẫu và nhân √252.

**Ví dụ số:** Nếu std ngày=0,01 thì annualized=0,158745.

**Ý nghĩa tài chính:** Độ phân tán return năm hóa; 21 phiên phản ứng nhanh hơn 63.

**Dấu và mức 0:** Không âm; 0 khi mọi return trong cửa sổ bằng nhau.

**Giới hạn và hiểu nhầm:** Cao không có nghĩa giá chắc chắn giảm; 252 là quy ước.

**Tham chiếu Python:** add_volatility_features: returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(days)

#### volatility_63

**Tên feature:** volatility_63

**Nhóm:** Volatility

**Công thức:** std_ddof=1(r_(t−63+1),…,r_t) × √252

**Ký hiệu:** r là daily log return; 63 return trong cửa sổ.

**Cách tính:** Tính 63 return, độ lệch chuẩn mẫu và nhân √252.

**Ví dụ số:** Nếu std ngày=0,01 thì annualized=0,158745.

**Ý nghĩa tài chính:** Độ phân tán return năm hóa; 63 phiên mượt hơn 21.

**Dấu và mức 0:** Không âm; 0 khi mọi return trong cửa sổ bằng nhau.

**Giới hạn và hiểu nhầm:** Cao không có nghĩa giá chắc chắn giảm; 252 là quy ước.

**Tham chiếu Python:** add_volatility_features: returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(days)

True Range = max(H−L, |H−C_trước|, |L−C_trước|). Previous Close nắm bắt gap qua đêm mà H−L bỏ sót. TR đầu tiên không có Close trước nên max bỏ NaN và còn H−L. ATR seed bằng trung bình 14 TR đầu, sau đó cập nhật Wilder.

#### atr_ratio_14

**Tên feature:** atr_ratio_14

**Nhóm:** Volatility

**Công thức:** ATR14_t / C_t

**Ký hiệu:** ATR14 là Wilder smoothing của TR; C_t dương.

**Cách tính:** Tính TR, seed 14 TR, cập nhật Wilder, chia Close.

**Ví dụ số:** Nếu H=105, L=99, C_trước=100 thì TR=max(6,5,1)=6; nếu ATR=2,5 và C=100 thì ratio=0,025.

**Ý nghĩa tài chính:** Biên độ giá đã chuẩn hóa, gồm gap, khác độ phân tán close-to-close.

**Dấu và mức 0:** Không âm; 0 khi ATR bằng 0, ví dụ OHLC phẳng đủ dài.

**Giới hạn và hiểu nhầm:** Không annualize; ATR thấp không đảm bảo biến động tương lai thấp.

**Tham chiếu Python:** add_volatility_features: tr = concat([...]).max(axis=1); _wilder(tr, atr_window) / close

Bollinger dùng M=SMA20, σ độ lệch chuẩn tổng thể ddof=0 của 20 Close, U=M+2σ, L=M−2σ; cửa sổ trailing gồm phiên hiện tại. Ví dụ M=100, σ=5, U=110, L=90, C=105: width=20/100=0,2; %B=15/20=0,75. Nếu σ=0 thì width=0 còn %B là NaN.

#### bollinger_width_20

**Tên feature:** bollinger_width_20

**Nhóm:** Volatility

**Công thức:** (U−L)/M = 4σ/M

**Ký hiệu:** M=SMA20; σ=std Close ddof=0; U/L là dải.

**Cách tính:** Tính M và σ của 20 Close, tính chênh hai dải rồi chia M.

**Ví dụ số:** M=100, σ=5: width=20/100=0,20.

**Ý nghĩa tài chính:** Độ rộng dải giá tương đối so với trung bình.

**Dấu và mức 0:** Không âm; bằng 0 khi 20 Close giống nhau.

**Giới hạn và hiểu nhầm:** Độ phân tán mức giá khác volatility của return.

**Tham chiếu Python:** add_volatility_features: band_span = 2 * multiplier * std; band_span / middle

#### bollinger_percent_b_20

**Tên feature:** bollinger_percent_b_20

**Nhóm:** Volatility

**Công thức:** (C−L)/(U−L)

**Ký hiệu:** C là Close, L/U là dải dưới/trên.

**Cách tính:** Trừ dải dưới khỏi C rồi chia độ rộng; nếu độ rộng 0 cho NaN.

**Ví dụ số:** L=90, U=110, C=105: (105−90)/20=0,75.

**Ý nghĩa tài chính:** Vị trí Close so với hai dải.

**Dấu và mức 0:** 0 ở dải dưới; 0,5 ở giữa; 1 ở dải trên; có thể âm hoặc lớn hơn 1.

**Giới hạn và hiểu nhầm:** Không chặn trong [0,1]; vượt dải không chắc đảo chiều.

**Tham chiếu Python:** add_volatility_features: (close − (middle − multiplier * std)) / band_span.where(band_span != 0)

## 7. Volume Features

Rolling mean và sample std (ddof=1) của volume dùng 20 phiên gồm phiên t, vì vậy chính volume đang đo cũng ảnh hưởng đến chuẩn tham chiếu.

#### volume_zscore_20

**Tên feature:** volume_zscore_20

**Nhóm:** Volume

**Công thức:** (V_t−mean20)/std20

**Ký hiệu:** mean20 và std20 tính trên 20 volume, ddof=1.

**Cách tính:** Tính trung bình và sample std; chia độ lệch cho std, nếu std=0 cho NaN.

**Ví dụ số:** V=20 triệu, mean=10 triệu, std=5 triệu: z=2.

**Ý nghĩa tài chính:** V cao hơn trung bình hai độ lệch chuẩn.

**Dấu và mức 0:** Dương/âm/0: trên/dưới/bằng trung bình.

**Giới hạn và hiểu nhầm:** Phụ thuộc độ lệch chuẩn; số cực lớn không chứng minh smart money.

**Tham chiếu Python:** add_volume_features: (volume − mean) / std.where(std != 0)

#### volume_change_1

**Tên feature:** volume_change_1

**Nhóm:** Volume

**Công thức:** V_t / V_(t−1) − 1

**Ký hiệu:** V là volume; mẫu số là phiên trước.

**Cách tính:** Chia volume hiện tại cho phiên trước rồi trừ 1; mẫu số 0 cho NaN.

**Ví dụ số:** 100 → 150: +0,5 = +50%; 100 → 80: −0,2 = −20%.

**Ý nghĩa tài chính:** Thay đổi tương đối qua một phiên.

**Dấu và mức 0:** Dương: tăng; âm: giảm; 0: không đổi; tối thiểu −1 với mẫu số dương.

**Giới hạn và hiểu nhầm:** Mẫu số rất nhỏ khiến tỷ lệ rất lớn, không nhất thiết do hoạt động kinh tế.

**Tham chiếu Python:** add_volume_features: volume / volume.shift(1).where(volume.shift(1) != 0) − 1

#### volume_ratio_20

**Tên feature:** volume_ratio_20

**Nhóm:** Volume

**Công thức:** V_t / mean20

**Ký hiệu:** mean20 là trung bình 20 volume gồm t.

**Cách tính:** Tính trung bình rồi chia volume cho mean; mean=0 cho NaN.

**Ví dụ số:** V=20 triệu, mean=10 triệu: ratio=2 lần.

**Ý nghĩa tài chính:** So volume hiện tại với mức trung bình.

**Dấu và mức 0:** 1: bằng; lớn hơn 1: cao hơn; nhỏ hơn 1: thấp hơn; không âm.

**Giới hạn và hiểu nhầm:** Đây không phải z-score và không điều chỉnh theo std.

**Tham chiếu Python:** add_volume_features: volume / mean.where(mean != 0)

Kiểm tra dữ liệu repository: raw parquet ghi 04/10/2021 volume 7.610 và 05/10/2021 là 50.156.363; processed parquet ghi volume_change_1 = 6.589,849277. Đây là tỷ lệ, tương đương khoảng 658.985%, không phải 6.589,85%. Giá trị này nhạy với mẫu số nhỏ. Cần đối chiếu nguồn gốc và corporate actions, quy ước điều chỉnh giá/volume trước khi quyết định xử lý; không tự động xóa outlier và không suy ra smart money.

## 8. So sánh các features

| **So sánh** | **Khác biệt và cách dùng** |
| --- | --- |
| Lag / momentum | Lag là một return tại t−k; momentum là tổng k return đến t. |
| SMA / EMA | SMA trọng số đều trong cửa sổ; EMA giảm trọng số theo thời gian, seed Close đầu, thường nhạy và nhiễu hơn. |
| RSI / MACD | RSI là oscillator 0–100 từ gain/loss; MACD ratio là chênh EMA chia giá, không có thang cố định. |
| ATR / volatility | ATR dùng high, low, close trước và chuẩn hóa theo giá; volatility dùng std log return Close và năm hóa. |
| Width / volatility | Cùng đo độ phân tán nhưng width dùng mức Close, volatility dùng return; đơn vị và diễn giải khác. |
| Z-score / ratio | Z-score chia độ lệch khỏi mean cho std; ratio chia volume cho mean. Std thay đổi có thể làm z-score khác dù ratio giống. |

Feature redundancy: nhiều cửa sổ và chỉ báo đo cùng hiện tượng nên tương quan có thể cao. Tương quan không chứng minh nhân quả hay tự động khiến feature vô dụng: mô hình, tính phi tuyến và chi phí có thể thay đổi giá trị bổ sung. Chọn feature chỉ trong train/validation theo thời gian; không sử dụng toàn bộ test period để chọn.

## 9. NaN, warm-up và numerical stability

Bảng dùng vị trí zero-based, giả sử OHLCV hợp lệ, đủ dài và mẫu số khác 0; quan sát đầu tiên ở vị trí 0. Những điều kiện dữ liệu đặc biệt có thể làm thời điểm hợp lệ muộn hơn hoặc không bao giờ xuất hiện.

| **Feature** | **Số OHLCV tối thiểu** | **Vị trí đầu** | **Nguyên nhân** |
| --- | --- | --- | --- |
| log_return | 2 | 1 | Thiếu Close trước |
| return_lag_k | k+2 | k+1 | Return đầu thiếu và dịch k |
| sma_ratio_n | n | n−1 | Đủ n Close |
| ema_ratio_n | n | n−1 | min_periods=n |
| macd_ratio | 26 | 25 | Đợi EMA26 |
| momentum_k | k+1 | k | Thiếu Close cách k phiên |
| rsi_14 | 15 | 14 | 14 thay đổi giá |
| volatility_n | n+1 | n | Cần n return |
| atr_ratio_14 | 14 | 13 | 14 TR gồm TR đầu |
| Bollinger 20 | 20 | 19 | 20 Close; %B cần span khác 0 |
| volume_zscore_20 | 20 | 19 | Cần std khác 0 |
| volume_change_1 | 2 | 1 | Volume trước phải khác 0 |
| volume_ratio_20 | 20 | 19 | Mean phải khác 0 |

validate_ohlcv từ chối NaN/Inf input, giá không dương, volume âm, sai hình học nến, trùng index và thiếu cột; đó là lỗi input chứ không phải warm-up. NaN cũng có thể là tỷ lệ không xác định; label NA cuối chuỗi lại do thiếu tương lai. Không fillna(0) toàn bộ: 0 có ý nghĩa kinh tế riêng. Raw và processed parquet đang có 2.639 hàng; processed có 2.440 hàng đầy đủ feature, bắt đầu 21/10/2016. Đây là kiểm tra file hiện có, không bảo đảm kết quả khi dữ liệu thay đổi.

## 10. Data leakage và causal feature design

Look-ahead bias xảy ra khi quyết định dùng thông tin chưa có; future leakage là một nguyên nhân, còn data snooping là thử/chọn nhiều giả thuyết dựa trên kết quả kiểm thử. In-sample dùng để fit; out-of-sample kiểm tra trên thời gian chưa dùng để fit/chọn. Temporal validation và purge các nhãn có khoảng tương lai giao với test giúp tránh đánh giá quá lạc quan, nhưng chỉ cấu hình walk_forward không chứng minh đã triển khai đầy đủ.

Kiểm tra code: return dùng shift(1), lag dùng shift(k) dương; SMA, volatility, Bollinger và volume rolling trailing, không center; EMA và Wilder đệ quy; momentum dùng shift(k); ATR dùng Close trước. Không có negative shift trong features. Ví dụ đúng: close.shift(1); sai cho X_t: close.shift(-1). Trong labeling.py shift(-1) là hợp lệ để tạo target, không được đưa vào X.

Prefix invariance và thay đổi tương lai trong test kiểm tra tính nhân quả của phép tính trên input đã cho, không chứng minh dữ liệu point-in-time. Giá lịch sử điều chỉnh hồi tố có thể chứa ảnh hưởng sự kiện tương lai; OHLC phải cùng quy ước điều chỉnh, volume có thể khác. Unit test PASS không loại trừ mọi leakage, chi phí hay lỗi thực thi.

## 11. Features trong Machine Learning

Raw OHLCV → features.py → 26 features → dataset_builder.py → processed master → temporal split → preprocessing fit trên train → Machine Learning → probability forecasts → backtesting. Đây là sơ đồ công việc, không khẳng định các bước sau builder đã được triển khai.

expected_feature_columns() xác định X theo schema cấu hình; get_training_view() chọn hàng đủ 26 feature và target có giá trị, chưa chia tập. RQ1 target_direction=1 khi ln(C_t+1/C_t)>0, bằng 0 nếu không tăng, NA nếu thiếu phiên sau. X_t xác định cuối t. Không đưa target_direction, target_return_above_cost_proxy, target_fixed_hold_net_proxy, forward_log_return_1, forward_trade_gross_return hoặc forward_trade_net_return_proxy vào X; chúng dùng tương lai. Giá OHLCV thô cũng không thuộc schema X hiện tại.

RSI có thang 0–100, ratio thường gần 0, volatility theo tỷ lệ năm hóa, volume có thể có ngoại lệ. StandardScaler dùng mean/std của train; RobustScaler dùng median và khoảng tứ phân vị của train, thường bớt nhạy với outlier nhưng không sửa dữ liệu sai. Fit từng fold train rồi transform validation/test; không tính thống kê trên cả train và test. Không fit scaler hay huấn luyện trong tài liệu này.

## 12. Cách diễn giải giá trị feature

| **Feature** | **Giá trị ví dụ** | **Diễn giải** |
| --- | --- | --- |
| log_return | 0,02 | Log return 0,02; simple return e^0,02−1 ≈ 2,02%. |
| return_lag_5 | −0,01 | Return log của phiên t−5 âm 0,01. |
| momentum_5 | 0,05 | Log return tích lũy 5 phiên 0,05; simple ≈ 5,13%. |
| sma_ratio_20 | 0,05 | Close cao hơn SMA20 5%. |
| ema_ratio_12 | −0,03 | Close thấp hơn EMA12 3%. |
| macd_ratio | 0,01 | EMA12 cao hơn EMA26 bằng 1% Close. |
| rsi_14 | 75 | Tỷ trọng gain đã làm mượt lớn; không phải lệnh bán. |
| volatility_21 | 0,30 | Volatility log return năm hóa khoảng 30%. |
| atr_ratio_14 | 0,025 | ATR bằng 2,5% Close. |
| bollinger_width_20 | 0,20 | Độ rộng dải bằng 20% SMA. |
| bollinger_percent_b_20 | 1,2 | Close vượt dải trên 20% độ rộng. |
| volume_zscore_20 | 2 | Volume trên mean hai sample std. |
| volume_change_1 | 0,5 | Volume tăng 50% so với phiên trước. |
| volume_ratio_20 | 1,5 | Volume bằng 1,5 lần mean. |

Đây là số giả định, không phải quan sát HPG hay khuyến nghị BUY/SELL. Để tự học: tính lại ví dụ bằng tay; dùng một chuỗi ngắn tái tạo pandas; kiểm tra warm-up và trường hợp phẳng; sau đó so sánh hiệu quả trên các fold thời gian và dữ liệu point-in-time.

## 13. Câu hỏi ôn tập

**1. Vì sao dùng log return?** Nó mô tả thay đổi tương đối và cộng được qua thời gian; vẫn cần xét điều chỉnh dữ liệu.

**2. Lag 5 khác momentum 5?** Lag lấy return tại t−5; momentum cộng năm return từ t−4 đến t.

**3. SMA ratio âm khi nào?** Khi Close nhỏ hơn SMA.

**4. EMA khác SMA thế nào?** EMA có trọng số giảm theo tuổi dữ liệu và thường phản ứng nhanh hơn.

**5. MACD ratio khác MACD truyền thống?** Nó chia hiệu EMA cho Close; code không có signal line.

**6. RSI 80 có chắc phải bán?** Không. Ngưỡng mô tả động lượng, không bảo đảm đảo chiều.

**7. Vì sao annualize volatility?** Để so sánh theo thang năm bằng quy ước √252; giả định có thể không đúng.

**8. Previous Close trong TR để làm gì?** Để nắm khoảng nhảy giá giữa các phiên.

**9. ATR và volatility có giống nhau?** Không: ATR dùng biên độ OHLC và gap, volatility dùng phân tán return Close.

**10. %B có lớn hơn 1 không?** Có, khi Close cao hơn dải trên.

**11. Vì sao volume z-score có thể lớn?** Volume lệch mean nhiều so với std; cũng cần kiểm tra dữ liệu.

**12. Vì sao SMA200 có 199 NaN đầu?** Cửa sổ cần đủ 200 Close; vị trí đầu là 199.

**13. Vì sao không fillna(0) toàn bộ?** 0 có nghĩa riêng; NaN có thể là warm-up hoặc phép chia không xác định.

**14. Tương quan cao có xóa ngay?** Không; kiểm tra giá trị bổ sung trong validation theo thời gian.

**15. Vì sao không dùng forward return làm input?** Nó dùng giá tương lai chưa biết tại thời điểm dự báo.

**16. Vì sao temporal validation?** Để đánh giá trên tương lai và hạn chế trộn thông tin giữa các thời kỳ.

**17. Feature có ý nghĩa chắc dự báo tốt?** Không; phải kiểm định ngoài mẫu và sau chi phí.

**18. Vì sao kiểm tra chất lượng dữ liệu?** Sai giá, điều chỉnh hay volume bất thường có thể làm méo đặc trưng.

**19. Những feature nào mô tả momentum?** momentum_5/10/21 và rsi_14; các lag có thể hỗ trợ nghiên cứu.

**20. Những feature nào mô tả volatility?** volatility_21/63, atr_ratio_14, Bollinger width và %B mô tả độ rộng hoặc vị trí dải.

Kết luận: 26 feature là các phép đo nhân quả trên dữ liệu đầu vào, không phải bằng chứng lợi nhuận. Bước tiếp theo là xác minh chất lượng và thời điểm dữ liệu, thiết kế validation theo thời gian, rồi đánh giá giá trị bổ sung và tính ổn định ngoài mẫu.

Nguồn nội bộ: src/features.py; tests/test_features.py; configs/config.yaml; src/dataset_builder.py; src/labeling.py; data/raw/HPG_10y_daily.parquet; data/processed/HPG_features_labeled.parquet. Cấu hình validation có comment “chống look-ahead bias tuyệt đối”; đó là khẳng định quá mạnh so với bằng chứng từ các phép tính và test.
