# Data Loader

Data Loader hiện hoạt động offline với Raw hiện tại, cache KBS và bằng chứng Yahoo đã có. Không gọi API khi chạy các lệnh dưới đây.

```powershell
python -m src.data_loader audit
python -m src.data_loader audit --output-dir results/data_audit/audit_moi
python -m src.data_loader build
python -m src.data_loader build --profile research
```

Profile `research` đọc 11 snapshot VCI, kiểm tra hash, warning, OHLCV, outlier và calendar đối chiếu, rồi tạo `data/raw/HPG_10y_daily_research.parquet` cùng provenance nếu research gate đạt. Nó không ghi đè file đã có và không thay thế gate `validated`. Giá VCI theo nghìn đồng được nhân 1.000 thành VND; volume không đổi. Hai bản ghi volume bằng 0 và OHLC phẳng được ghi rõ lý do loại. Calendar chính thức, corporate actions và tính point-in-time còn hạn chế; không xem đây là dataset validated hoặc giá thực thi backtest.

Sau khi có Raw Research, chạy `python -m scripts.run_research_phase2` để tạo Processed Research và báo cáo riêng. Script không ghi đè các kết quả cũ. Feature Analysis chỉ dùng train; EDA mô tả sử dụng toàn bộ lịch sử, bao gồm cả giai đoạn test, do đó test không còn hoàn toàn kín đối với EDA.

`audit` mặc định chỉ in báo cáo. Tùy chọn `--output-dir` ghi báo cáo vào thư mục **chưa tồn tại**, không ghi đè dữ liệu Raw hoặc processed. `build` chạy cùng audit rồi áp dụng Quality Gate. Lệnh trả mã 2 và không tạo dataset nếu còn blocker. Hiện chính sách và bằng chứng chưa đầy đủ, vì vậy build chưa thể thành công.

`sessions.csv` ghi từng quan sát theo nguồn. `calendar_match_only` chỉ xác nhận ngày có trong cả hai nguồn, không xác nhận OHLCV; `hold_review` là cảnh báo cần xử lý; `retain_unverified` giữ nguyên phiên chưa có KBS đối chiếu. `yearly_source_quality.csv` tổng hợp theo năm và nguồn; không cộng các nguồn để tính số phiên duy nhất. `metadata.json` ghi phạm vi, hash SHA-256 của dữ liệu đầu vào, blocker và chính sách. Snapshot Yahoo chỉ chứa các phiên thiếu từng được audit, không phải toàn bộ Yahoo.

Volume được sàng lọc theo median của tối đa 30 phiên **trước đó**, tối thiểu 10 phiên, cả trong cùng nguồn và tỷ lệ giữa hai nguồn. Ngưỡng 10 lần chỉ là chẩn đoán hai chiều, không xác nhận sai sót hay chuyển đổi đơn vị. Date alignment 2025 sử dụng lịch KBS trong cache trước khi ghép với Raw; các hệ số trong phép so sánh là giả định của audit cũ, không áp dụng vào dữ liệu. Cache không tự chứng minh lịch giao dịch đầy đủ.

Nến có bốn giá OHLC giống nhau được giữ để kiểm tra placeholder, không tự kết luận sai. Không nội suy, tự dịch ngày, ghép OHLCV giữa nguồn hay chuyển đổi volume. Các interface Yahoo cũ còn được giữ để tái lập audit lịch sử; `run_data_pipeline` cũ bị khóa vì trước đây ghi đè Raw mà không có Quality Gate. Chỉ cân nhắc xuất dataset sau khi xác minh độc lập lịch, cơ sở điều chỉnh, corporate actions, outlier và các phiên bất thường.

## Hồ sơ xác minh cho một phạm vi liên tục

Gate hỗ trợ `--evidence` trỏ tới manifest JSON trong project. Không có hồ sơ thật đi kèm: dữ liệu hiện tại vẫn chưa đạt. Khi có bằng chứng độc lập đã được thẩm định, chạy:

```powershell
python -m src.data_loader audit --evidence duong_dan/manifest.json --output-dir results/data_audit/audit_da_tham_dinh
python -m src.data_loader build --evidence duong_dan/manifest.json
```

Manifest schema 1 cần `start_inclusive`, `end_exclusive`, `reviewer`, `reviewed_at` (timestamp có timezone), `excluded_scope_reason`, `calendar_evidence_explanation`, `same_basis_confirmed: true` và `calendar_complete_confirmed: true`. Khoảng chọn phải nằm trong cấu hình; các phiên ngoài khoảng được báo rõ, không tự xóa khỏi Raw.

Đối tượng `basis` cần `price_unit`, `price_adjustment`, `volume_unit`, `volume_adjustment`, `corporate_actions_version`, `evidence_explanation` và `point_in_time: "verified"`. Xác nhận point-in-time phải dựa vào tài liệu thực tế, bao gồm ảnh hưởng của các corporate actions; không suy luận từ ratio. Chỉ khớp OHLCV giữa hai nguồn chưa đủ chứng minh cơ sở này.

Đối tượng `artifacts` có năm mục `raw`, `reference`, `calendar`, `decisions`, `basis_evidence`; mỗi mục gồm `path` tương đối project root, `sha256`, `provider` và `version`. Các file phải riêng biệt; provider của `reference` và `calendar` phải khác Raw. `raw` phải đúng file CSV trong cấu hình. `reference` là CSV OHLCV độc lập có cột `date`; `calendar` là CSV có cột `date`; `basis_evidence` là tài liệu làm căn cứ chính sách. Phiên bản, tính độc lập và độ tin cậy của nguồn cần được người duyệt xác minh, không thể suy ra chỉ từ tên provider hay hash.

CSV `decisions` phải có `date`, `decision`, `reason`, `reviewer`, `evidence_reference`; mỗi ngày Raw có đúng một dòng. Phiên trong phạm vi phải là `accept`, có lý do, người duyệt, và `evidence_reference` bằng `reference`; phiên ngoài phạm vi phải là `exclude` kèm lý do và người duyệt. Phiên bị `reject` hoặc thiếu quyết định khiến gate chặn. Giữ lại file decisions và các artifact để kiểm toán.

Gate kiểm tra hash, phạm vi, ngày duy nhất, OHLCV hợp lệ, lịch đầy đủ theo calendar đã được chứng thực, và Raw khớp **chính xác** OHLCV với reference cùng basis trên toàn bộ phạm vi. Không có chuyển đổi ngầm hoặc chấp nhận sai lệch theo tolerance. Phương pháp bảo thủ này có thể chặn hai nguồn thực sự tương thích nhưng không giống hệt; phải bổ sung phương pháp được chứng minh thay vì điều chỉnh ngưỡng tùy ý. Khi PASS, dữ liệu giữ nguyên từ Raw; file provenance đi kèm ghi phạm vi bị loại trừ, hash dataset, hồ sơ, artifacts, phiên bản và basis. `build` không ghi đè file đã tồn tại.

**Giới hạn:** SHA-256 chỉ xác nhận file không đổi, không xác nhận nội dung đúng, provider độc lập hay reviewer thực sự có thẩm quyền. Các xác nhận trong manifest là thẩm định thủ công, cần đối chiếu tài liệu nguồn. Lệnh audit/build không gọi API; việc thu thập chứng cứ từ bên ngoài phải lưu provenance và tuân thủ rate limit của provider trước khi tạo hồ sơ.

## Candidate Dataset Builder (schema 2)

Schema 2 cho phép candidate chọn từng phiên từ các snapshot khác Raw cũ, kể cả phiên Raw thiếu hoặc cần thay thế. Không tạo quyết định hay manifest kiểm chứng tự động từ kết quả so sánh. Mỗi phiên phải có `date`, `decision=accept`, `source`, `reason`, `reviewer` và `evidence_reference=reference`; ngày phải tồn tại trong snapshot được chọn. Calendar đã được thẩm định và reference phải bao phủ chính xác các phiên trong khoảng; cả candidate và reference phải đạt OHLCV validation và khớp từng giá trị sau chuẩn hóa đơn vị. Các phiên chưa được chấp nhận làm gate BLOCKED, không nội suy.

Manifest schema 2 kế thừa trường `basis`, `artifacts.reference`, `artifacts.calendar`, `artifacts.decisions`, `artifacts.basis_evidence` và các xác nhận của schema 1. Thêm `sources` là mapping từ source ID sang artifact snapshot. Mỗi source và reference cần `path`, `sha256`, `provider`, `version`, `date_column`, `price_unit`, `price_adjustment`, `volume_unit`, `volume_adjustment`, `corporate_actions_version`, `point_in_time`. Các basis phải giống nhau và có bằng chứng; provider của reference và calendar phải khác các source được chọn. Tính độc lập, license và sự thật của basis vẫn cần được người có trách nhiệm thẩm định.

Chuyển đổi duy nhất được hỗ trợ là `1000_VND × 1000 = VND`; `VND` giữ nguyên, volume phải theo `shares` và giữ nguyên. Không có phép điều chỉnh giá, volume hay dịch ngày. Sự kiện doanh nghiệp và price basis phải được xác minh point-in-time. Dữ liệu phù hợp Feature Engineering chưa mặc nhiên phù hợp Backtesting: Backtesting cần giá khớp lệnh và corporate actions theo ngày hiệu lực, thay vì mặc định dùng giá adjusted làm giá thực thi.

Khi dùng schema 2, `candidate_sessions.csv` trong báo cáo audit ghi nguồn và quyết định từng phiên, còn `verification_decisions.csv` và provenance có hash của mọi artifact. File validated chỉ được tạo sau Quality Gate; dữ liệu VCI tải qua `collect_vci` vẫn là snapshot chưa xác minh, không phải manifest hoặc candidate đã được duyệt.
