**Lộ trình từ code synthetic của bạn sang thí nghiệm văn bản/ảnh**

Tài liệu đối chiếu `BudgetLIME.pdf` (20 trang), các driver trong
`BudgetLIME-main`, `synthetic.py` và `my_core.py` tại workspace này.
Ưu tiên theo lựa chọn của bạn: đi tiếp sang black box. Bước đầu nên dùng một
câu tiếng Anh ngắn với DistilBERT, vì có thể tính chính xác đáp án để kiểm tra.

**1. Bạn đang đứng ở đâu?**

`my_core.py` đã có phần lõi cần để fit một surrogate thật: sinh mặt nạ,
Walsh design, OLS có intercept, `C_est`, floor và tập vượt floor.
Đã so sánh trực tiếp với `bl_core.py` trên cùng Z, y ở bốn cấu hình
`(d,K,N)=(6,1,200),(6,2,300),(30,1,1000),(49,1,2000)`:
design trùng, OLS/C_est/floor khớp trong sai số số học; chênh lệch hệ số lớn
nhất dưới `6e-17`. Đây là kiểm tra trên các cấu hình cụ thể, không phải chứng
minh mọi đầu vào đều đúng.

`synthetic.py` đã dựng lại phần lớn Tier 1 và hai phần Tier 1b:

| Hàm của bạn | Vai trò | Điểm cần hiểu |
|---|---|---|
| `make_synthetic_function` | Tự biết beta và phần bậc cao | Thay hàm này bằng model query là cầu nối sang black box |
| `test_empirical_leakage` | Full cube so với hữu hạn mặt nạ | Trực giao population không làm tương quan thực nghiệm bằng 0 |
| `calibrate_leakage` | Hiệu chỉnh `C_m` | `B_sample` chỉ dùng chẩn đoán, không tự thành chặn trên population |
| `run_forward_collapse`, `u_audit` | Chuẩn hóa sai số, xác suất đúng dấu active | Chưa kiểm tra quy tắc `abs(beta_hat)>certified_floor` |
| `backward_budget` | Tìm N đạt 90% theo tiêu chí calibration | Dùng SE theo residual và ngưỡng family-wise của thí nghiệm mẫu; không phải full floor của Eq. (6) |
| `sweep_d_stability` | So sánh hằng số theo d | Mean qua d khác kết quả riêng d=30 |
| `cest_mechanism`, `cest_transfer` | Khảo sát điều kiện thiết kế | Phải đo lại C_est cho mỗi Z |

Những điểm cần chú ý khi nối code:

- `my_core.design_matrix` mặc định `intercept=True`, còn `bl_core` mặc định
  `False`. Luôn ghi rõ tham số này ở chỗ tính dự đoán/residual.
- `my_core.certified_set` trả boolean mask; bản mẫu trả `(set chỉ số, signs)`.
  Hai API không thay thế trực tiếp cho nhau được.
- `synthetic.py` hiện chỉ định nghĩa hàm, chưa có `main`/CLI để tự chạy thí nghiệm.
  Trong `.venv` hiện tại import file còn vướng `matplotlib` chưa được cài.
- Trong `run_forward_collapse`, success chỉ là cả nhóm active đúng dấu. Đừng
  diễn giải đường SDR này thành tỷ lệ certificate đúng ở mức tin cậy `1-delta`.
- `make_synthetic_function` chưa bảo vệ các trường hợp `d<2` hoặc `n_hi=0`
  (chia cho số hạng bằng 0); `sample_degree1_problem(rng=None)` cũng chưa tự
  tạo RNG. Các mặc định d=30 vẫn tránh được các trường hợp này.
- `log_pk_over_delta` và `certified_floor` cần validation trước khi nhận input
  tùy ý; khi dùng công thức Walsh này phải giữ `p_keep=0.5`.
- Các seed ở forward của bạn khác bản mẫu. Tái lập xu hướng và khớp từng số
  là hai mục tiêu khác nhau; không sửa hằng số chỉ để ép khớp bảng báo.

**2. Bản chất bước chuyển sang black box**

Ở synthetic, bạn tự viết `g_fn(Z)`. Ở NLP, `g_fn` trở thành:

```text
một câu x + một class c cố định + một reference rho
             |
mặt nạ Z -> thay embedding của token bị che -> DistilBERT -> y
             |
       my_core.ols_fit(Z, y, K)
             |
       beta_hat, C_est(Z), floor
```

Không huấn luyện lại DistilBERT. Ta huấn luyện một hồi quy nhỏ để xấp xỉ đáp
ứng của mô hình quanh câu đang giải thích, theo phân phối mặt nạ của bài báo.

- `d` là số token/subword tự do sau tokenization, không nhất thiết bằng số từ.
  Special tokens được giữ cố định.
- `z_i=1`: giữ embedding; `z_i=0`: thay bằng embedding `[MASK]`, `[PAD]`, hoặc
  vector zero. Wrapper hiện giữ attention mask, nên `pad` không đồng nghĩa
  xóa token khỏi câu.
- `c` lấy từ class mô hình dự đoán trên câu nguyên vẹn, rồi giữ nguyên cho mọi
  mặt nạ. Đổi class theo từng mặt nạ sẽ đổi hàm đang giải thích.
- Với `K=1`, `X=[1, chi_1,...,chi_d]`, `chi_i=2z_i-1`.
  Với `K=2`, thêm các cột `chi_i*chi_j`, `i<j`.
- Hệ số population là `beta_S=E[g_rho(Z)*chi_S(Z)]`.
  Dưới uniform masks, `2*beta_i` là chênh lệch trung bình khi bật/tắt token i,
  lấy trung bình qua trạng thái các token còn lại. Đây không phải lời khẳng
  định nhân quả về ngôn ngữ.

Trong wrapper hiện tại cả NLP và ảnh đều dùng `eval()`/`no_grad()`;
`sigma_obs=0` theo mô hình truy vấn xác định. Xác suất 0.7 là một số được trả
về, không có nghĩa wrapper tung Bernoulli(0.7). Nhiễu query khác với sự không
chắc chắn của class. Sai số beta vẫn có thể khác 0 do mismatch và mặt nạ hữu hạn.

Công thức cần giữ nguyên từ Eq. (6), trang 7:

```text
pK = 1+d                         (K=1)
pK = 1+d+d(d-1)/2               (K=2)
G = X.T @ X / N
C_est = max(lambda_min(G)^(-1/2), ||G^(-1)||_inf)
L = log(2 * split * pK / delta)
floor = C_est * [sigma_obs*sqrt(2L/N)
                + sqrt(2*m_upper*L/N)
                + (2/3)*B_population_upper*L/N]
```

`abs(beta_hat)>floor` chứng nhận dấu khi các giả thiết/chặn trên hợp lệ.
`abs(beta_hat)<=floor` là chưa phân giải được, không phải hệ số bằng 0.
Nếu `abs(beta_true)>2*floor`, định lý hứa phát hiện trên sự kiện của bound.
Confidence là đồng thời trên các tọa độ **trong một run**; nhiều ngân sách
hay nhiều câu không tự được bảo đảm chung ở cùng delta.

`C_m=0.795`, `C_budget=1.508` trong bản mẫu chỉ phục vụ planning:
`sigma_eff=sigma_obs+C_m*sqrt(m_plan)` và
`N_pred=ceil(2*C_budget^2*sigma_eff^2*L/beta_min^2)`.
Sau đó lấy ít nhất `3*pK` theo quy tắc thực hành và đo lại floor.
`3*pK` không tự bảo đảm Gram luôn tốt hoặc floor đạt mục tiêu.
Muốn chứng nhận estimates vượt `beta_min` cần `floor<=beta_min`;
muốn phát hiện mọi true effect lớn hơn `beta_min` cần điều kiện mạnh hơn
`floor<=beta_min/2`.

**3. Bài thực hành đã chuẩn bị: exact beta trên một câu ngắn**

File mới `blackbox_walkthrough.py` dùng wrapper `bl_models.TextClassifier`
cho model query, nhưng dùng **my_core.py của bạn** cho OLS/C_est/floor.
Nó có ba mode: `demo` (NumPy), `nlp` (DistilBERT), `cache` (tính lại từ dữ liệu
đã query). Hai file bạn tự viết và code mẫu không bị chỉnh sửa.

Với d token ngắn, ta làm được thứ mà black box lớn không làm nổi:

1. Query đủ `2^d` mặt nạ để biết hàm `g_rho` trên toàn cube.
2. Tính exact beta bằng `X_cube.T @ y_cube / 2^d`; ở đây Gram chính xác là I.
3. Tính `r_exact`, `m_exact=mean(r_exact**2)`, `B_exact=max(abs(r_exact))`.
   Lấy max trên **toàn cube** nên đây thật sự là sup-norm của hàm trên miền này.
4. Sinh N mặt nạ i.i.d. có hoàn lại bằng `sample_masks`, tra output trong cache.
5. Fit bằng code của bạn; so sánh sai số, floor, số vượt floor, sai dấu và
   khả năng thu hồi các true effects vượt `2*floor`.

Chi phí `2^d` là chi phí lấy ground truth riêng. N trong bảng là ngân sách
mô phỏng của một run. Có thể N lớn hơn `2^d` vì lấy mẫu có hoàn lại.
Không lấy một prefix của cube đã xáo trộn rồi gọi là i.i.d.

Môi trường được kiểm tra trong phiên này: Python 3.12.3, NumPy 2.5.3,
Torch 2.14.1+cu130. `torch.cuda.is_available()` trả False trong môi trường
thực thi của phiên này; điều đó không kết luận máy vật lý của bạn không có GPU.
Chưa có `transformers`, `torchvision`, `Pillow`, `matplotlib`, `scipy`, `tqdm`.
Chưa tải/chạy model thật hoặc cài thêm package trong phiên này.

Chạy thử pipeline ngay, không cần cài thêm:

```bash
cd /home/tiem/Research/FiniteBudgetLime
OPENBLAS_NUM_THREADS=1 .venv/bin/python blackbox_walkthrough.py demo \
  --K 1 --out-dir tmp/demo_k1
OPENBLAS_NUM_THREADS=1 .venv/bin/python blackbox_walkthrough.py demo \
  --K 2 --out-dir tmp/demo_k2
```

Kết quả demo đã kiểm tra (seed=0, không phải kết quả bài báo hay mô hình NLP):

| K | pK | m_exact | N | C_est | floor | max error, gồm intercept | cert / false signs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 0.005525 | 128 | 1.722 | 0.04299 | 0.01654 | 2 / 0 |
| 2 | 22 | 0.000625 | 128 | 4.496 | 0.04434 | 0.00458 | 3 / 0 |
| 1 | 7 | 0.005525 | 4000 | 1.120 | 0.00454 | 0.00155 | 2 / 0 |
| 2 | 22 | 0.000625 | 4000 | 1.348 | 0.00217 | 0.00098 | 3 / 0 |

Ở N=128, K=2 giảm mismatch nhưng C_est tăng nhiều nên floor vẫn cao hơn K=1.
Đây là một ví dụ cụ thể để hiểu sự đánh đổi của việc thêm tương tác.

Để chạy câu thật, cài phần NLP còn thiếu rồi chạy:

```bash
cd /home/tiem/Research/FiniteBudgetLime
.venv/bin/python -m pip install transformers
.venv/bin/python -m pip check
HF_HUB_CACHE="$PWD/tmp/hf_hub" .venv/bin/python blackbox_walkthrough.py nlp \
  --sentence "This movie is really good." \
  --reference mask --K 1 --device cpu --chunk-size 16 \
  --out-dir tmp/distilbert_mask_k1
```

Lệnh cài và cơ chế tải/cache đối chiếu với [tài liệu Transformers chính thức](https://huggingface.co/docs/transformers/installation).
Checkpoint là [DistilBERT đã fine-tune trên SST-2 tiếng Anh](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
Lần đầu cần mạng để lấy tokenizer/weights. Script mặc định chặn d>10 để
thí nghiệm đầu nhỏ; có thể đặt `--max-d 13` khi đã hiểu chi phí. Model thật
chưa được kiểm thử trong môi trường hiện tại, nên tương thích package/checkpoint
còn cần xác nhận khi bạn chạy lệnh này.

Đọc các file trong `tmp/distilbert_mask_k1`:

- `coefficients.csv`: tên/vị trí token, beta_exact, beta_hat, floor, certified,
  đúng dấu và thuộc vùng guarantee margin hay không, ở từng N.
- `summary.csv`: số cert, false signs, max error, bound_holds và margin recovery.
  `false_signs=0` với `certified=0` là chưa có thông tin về khả năng phát hiện.
- `metadata.json`: câu, target class, reference, tokenization, model revision
  nếu wrapper cung cấp, môi trường, seed, K, delta và chi phí query mới.
- `cube.npz`: toàn bộ output đã query. Chứa nguyên câu trong metadata.

Đổi K hoặc seed **trên cùng câu/reference** mà không tải/query mô hình lại:

```bash
.venv/bin/python blackbox_walkthrough.py cache \
  --cache tmp/distilbert_mask_k1/cube.npz \
  --K 2 --seed 0 --out-dir tmp/distilbert_mask_k2

.venv/bin/python blackbox_walkthrough.py cache \
  --cache tmp/distilbert_mask_k1/cube.npz \
  --K 1 --seed 1 --out-dir tmp/distilbert_mask_seed1
```

Mode cache dùng câu/reference lưu trong cache; muốn đổi reference phải chạy
mode nlp và tạo cube mới. Mỗi thư mục output giữ kết quả một cấu hình; chạy
lại cùng thư mục sẽ thay các file kết quả trong thư mục đó.

Thứ tự tự đọc file thực hành: `collect_cube` -> `experiment` -> đoạn
`beta_full_exact` -> đoạn `Zbank` -> vòng `for N`. Câu hỏi cần tự trả lời:
tại sao exact beta không cần C_m; tại sao beta_i population không đổi khi
nâng K=1 lên K=2; vì sao beta_hat_i hữu hạn có thể đổi; vì sao reference mới
cần ground truth mới?

**4. Bản đồ toàn bộ thí nghiệm mẫu**

| File | Phần bài báo | Câu hỏi/đại lượng chính |
|---|---|---|
| `tier1_synthetic.py` | §5.1, Table 2 | C_m, C_budget; SDR normalization; sweep theo d |
| `tier1b_cert_transfer.py` | §5.1, Appendix A | C_est theo d,K,N; không coi là hằng số transferable |
| `tier2_blackbox.py nlp/image` | §5.2, Table 3 | Đổi dấu ở giao hai tập vượt floor; kiểm tra ngân sách dự đoán |
| `tier2_blackbox.py exact-nlp` | §5.3 | So dấu với exact beta; false signs và hai loại power |
| `tier2b_reseed.py` | §5.4, Table 4 | Cố định N, đổi seed độc lập; majority disagreement/Jaccard |
| `tier3_feasibility.py` | §5.5, Appendix B | K=2 giảm mismatch nhưng tăng pK và yêu cầu thiết kế |
| `baselines.py` | Appendix D, Table D1 | Cùng mask bank: floor, pairs bootstrap, Wald naive/fair |
| `visualize/*.py` | Hình minh họa liên quan Fig. 1 | Snapshot code vẽ đang dùng công thức cũ; xem mục 6 |

`bl_core.py` là lõi toán; `bl_models.py` là adapter query mô hình.
Appendix E là suy dẫn Ridge; snapshot này chưa có driver thực nghiệm Ridge.

Các mốc trong bài để đối chiếu sau khi tái lập đầy đủ: Table 2 freeze mean
`C_m=0.795`, `C_budget=1.508`; Table 3 pooled NLP `0/7713` reversals;
§5.3 có 27 probes, `0/170` false signs, guarantee-margin recovery `87/87`,
cube-resolution recovery `103/240`. Đây là **số báo cáo trong bài**, chưa phải
kết quả tái chạy của phiên này. Cấu hình demo/một câu không được kỳ vọng
cho đúng các tổng này.

**5. Chạy code mẫu theo từng nấc**

Sau bài một câu, chạy một model, một reference, một input trước. Tất cả lệnh
trong khối này chạy từ `BudgetLIME-main`; interpreter là `../.venv/bin/python`.
Log nên lưu cùng môi trường:

```bash
cd /home/tiem/Research/FiniteBudgetLime/BudgetLIME-main
mkdir -p ../tmp/reproduction
../.venv/bin/python -m pip freeze > ../tmp/reproduction/environment.txt

../.venv/bin/python -u tier2_blackbox.py nlp \
  --backbones distilbert --references mask --dataset sst2 \
  --sentences text_samples/sst2_short.txt --subset 1 \
  --K 1 --N_ladder 512,1000,2000,4000 --beta_min 0.02 \
  > ../tmp/reproduction/tier2_distilbert_mask.log 2>&1
```

Khối lệnh có redirect nên xem tiến độ bằng `tail -f` file log trong terminal
khác, hoặc bỏ redirect để đọc trực tiếp. Nếu muốn dùng cache weights của bài
thực hành, đặt `HF_HUB_CACHE` về cùng đường dẫn tuyệt đối
`/home/tiem/Research/FiniteBudgetLime/tmp/hf_hub` trước lệnh model.

Đọc `flips/cmp`: chỉ so dấu ở các tọa độ vượt floor tại **cả hai** rung kế
tiếp. `0/0` không có chứng cứ ổn định. `BWD ratio=floor/beta_min` lớn hơn 1
là kế hoạch chưa đạt độ phân giải mong muốn. Tập cert không bắt buộc tăng
đơn điệu theo N, dù mặt nạ là prefix.

Exact-nlp với code mẫu, nâng lên K=2:

```bash
../.venv/bin/python -u tier2_blackbox.py exact-nlp \
  --backbones distilbert --references mask --dataset sst2 \
  --sentences text_samples/sst2_short.txt \
  --K 2 --max_d 13 --max_free 13 --subset 1 --beta_min 0.02
```

`--max_free 13` bỏ qua câu dài trước khi tính subset, giúp lấy một câu thực
sự enumerable. `--max_d` một mình chưa bảo đảm điều này: câu dài bị skip vẫn
có thể tiêu tốn lượt subset trong driver. Để so với một manifest nghiên cứu
cố định, phải ghi rõ câu nào được dùng/skip thay vì âm thầm đổi selection.

Tiếp đó kiểm tra independent reseeding; lúc học chỉ dùng 4 seed:

```bash
../.venv/bin/python -u tier2b_reseed.py nlp \
  --backbones distilbert --references mask --dataset sst2 \
  --sentences text_samples/sst2_samples.txt \
  --N 2000 --R 4 --K 1 --subset 1 --pilot ucb
```

Nâng lên `R=40`, `subset=10` sau khi pipeline chạy đúng. UCB có thể làm tập
cert nhỏ hoặc rỗng. Đọc cả số cert và disagreement. Majority qua seed chỉ
đo ổn định; không thay exact beta để kết luận đúng dấu.

So sánh baseline, đầu tiên giảm số bootstrap:

```bash
../.venv/bin/python -u baselines.py nlp \
  --backbones distilbert --references mask --dataset sst2 \
  --sentences text_samples/sst2_samples.txt \
  --N 2000 --B 20 --K 1 --subset 1
```

Để theo cấu hình Table D1, dùng `--B 200 --subset 10`. B là số refit bootstrap,
không phải thêm B*N truy vấn model: bootstrap tái lấy mẫu từ bank đã có.

NLP đầy đủ phải tách ngôn ngữ/dữ liệu:

```bash
../.venv/bin/python -u tier2_blackbox.py nlp \
  --backbones distilbert,roberta --dataset sst2 \
  --references mask,pad,zero --sentences text_samples/sst2_short.txt \
  --K 1 --N_ladder 512,1000,2000,4000 --beta_min 0.02

../.venv/bin/python -u tier2_blackbox.py nlp \
  --backbones visobert --dataset vsfc \
  --references mask,pad,zero --sentences text_samples/vsfc_short.txt \
  --K 1 --N_ladder 512,1000,2000,4000 --beta_min 0.02
```

Không dùng một file SST-2 tiếng Anh cho cả ViSoBERT rồi gọi đó là thí nghiệm
VSFC. `--dataset vsfc` chọn checkpoint, **không tự đổi file sentences**.
ViSoBERT có thể cần thêm dependency tokenizer theo thông báo khi load.

Ảnh: cài `torchvision` tương thích với Torch và `pillow`; kiểm tra import
trước. Có thể dùng `python -m pip install torchvision pillow`, nhưng xem
dependency resolver vì nó có thể đổi bản Torch; lưu lại `pip freeze`.
[Tài liệu torchvision](https://docs.pytorch.org/vision/stable/index.html)
là nguồn đối chiếu API của wrapper.

```bash
../.venv/bin/python -c "import torch, torchvision; from PIL import Image; print(torch.__version__, torchvision.__version__)"

../.venv/bin/python -u tier2_blackbox.py image \
  --backbones resnet18 --references mean \
  --images_dir image_samples --glob 'n03028079_church.JPEG' \
  --subset 1 --grid 7 --N_ladder 512,1000 --beta_min 0.05
```

Đây là run học pipeline giảm kích thước, không phải tái lập đủ bảng ảnh.
Sau đó mở rộng `resnet50,resnet18,vit_b_16`, `white,black,mean`,
`--glob '*.JPEG'`, ladder `512,1000,2000,4000` và bỏ subset khi phù hợp.
Ảnh 7x7 có d=49, p1=50, p2=1226; không enumerate `2^49`.
Driver image hiện hard-code K=1: truyền `--K 2` không biến nó thành bài
pairwise. Muốn học K=2 trước, dùng câu ngắn và exact-nlp.

Lệnh NumPy để đối chiếu các phần còn lại:

```bash
OPENBLAS_NUM_THREADS=1 ../.venv/bin/python -u tier1_synthetic.py all
OPENBLAS_NUM_THREADS=1 ../.venv/bin/python -u tier1b_cert_transfer.py all
OPENBLAS_NUM_THREADS=1 ../.venv/bin/python -u tier3_feasibility.py
OPENBLAS_NUM_THREADS=1 ../.venv/bin/python -u tier2b_reseed.py selftest \
  --N 2000 --R 4 --subset 1 --pilot ucb
```

Các run `all` gồm nhiều Monte Carlo; có thể chọn `leakage`, `forward`,
`backward`, `dsweep`, `grid` cho Tier 1; `mechanism`, `transfer`, `operating`
cho Tier 1b. `matplotlib` chỉ cần khi dùng phần vẽ trong file synthetic của
bạn, không phải cho những driver NumPy này.

**6. Những chỗ cần thận trọng khi đọc kết quả snapshot này**

Đây là khác biệt quan sát trực tiếp trong code, không phải kết luận rằng
các bảng của bài báo đã sai. Muốn tái lập đúng từng số còn cần xác nhận
snapshot, manifest input, checkpoint revision và log gốc.

| Vị trí | Hiện trạng | Hệ quả khi tái lập |
|---|---|---|
| README | Có công thức log thiếu factor 2 và mô tả NLP sigma_obs>0 | Dùng Eq. (6), `bl_core.log_pk_over_delta` và wrapper hiện tại làm căn cứ |
| `tier2_blackbox.forward_backward_on_probe` | m dùng point pilot; NLP có B population chung, ảnh fallback B_hat | Đây là kiểm tra ổn định với đầu vào plug-in; chưa tự thành certificate vô điều kiện |
| `tier2_blackbox.exact_sign_check` | m/B lấy chính xác từ toàn cube | Thích hợp nhất để học kiểm tra trực tiếp định lý |
| `tier2b_reseed.audit_probe` | UCB dùng honest split, split=4; plain dùng plug-in | Phân biệt rõ hai hàng khi báo cáo |
| UCB ảnh | `image_probe` không có output_abs_bound/R_val | Chỉ truyền `--B_pop_ub` vẫn chưa đủ: pilot UCB còn cần bound của validation residual; driver hiện sẽ skip |
| `bl_core.plan_budget` | Sinh Zdesign, tính floor, không query/fit model ở N_run; rng mặc định không seed | BWD ratio là kiểm tra trên thiết kế dự kiến với pilot cố định; chưa phải một backward deployment đầy đủ, có thể dao động khi chạy lại |
| `bl_core.sweep_prefix_ladder` | `len(prev_set-cur_set)>1` mới báo không nested | Có thể mất đúng 1 tọa độ mà cột set_nested vẫn True; chỉ là diagnostic, không dùng làm bằng chứng định lý |
| `baselines.compare_on_probe` | Floor dùng m_hat, B_hat từ pilot | Baseline floor cũng là plug-in; không diễn giải mọi output là coverage lý thuyết |
| `baselines.report` | Câu in cuối nói bootstrap không thấy mismatch | Pairs bootstrap có thể phản ánh biến động do lấy mẫu residual xác định; bài Appendix D cũng nói vậy. Hạn chế là không tự cung cấp certificate đồng thời riêng cho population mismatch |
| `visualize/lime_image.py` | C_est cố định, floor một hạng với log(p1); fit y-y_mean nhưng X chưa centered, không fit intercept đầy đủ | Chưa dùng để tái lập Eq. (6)/Fig. 1 theo toán hiện tại; cần nối về bl_core trước |
| Đường dẫn mặc định Tier 2b/baselines | `sst2_samples.txt`, `benchmark_50` không đúng layout workspace | Dùng rõ `text_samples/sst2_samples.txt`, `image_samples` |
| Model load thất bại | Driver bắt exception rồi in `[skip ...]` | Exit code 0/bảng trống không có nghĩa thí nghiệm đã chạy thành công |

Đã chạy Tier 3 mặc định: K1 có pK=19, sigma_eff≈0.2389, N_run=4983;
K2 có pK=172, sigma_eff≈0.2190, N_run=6592. Cả hai vẫn `resolution`-bound.
Toàn grid d=8..30 mặc định cũng chưa có hàng feasibility-bound. Vì vậy log
này cho thấy chi phí tăng khi K tăng, nhưng **chưa minh họa điểm giao** nơi
feasibility chi phối; cần quét thêm beta_min/d để thấy điểm giao đó.

**7. Tiêu chí hoàn thành buổi thực hành đầu tiên**

Bạn đã hiểu bước nối synthetic -> NLP khi có thể giải thích được một dòng
`coefficients.csv` từ đầu đến cuối: token nào, target class nào, reference nào,
beta_exact là kỳ vọng gì, beta_hat lấy từ những mặt nạ nào, và vì sao floor
cho phép hoặc chưa cho phép kết luận dấu.

Tiếp theo tự triển khai từng phần còn thiếu trong `my_core.py`: point pilot
cho planning; UCB mismatch trên split độc lập; tách `R_val` với `B_pop`;
budget planner có seed, rồi query/fit thật ở ngân sách đã chọn. Sau cùng
thêm reseeding và baseline. Đừng gộp tất cả vào một file chạy `all` ngay từ
đầu: sau mỗi bước giữ lại CSV, seed, input/reference và một giải thích ngắn
về điều đã kiểm tra.

Kiểm tra đã thực hiện trong phiên chuẩn bị tài liệu: đối chiếu core ở bốn
cấu hình; demo K1/K2 với mismatch energy biết trước; cache replay trùng
CSV; ngân sách không đủ được báo unresolved; selftest UCB của repo với
N=2000, R=4, subset=1; Tier 3 mặc định. Chưa chạy model NLP/ảnh thật và chưa
tái lập toàn bộ bảng số trong PDF.
