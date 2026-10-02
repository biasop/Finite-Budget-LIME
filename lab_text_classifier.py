"""
lab_text_classifier.py
======================
BÀI TẬP TỰ CODE: Black-Box Model Wrapper (TextClassifier)
Mục tiêu: Tự tay hiện thực hóa cơ chế trích xuất biểu diễn (embedding),
lọc thuộc tính giải thích (free tokens) và tạo nhiễu theo khối (chunked query)
cho mô hình ngôn ngữ Transformer (DistilBERT/BERT/RoBERTa).

HƯỚNG DẪN LÀM BÀI:
1. Đọc kỹ phần docstring và các gợi ý kỹ thuật tại mỗi hàm có gắn nhãn `# TODO: THỬ THÁCH`.
2. Tự viết mã lệnh của bạn bên dưới các nhãn `# TODO`.
3. Chạy file này trong terminal để kiểm tra kết quả qua bộ Test tự động:
       python lab_text_classifier.py
"""

from __future__ import annotations
import numpy as np


class TextClassifier:
    """Black-box wrapper: Câu văn bản + Mặt nạ nhị phân -> Xác suất lớp."""

    def __init__(self, model="distilbert", dataset="sst2", device=None, chunk_size=64):
        """Khởi tạo model và tokenizer từ Hugging Face.
        Phần này đã được cài đặt sẵn để bạn tập trung vào logic cốt lõi.
        """
        import torch
        import inspect
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.inspect = inspect
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.model_key = model
        self.model_name = self._resolve_model_name(model, dataset)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name)
            .eval()
            .to(self.device)
        )
        # Lớp embedding đầu tiên của mô hình Transformer
        self.embed = self.model.get_input_embeddings()

    def close(self):
        """Giải phóng bộ nhớ GPU khi không sử dụng."""
        try:
            del self.model, self.embed
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _resolve_model_name(model, dataset):
        table = {
            ("distilbert", "sst2"): "distilbert-base-uncased-finetuned-sst-2-english",
            ("distilbert", "imdb"): "textattack/distilbert-base-uncased-imdb",
            ("bert", "sst2"): "textattack/bert-base-uncased-SST-2",
            ("roberta", "sst2"): "textattack/roberta-base-SST-2",
        }
        return table.get((model, dataset), model)

    # =========================================================================
    # THỬ THÁCH 1: BASELINE EMBEDDING FACTORY
    # =========================================================================
    def _baseline_embedding(self, X, kind: str):
        """Trả về vector embedding nền (baseline) có shape (1, L, d_model), cùng device và dtype với X.

        Khi một token bị tắt (z = 0), embedding của nó sẽ được thay thế bằng baseline này.

        Yêu cầu hỗ trợ 3 loại 'kind':
          1. "zero": Toàn bộ các giá trị bằng 0 (dùng torch.zeros).
          2. "mask": Dùng token [MASK] của tokenizer.
             - Chú ý: `tok.mask_token_id` có thể bằng 0 (một ID hợp lệ), không nên dùng `or`.
             - Nếu không có mask_token_id, fallback về pad_token_id hoặc unk_token_id.
             - Truyền ID qua `self.embed(...)` rồi dùng `.expand(1, L, d).clone()` để khớp kích thước.
          3. "pad": Dùng token [PAD] của tokenizer (tương tự như trên).

        Tham số:
          X: Tensor embedding gốc shape (1, L, d_model)
          kind: str ("zero" | "mask" | "pad")

        Trả về:
          Tensor shape (1, L, d_model)
        """
        torch = self.torch
        tok = self.tokenizer
        embed = self.embed
        L, d = X.shape[1], X.shape[2]

        # TODO: THỬ THÁCH 1 - Hãy viết code sinh baseline embedding ở đây
        # Gợi ý:
        # if kind == "zero":
        #     return ...
        # elif kind == "mask":
        #     ...
        # elif kind == "pad":
        #     ...
        # else:
        #     raise ValueError(f"Unknown kind '{kind}'")

        if kind == "zero":
            return torch.zeros(1, L, d, device=self.device, dtype=X.dtype)
        elif kind == "mask":
            tid = tok.mask_token_id
            if tid is None:
                tid = tok.pad_token_id
            if tid is None:
                tod = tok.unk_token_id
            if tid is None:
                raise ValueError("tokenizer exposes no mask/pad/unk id for baseline='mask'; "
                                    "use baseline='zero' instead.")
            with torch.no_grad():
                base = embed(torch.tensor([[tid]], device=self.device))
            return base.expand(1, L, d).clone()
        elif kind == "pad":
            tid = tok.pad_token_id
            if tid is None:
                tid = tok.eos_token_id if tok.eos_token_id is not None \
                    else tok.unk_token_id
            if tid is None:
                raise ValueError(
                    "tokenizer exposes no pad id for baseline='pad'; "
                    "use baseline='zero' instead.")
            with torch.no_grad():
                base = embed(torch.tensor([[tid]], device=self.device))
            return base.expand(1, L, d).clone()
        else:
            raise ValueError("unknow kind")

        raise NotImplementedError("Hãy hoàn thành Thử thách 1: _baseline_embedding")

    def make_baseline(self, ctx, kind: str):
        return self._baseline_embedding(ctx["X"], kind)

    # =========================================================================
    # THỬ THÁCH 2: CONTEXT ENCODER (TIỀN XỬ LÝ & LỌC TOKEN)
    # =========================================================================
    def encode(self, sentence: str) -> dict:
        """Tokenize câu văn bản, trích xuất embedding X ban đầu và lọc các token tự do (free tokens).

        Các bước cần thực hiện:
          Bước 1: Dùng `self.tokenizer` với:
                  return_tensors="pt", truncation=True, return_special_tokens_mask=True
                  Đưa toàn bộ tensor trong kết quả sang `self.device`.

          Bước 2: Xử lý tham số mở rộng (extra_kwargs) an toàn:
                  Dùng `self.inspect.signature(self.model.forward).parameters` để kiểm tra.
                  Nếu mô hình hỗ trợ 'token_type_ids' và tokenizer có trả về trường này,
                  hãy lưu vào dict extra = {"token_type_ids": ...}.

          Bước 3: Trích xuất embedding X ban đầu:
                  Dùng `with torch.no_grad(): X = self.embed(input_ids)`. Shape sẽ là (1, L, d_model).

          Bước 4: Xác định token cố định và token tự do:
                  - Token cố định (fixed): là token đặc biệt ([CLS], [SEP]) HOẶC token đệm ([PAD]).
                  - Dùng phép toán boolean tensor: fixed = (is_special | is_pad).
                  - free_idx: Vị trí của các token KHÔNG cố định (~fixed), shape 1D tensor các chỉ số.

          Bước 5: Lấy chuỗi ký tự của các token:
                  - `tokens`: Dùng `self.tokenizer.convert_ids_to_tokens(input_ids[0])`
                  - `free_tokens`: Lọc `tokens` theo các vị trí trong `free_idx`.

        Trả về:
          dict {
              "X": Tensor (1, L, d_model),
              "attention_mask": Tensor (1, L),
              "extra_kwargs": dict,
              "free_idx": 1D Tensor chứa các chỉ số token nội dung,
              "L": int (độ dài toàn bộ chuỗi),
              "tokens": list[str],
              "free_tokens": list[str],
          }
        """
        torch = self.torch

        # TODO: THỬ THÁCH 2 - Hãy viết code hàm encode ở đây

        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True,
                                     return_special_tokens_mask=True)

        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        special = enc.get("special_tokens_mask",
                            torch.zeros_like(input_ids)).to(self.device)

        token_type_ids = enc.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        fwd_params = self.inspect.signature(self.model.forward).paramaters
        extra = {}
        if "token"

        
        

        raise NotImplementedError("Hãy hoàn thành Thử thách 2: encode")

    # =========================================================================
    # THỬ THÁCH 3: DỰ ĐOÁN LỚP GỐC
    # =========================================================================
    def target_class(self, ctx: dict) -> int:
        """Dự đoán nhãn có xác suất cao nhất (argmax) của câu gốc.

        Dùng `self.model(inputs_embeds=ctx["X"], attention_mask=ctx["attention_mask"], **ctx["extra_kwargs"])`
        dưới khối `with torch.no_grad():`, sau đó lấy argmax của logits.
        """
        torch = self.torch

        # TODO: THỬ THÁCH 3 - Hãy viết code lấy class dự đoán ở đây
        raise NotImplementedError("Hãy hoàn thành Thử thách 3: target_class")

    # =========================================================================
    # THỬ THÁCH 4: CHUNKED MASKED FORWARD PASS (QUERY ENGINE)
    # =========================================================================
    def query(self, ctx: dict, X_baseline, Z_free: np.ndarray, target: int) -> np.ndarray:
        """Đánh giá mô hình trên N mẫu mặt nạ nhị phân Z_free.

        Đầu vào:
          ctx: Ngữ cảnh từ hàm encode()
          X_baseline: Tensor baseline từ hàm make_baseline(), shape (1, L, d_model)
          Z_free: Mảng numpy nhị phân shape (N, d), với d = len(free_idx).
                  Giá trị 1 nghĩa là giữ token gốc, 0 là thay bằng baseline.
          target: Lớp mục tiêu cần lấy xác suất (int).

        Quy trình xử lý:
          1. Xây dựng ma trận Z_full shape (N, L):
             - Khởi tạo toàn bộ bằng 1.0 (vì các token đặc biệt luôn được giữ lại).
             - Gán các cột tương ứng với `free_idx` bằng dữ liệu từ `Z_free`.
             - Đưa Z_full về dạng tensor trên `self.device`.

          2. Chuẩn bị phép Vector hóa:
             - X_sq = ctx["X"].squeeze(0)          # shape (L, d_model)
             - Xref_sq = X_baseline.squeeze(0)      # shape (L, d_model)

          3. Chia khối theo self.chunk_size (tránh tràn VRAM):
             Lặp qua các batch i -> j = min(i + cs, N):
             - Lấy slice z = Z_full[i:j].unsqueeze(-1)    # shape (batch, L, 1)
             - Nội suy embedding:
                   X_pert = X_sq * z + Xref_sq * (1.0 - z) # shape (batch, L, d_model)
             - Mở rộng attention_mask và extra_kwargs cho batch hiện tại bằng `.expand(batch_size, -1)`
             - Cho qua model: `out = self.model(inputs_embeds=X_pert, attention_mask=attn_b, **extra_b)`
             - Tính xác suất bằng Softmax: `probs = torch.nn.functional.softmax(out.logits, dim=-1)`
             - Lấy xác suất của lớp `target` cho batch này đưa vào mảng kết quả y[i:j].

        Trả về:
          Mảng numpy 1D shape (N,) chứa xác suất của lớp `target`.
        """
        import torch
        import torch.nn.functional as Fnn

        # TODO: THỬ THÁCH 4 - Hãy viết code hàm query ở đây
        raise NotImplementedError("Hãy hoàn thành Thử thách 4: query")


# =============================================================================
# BỘ KIỂM THỬ TỰ ĐỘNG (UNIT TESTS)
# Bạn không cần sửa phần này. Hãy chạy `python lab_text_classifier.py` để test.
# =============================================================================
def run_tests():
    print("=" * 65)
    print("🚀 BẮT ĐẦU KIỂM THỬ CÁC THỬ THÁCH TRONG LAB_TEXT_CLASSIFIER")
    print("=" * 65)

    print("\n[1/5] Đang khởi tạo TextClassifier với DistilBERT...")
    clf = TextClassifier(model="distilbert", dataset="sst2")
    sample_sentence = "Antigravity is surprisingly powerful and fast."

    # --- Test 1: Baseline Embedding ---
    print("\n👉 Đang kiểm tra Thử thách 1: _baseline_embedding...")
    dummy_X = clf.torch.randn(1, 10, 768, device=clf.device)
    try:
        base_zero = clf._baseline_embedding(dummy_X, kind="zero")
        assert base_zero.shape == (1, 10, 768), f"Sai shape 'zero': {base_zero.shape}"
        assert clf.torch.all(base_zero == 0), "Giá trị baseline 'zero' phải toàn số 0!"

        base_mask = clf._baseline_embedding(dummy_X, kind="mask")
        assert base_mask.shape == (1, 10, 768), f"Sai shape 'mask': {base_mask.shape}"
        # Mọi vị trí trong L phải có embedding giống nhau (broadcasted)
        assert clf.torch.allclose(base_mask[:, 0, :], base_mask[:, 1, :]), "Mọi token trong baseline 'mask' phải giống nhau!"

        base_pad = clf._baseline_embedding(dummy_X, kind="pad")
        assert base_pad.shape == (1, 10, 768), f"Sai shape 'pad': {base_pad.shape}"

        print("   ✅ THỬ THÁCH 1: HOÀN THÀNH XUẤT SẮC!")
    except NotImplementedError:
        print("   ⏳ Thử thách 1 chưa hoàn thành (NotImplementedError).")
        return
    except Exception as e:
        print(f"   ❌ THỬ THÁCH 1 BỊ LỖI: {e}")
        return

    # --- Test 2: Context Encoder ---
    print("\n👉 Đang kiểm tra Thử thách 2: encode...")
    try:
        ctx = clf.encode(sample_sentence)
        required_keys = {"X", "attention_mask", "extra_kwargs", "free_idx", "L", "tokens", "free_tokens"}
        assert required_keys.issubset(ctx.keys()), f"Thiếu khóa trong ctx! Cần có {required_keys}"
        
        L = ctx["L"]
        assert ctx["X"].shape == (1, L, 768), f"Shape của ctx['X'] không đúng: {ctx['X'].shape}"
        assert ctx["attention_mask"].shape == (1, L), f"Shape của attention_mask không đúng: {ctx['attention_mask'].shape}"
        assert len(ctx["tokens"]) == L, f"Số lượng tokens ({len(ctx['tokens'])}) không khớp với L ({L})"
        
        # [CLS] và [SEP] tuyệt đối không được nằm trong free_tokens
        assert "[CLS]" not in ctx["free_tokens"], "[CLS] là token đặc biệt, không được coi là free token!"
        assert "[SEP]" not in ctx["free_tokens"], "[SEP] là token đặc biệt, không được coi là free token!"
        assert len(ctx["free_tokens"]) == len(ctx["free_idx"]), "Độ dài free_tokens phải khớp với free_idx!"
        
        print(f"   Tokens đầy đủ ({L}): {ctx['tokens']}")
        print(f"   Free tokens ({len(ctx['free_tokens'])}): {ctx['free_tokens']}")
        print("   ✅ THỬ THÁCH 2: HOÀN THÀNH XUẤT SẮC!")
    except NotImplementedError:
        print("   ⏳ Thử thách 2 chưa hoàn thành (NotImplementedError).")
        return
    except Exception as e:
        print(f"   ❌ THỬ THÁCH 2 BỊ LỖI: {e}")
        return

    # --- Test 3: Target Class ---
    print("\n👉 Đang kiểm tra Thử thách 3: target_class...")
    try:
        pred_label = clf.target_class(ctx)
        assert isinstance(pred_label, int), "target_class phải trả về int!"
        assert pred_label in (0, 1), f"Label nhị phân không hợp lệ: {pred_label}"
        print(f"   Dự đoán của mô hình cho câu: Lớp {pred_label}")
        print("   ✅ THỬ THÁCH 3: HOÀN THÀNH XUẤT SẮC!")
    except NotImplementedError:
        print("   ⏳ Thử thách 3 chưa hoàn thành (NotImplementedError).")
        return
    except Exception as e:
        print(f"   ❌ THỬ THÁCH 3 BỊ LỖI: {e}")
        return

    # --- Test 4: Query Engine ---
    print("\n👉 Đang kiểm tra Thử thách 4: query...")
    try:
        d = len(ctx["free_idx"])
        X_base = clf.make_baseline(ctx, kind="mask")
        
        # Test case A: Mặt nạ toàn số 1 (không tắt từ nào)
        # Xác suất dự đoán phải khớp gần như tuyệt đối với dự đoán gốc của câu!
        Z_all_ones = np.ones((5, d), dtype=np.float32)
        probs_ones = clf.query(ctx, X_base, Z_all_ones, target=pred_label)
        
        assert probs_ones.shape == (5,), f"Shape output query không đúng: {probs_ones.shape}"
        assert np.all(probs_ones >= 0.0) and np.all(probs_ones <= 1.0), "Xác suất phải nằm trong đoạn [0, 1]!"
        assert np.allclose(probs_ones, probs_ones[0]), "Khi mặt nạ toàn 1, kết quả các hàng phải giống nhau!"

        # Test case B: Mặt nạ ngẫu nhiên N = 100 (vượt quá chunk_size=64 để test chunking)
        N = 100
        np.random.seed(42)
        Z_random = np.random.randint(0, 2, size=(N, d)).astype(np.float32)
        probs_rand = clf.query(ctx, X_base, Z_random, target=pred_label)
        
        assert probs_rand.shape == (N,), f"Shape output khi N=100 không đúng: {probs_rand.shape}"
        assert np.all(probs_rand >= 0.0) and np.all(probs_rand <= 1.0), "Xác suất ngẫu nhiên phải hợp lệ [0, 1]"

        print(f"   Xác suất khi giữ nguyên câu gốc (z=1): {probs_ones[0]:.4f}")
        print(f"   Đã test thành công chunking với N={N} mẫu (chunk_size={clf.chunk_size})")
        print("   ✅ THỬ THÁCH 4: HOÀN THÀNH XUẤT SẮC!")
    except NotImplementedError:
        print("   ⏳ Thử thách 4 chưa hoàn thành (NotImplementedError).")
        return
    except Exception as e:
        print(f"   ❌ THỬ THÁCH 4 BỊ LỖI: {e}")
        return

    print("\n" + "=" * 65)
    print("🎉 CHÚC MỪNG BẠN ĐÃ VƯỢT QUA TOÀN BỘ CÁC THỬ THÁCH CỦA BÀI LAB!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()