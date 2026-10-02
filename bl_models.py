from __future__ import annotations
import numpy as np

NLP_REFERENCES = ("mask", "pad", "zero")
IMAGE_REFERENCES = ("white", "black", "mean")
NLP_BACKBONES = ("distilbert", "roberta", "visobert")
IMAGE_BACKBONES = ("resnet50", "resnet18", "vit_b_16")

class TextClassifier:
    def __init__(self, model="distilbert", dataset="sst2", device=None,
                     chunk_size=64):
        import torch
        import inspect
        from transformers import  (AutoTokenizer,
                                   AutoModelForSequenceClassification)

        self.torch = torch
        self.inspect = inspect
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.model_key = model
        self.model_name = self._resolve_model_name(model, dataset)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                        use_fast=True)
        self.model = (AutoModelForSequenceClassification
                        .from_pretrained(self.model_name)
                        .eval().to(self.device))
        self.embed = self.model.get_input_embeddings()
        
    def close(self):
        try:
            del self.model, self.embed
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass
    @staticmethod
    def _resolve_model_name(model, dataset):
        table = {
            ("distilbert", "sst2"):
                "distilbert-base-uncased-finetuned-sst-2-english",
            ("distilbert", "imdb"):
                "textattack/distilbert-base-uncased-imdb",
            ("distilbert", "rotten"):
                "textattack/distilbert-base-uncased-rotten-tomatoes",
            ("bert", "sst2"): "textattack/bert-base-uncased-SST-2",
            ("bert", "imdb"): "textattack/bert-base-uncased-imdb",
            ("bert", "rotten"): "textattack/bert-base-uncased-rotten-tomatoes",
            ("roberta", "sst2"): "textattack/roberta-base-SST-2",
            ("roberta", "imdb"): "textattack/roberta-base-imdb",
            ("roberta", "rotten"): "textattack/roberta-base-rotten-tomatoes",
            # ---- Vietnamese sentiment models --------------------------- #
            # ViSoBERT-based 3-class (0=NEG, 1=POS, 2=NEU); SentencePiece
            # tokenizer -> interpretable units are SUBWORD pieces. Works on
            # RAW text (no external word segmentation, unlike PhoBERT).
            ("visobert", "vsfc"): "5CD-AI/Vietnamese-Sentiment-visobert",
            ("visobert", "vicomment"): "5CD-AI/Vietnamese-Sentiment-visobert",
            ("visobert", "sst2"): "5CD-AI/Vietnamese-Sentiment-visobert",
            # PhoBERT: requires word-segmented input; raw text => degraded.
            ("phobert", "vsfc"): "wonrax/phobert-base-vietnamese-sentiment",
            ("phobert", "vicomment"): "wonrax/phobert-base-vietnamese-sentiment",
        }
        return table.get((model, dataset), model)  # allow raw HF id

    def _baseline_embedding(self, X, kind):
            """Return a baseline embedding of shape (1, L, d), detached.
            Supported NLP references: mask | pad | zero (also mean | random)."""
            torch = self.torch
            L, d = X.shape[1], X.shape[2]
            tok = self.tokenizer
            embed = self.embed
            if kind == "mask":
                # `or` is wrong when mask_token_id == 0 (a valid id); test None.
                tid = tok.mask_token_id
                if tid is None:
                    tid = tok.pad_token_id
                if tid is None:
                    tid = tok.unk_token_id
                if tid is None:
                    raise ValueError(
                        "tokenizer exposes no mask/pad/unk id for baseline='mask'; "
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
            elif kind == "zero":
                return torch.zeros(1, L, d, device=self.device, dtype=X.dtype)
            elif kind == "mean":
                with torch.no_grad():
                    mean_vec = embed.weight.mean(dim=0)
                return mean_vec.view(1, 1, d).expand(1, L, d).clone()
            elif kind == "random":
                vocab = embed.weight.shape[0]
                rid = torch.randint(0, vocab, (1,), device=self.device)
                with torch.no_grad():
                    base = embed(rid.unsqueeze(0))
                return base.expand(1, L, d).clone()
            else:
                raise ValueError(f"unknown reference '{kind}' "
                                 "(mask|pad|zero|mean|random)")

    def encode(self, sentence):
        torch = self.torch
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

        if "token_type_ids" in fwd_params and token_type_ids is not None:
            extra["token_type_ids"] = token_type_ids

        with torch.no_grad():
            X = self.embed(input_ids)
        L = X.shape[1]

        is_special = special[0].bool()
        is_pad = (attention_mask[0] == 0)
        fixed = (is_special | is_pad)
        free_idx = torch.nonzero(~fixed, as_tuple=False).squeeze(-1)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        free_tokens = [tokens[i] for i in free_idx.tolist()]

        return {
            "X": X,
            "attention_mask": attention_mask,
            "extra_kwargs": extra,
            "free_idx": free_idx,
            "L": L,
            "tokens": tokens,
            "free_tokens": free_tokens,
        }

    def target_class(self, ctx):
        torch = self.torch
        with torch.no_grad():
            logits = self.model(input_embeds = ctx["X"].unsqueeze(0),
                                attention_mask = ctx["attention_mask"],
                                **ctx["extra_kwargs"]).logits
        return int(logits.argmax().item())

    def query(self, ctx, X_baseline, Z_free, target):
        import torch
        import torch.nn.functional as Fnn

        free_idx = ctx["free_idx"]
        L = ctx["L"]
        X = ctx["X"]
        N = Z_free.shape[0]

        Z_full = torch.ones(N, L, device=self.device, dtype=X.dtype)
        if free_idx.numel() > 0:
            bits = torch.as_tensor(Z_free, device=self.device)
            Z_full[:, free_idx] = bits

