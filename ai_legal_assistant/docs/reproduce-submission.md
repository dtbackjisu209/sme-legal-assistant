# Hướng dẫn tái lập bài nộp

Tài liệu này mô tả cách chuẩn bị dữ liệu, mã nguồn, mô hình và các bước chạy để tái lập file nộp `submission.zip` đã dùng trên leaderboard. Bản điểm tốt nhất hiện tại được tạo bằng cách sinh `results.json` đầy đủ trước, sau đó chạy bước lọc citation quyết định để giảm false positive trong `relevant_docs` và `relevant_articles`.

## 1. Kết quả cần tái lập

File nộp cần tạo:

```text
submission.zip
└── results.json
```

`results.json` là JSON array gồm 2.000 phần tử, mỗi phần tử có schema:

```json
{
  "id": 1,
  "question": "Nguyên văn câu hỏi",
  "answer": "Câu trả lời pháp lý bằng tiếng Việt",
  "relevant_docs": [
    "<mã văn bản>|<tên văn bản>"
  ],
  "relevant_articles": [
    "<mã văn bản>|<tên văn bản>|<Điều>"
  ]
}
```

Bản submission đạt `ARTICLES F2-MACRO = 0.3801` được tái lập từ:

```text
data/submissions/private_candidate/results.json
```

sau đó chạy:

```text
scripts/postprocess_submission_citations.py
```

với quota citation:

```text
exact lookup: 1 doc, 1 article
simple question: 2 docs, 2 articles
multi-issue question: 3 docs, 5 articles
```

Lưu ý: nếu chạy lại toàn bộ `generate_submission.py` từ đầu, kết quả có thể khác do retrieval, query planning, reranking và answer generation được thực hiện lại. Để tái lập đúng file đã nộp, dùng quy trình post-process deterministic bên dưới trên `results.json` gốc.

## 2. Nguồn dữ liệu

Các file dữ liệu cần có trong Google Drive hoặc máy local:

```text
sme-legal-data/
├── R2AIStage1DATA (1).json
├── law_articles.jsonl
├── bm25_index.pkl
├── law_chunks_qwen3_06b-480699765925131-2026-06-20-04-01-21.snapshot
├── tools/
│   └── qdrant-1.18.2-colab
└── submissions/
    └── private_candidate/
        └── results.json
```

Ý nghĩa:

- `R2AIStage1DATA (1).json`: tập câu hỏi đầu vào, gồm `id` và `question`.
- `law_articles.jsonl`: corpus Pháp điển đã xử lý, dùng để khôi phục citation chính thức.
- `bm25_index.pkl`: chỉ mục BM25 build từ cùng corpus.
- `law_chunks_qwen3_06b-...snapshot`: snapshot Qdrant chứa vector chunks đã embed.
- `submissions/private_candidate/results.json`: file kết quả đầy đủ đã sinh từ pipeline RAG trước bước lọc citation.

Trong Colab, khuyến nghị đặt dữ liệu tại:

```text
/content/drive/MyDrive/sme-legal-data
```

## 3. Mô hình và checkpoint sử dụng

Pipeline sinh `results.json` ban đầu sử dụng các mô hình công khai, dưới 14B tham số:

```text
Embedding model: Qwen/Qwen3-Embedding-0.6B
Query planner:   Qwen/Qwen3-0.6B
Answer model:    Qwen/Qwen3-4B
Reranker:        BAAI/bge-reranker-v2-m3
Vector store:    Qdrant collection law_chunks_qwen3_06b
```

Các model được tải từ Hugging Face khi chạy pipeline đầy đủ. Không sử dụng mô hình đóng như GPT-4o hoặc Gemini để sinh kết quả.

Để tái lập đúng `submission.zip` đã nộp, cần có `results.json` trung gian đã sinh bằng pipeline trên. Bước post-process không dùng thêm mô hình.

## 4. Mã nguồn

Repository:

```text
https://github.com/dtbackjisu209/sme-legal-assistant.git
```

Nhánh chứa mã tái lập:

```text
fix/reduce-submission-citation-noise
```

Các script chính:

```text
ai_legal_assistant/scripts/generate_submission.py
ai_legal_assistant/scripts/postprocess_submission_citations.py
ai_legal_assistant/scripts/validate_submission.py
ai_legal_assistant/scripts/evaluate_submission.py
```

Các module liên quan:

```text
ai_legal_assistant/src/ai_legal_assistant/domain/services/submission_citation_selector.py
ai_legal_assistant/src/ai_legal_assistant/domain/services/submission_metrics.py
ai_legal_assistant/src/ai_legal_assistant/application/use_cases/generate_competition_submission.py
```

## 5. Cài đặt môi trường

Môi trường khuyến nghị:

```text
Python 3.10+
CUDA GPU nếu chạy lại pipeline đầy đủ
Google Colab T4/A100 hoặc máy local có GPU
Qdrant 1.18.2 nếu chạy retrieval từ đầu
```

Nếu chỉ tái lập file `submission.zip` từ `results.json` đã có, không cần GPU, Qdrant hay cài model. Chỉ cần Python chuẩn.

Nếu chạy pipeline đầy đủ:

```bash
cd ai_legal_assistant
pip install -r requirements.txt
```

Trên Colab, có thể bỏ torch khỏi requirements để dùng bản torch sẵn có:

```bash
grep -v '^torch' requirements.txt > /tmp/requirements-colab.txt
pip install -r /tmp/requirements-colab.txt
pip install --no-cache-dir --force-reinstall \
  "numpy==1.26.4" \
  "scipy==1.14.1" \
  "transformers==4.51.3" \
  "tokenizers==0.21.1" \
  "lm-format-enforcer==0.11.3"
```

## 6. Cách tái lập đúng submission điểm 0.3801

Đây là quy trình nhanh và deterministic. Quy trình này không chạy lại model; nó dùng `results.json` đã sinh sẵn và lọc lại citation theo rule đã dùng để tạo submission tốt nhất.

### 6.1. Trên Colab không cần clone repo

Cell 1:

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

DATA_DIR = Path("/content/drive/MyDrive/sme-legal-data")

INPUT_RESULTS = DATA_DIR / "submissions/private_candidate/results.json"
OUTPUT_DIR = DATA_DIR / "submissions/private_candidate_filtered"

assert INPUT_RESULTS.exists(), f"Không tìm thấy: {INPUT_RESULTS}"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("INPUT_RESULTS:", INPUT_RESULTS)
print("OUTPUT_DIR:", OUTPUT_DIR)
```

Cell 2:

```python
import json, re, zipfile

OUTPUT_RESULTS = OUTPUT_DIR / "results.json"
OUTPUT_ZIP = OUTPUT_DIR / "submission.zip"

EXACT_DOCS = 1
EXACT_ARTICLES = 1
SIMPLE_DOCS = 2
SIMPLE_ARTICLES = 2
MULTI_DOCS = 3
MULTI_ARTICLES = 5

exact_re = re.compile(
    r"\b(?:điều|khoản|điểm)\s+\d|\b\d{1,5}/\d{4}/[\wÀ-ỹĐđ-]+",
    re.I,
)

multi_res = [
    re.compile(r"\b(?:như thế nào|ra sao)\s+và\s+", re.I),
    re.compile(r"\bvà\s+(?:phải|cần|bị|được|xử lý|khắc phục|áp dụng|nộp|thực hiện)\b", re.I),
    re.compile(r"\bvề\s+[^?]{3,80}\s+và\s+[^?]{3,80}\?", re.I),
    re.compile(r"\b(?:thuế|đất đai|kế toán|bảo hiểm|xử phạt|khắc phục)\b[^?]*\bvà\b", re.I),
]

protected = [
    "doanh nghiệp nhỏ và vừa",
    "nhỏ và vừa",
    "vừa và nhỏ",
]

def limits(question):
    q = str(question or "").casefold()

    if exact_re.search(q):
        return EXACT_DOCS, EXACT_ARTICLES

    cleaned = q
    for phrase in protected:
        cleaned = cleaned.replace(phrase, phrase.replace(" và ", " "))

    if any(p.search(cleaned) for p in multi_res):
        return MULTI_DOCS, MULTI_ARTICLES

    return SIMPLE_DOCS, SIMPLE_ARTICLES

def doc_from_article(article_entry):
    parts = [p.strip() for p in str(article_entry).split("|")]
    if len(parts) != 3 or not all(parts):
        return None
    return f"{parts[0]}|{parts[1]}"

with INPUT_RESULTS.open("r", encoding="utf-8") as f:
    records = json.load(f)

assert isinstance(records, list), "results.json phải là JSON array"

before_docs = 0
before_articles = 0
after_docs = 0
after_articles = 0

for r in records:
    max_docs, max_articles = limits(r.get("question", ""))

    old_articles = list(dict.fromkeys(r.get("relevant_articles", [])))
    old_docs = list(dict.fromkeys(r.get("relevant_docs", [])))

    before_articles += len(old_articles)
    before_docs += len(old_docs)

    new_articles = old_articles[:max_articles]

    docs_from_articles = []
    for article in new_articles:
        doc = doc_from_article(article)
        if doc and doc not in docs_from_articles:
            docs_from_articles.append(doc)

    new_docs = []
    for doc in docs_from_articles + old_docs:
        if doc not in new_docs:
            new_docs.append(doc)
        if len(new_docs) >= max_docs:
            break

    r["relevant_articles"] = new_articles
    r["relevant_docs"] = new_docs

    after_articles += len(new_articles)
    after_docs += len(new_docs)

with OUTPUT_RESULTS.open("w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(OUTPUT_RESULTS, arcname="results.json")

print(f"Done: {len(records)} records")
print(f"docs: {before_docs} -> {after_docs}")
print(f"articles: {before_articles} -> {after_articles}")
print("results:", OUTPUT_RESULTS)
print("zip:", OUTPUT_ZIP)
```

Cell 3:

```python
import json, zipfile

records = json.loads(OUTPUT_RESULTS.read_text(encoding="utf-8"))
assert len(records) == 2000, f"Số record không phải 2000: {len(records)}"

with zipfile.ZipFile(OUTPUT_ZIP) as z:
    assert z.namelist() == ["results.json"], z.namelist()

print("OK để nộp:", OUTPUT_ZIP)
```

Kết quả mong đợi:

```text
Done: 2000 records
docs: 7069 -> 4315
articles: 12612 -> 5638
```

File nộp:

```text
/content/drive/MyDrive/sme-legal-data/submissions/private_candidate_filtered/submission.zip
```

### 6.2. Dùng script trong repo

```bash
git clone --branch codex/reduce-submission-citation-noise \
  https://github.com/dtbackjisu209/sme-legal-assistant.git

cd sme-legal-assistant/ai_legal_assistant

python scripts/postprocess_submission_citations.py \
  /content/drive/MyDrive/sme-legal-data/submissions/private_candidate/results.json \
  --output-dir /content/drive/MyDrive/sme-legal-data/submissions/private_candidate_filtered
```

Kiểm tra file nộp:

```bash
python scripts/validate_submission.py \
  /content/drive/MyDrive/sme-legal-data/submissions/private_candidate_filtered/results.json \
  --questions "/content/drive/MyDrive/sme-legal-data/R2AIStage1DATA (1).json" \
  --zip /content/drive/MyDrive/sme-legal-data/submissions/private_candidate_filtered/submission.zip
```

## 7. Cách chạy lại pipeline đầy đủ từ câu hỏi

Quy trình này sinh lại `results.json` từ tập câu hỏi, vector store và model. Kết quả có thể không trùng tuyệt đối với file đã nộp do model sinh và retrieval/reranking được chạy lại.

### 7.1. Chuẩn bị dữ liệu

Đặt các file vào:

```text
ai_legal_assistant/data/raw/R2AIStage1DATA (1).json
ai_legal_assistant/data/processed/law_articles.jsonl
ai_legal_assistant/data/indexes/bm25_index.pkl
```

Restore Qdrant collection `law_chunks_qwen3_06b` từ snapshot:

```text
law_chunks_qwen3_06b-480699765925131-2026-06-20-04-01-21.snapshot
```

Collection cần có:

```text
points_count = 248500
```

### 7.2. Lệnh chạy

```bash
cd ai_legal_assistant

python -u scripts/generate_submission.py \
  --retrieval-mode auto \
  --retrieval-top-k 8 \
  --answer-model Qwen/Qwen3-4B \
  --answer-load-in-4bit \
  --embedding-device cuda \
  --planner-device cuda \
  --reranker-device cuda \
  --answer-device cuda \
  --answer-batch-size 2 \
  --checkpoint-every 10 \
  --answer-max-input-tokens 8192 \
  --answer-max-new-tokens 900 \
  --simple-citation-docs 2 \
  --simple-citation-articles 2 \
  --multi-citation-docs 3 \
  --multi-citation-articles 5 \
  --output-dir data/submissions/private_candidate \
  --checkpoint-path data/submissions/private_candidate/partial_results_rerun_8192_900.json
```

Sau khi chạy xong, script tự tạo:

```text
data/submissions/private_candidate/results.json
data/submissions/private_candidate/submission.zip
```

Nếu cần tái lập đúng bản `0.3801`, dùng file `results.json` đã sinh từ lần chạy gốc và thực hiện lại bước post-process ở mục 6.

## 8. Kiểm tra định dạng file nộp

File ZIP hợp lệ phải chứa duy nhất:

```text
results.json
```

Kiểm tra bằng Python:

```python
import json, zipfile
from pathlib import Path

results_path = Path("data/submissions/private_candidate_filtered/results.json")
zip_path = Path("data/submissions/private_candidate_filtered/submission.zip")

records = json.loads(results_path.read_text(encoding="utf-8"))
assert len(records) == 2000

with zipfile.ZipFile(zip_path) as z:
    assert z.namelist() == ["results.json"]
```

## 9. Ghi chú về tính tái lập

- Bước post-process citation là deterministic và không dùng model.
- Để tái lập đúng submission đã nộp, cần dùng đúng `results.json` gốc trước post-process.
- Chạy lại toàn bộ pipeline có thể tạo câu trả lời và thứ tự citation khác, nên điểm leaderboard có thể khác.
- File `submission.zip` cuối cùng phải nén trực tiếp `results.json` ở root, không chứa thư mục con.
