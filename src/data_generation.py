# Mục đích: Sinh dữ liệu huấn luyện tổng hợp cho SLM pháp luật Việt Nam
#           bằng Anthropic Claude Haiku 4.5 theo 2 chiến lược:
#           - Strategy A (MinLegal): Few-shot sampling
#           - Strategy B (Bosch@AI): Aspect-based Chain-of-Thought
#
# Tham khảo:
#   - MinLegal: https://aclanthology.org/2025.vlsp-1.24.pdf
#   - Bosch@AI: https://aclanthology.org/2025.vlsp-1.22.pdf
#
# Yêu cầu:
#   - Python 3.10+
#   - pip install anthropic datasets python-dotenv
#   - ANTHROPIC_API_KEY trong environment (.env) hoặc Colab Secrets
#
# Cách chạy:
#   python src/data_generation.py
#
# Output:
#   data/train_mc.jsonl
#   data/train_nli.jsonl
#   data/train_syllogism.jsonl
#
# Lưu ý:
#   - Dữ liệu evaluation lấy trực tiếp từ HuggingFace: VLSP2025-LegalSML/Public-Test
#   - Script này chỉ sinh dữ liệu training, không lưu dữ liệu gốc

import os
import json
import random
import re
import time
import hashlib
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # Anthropic API
    "model": "claude-haiku-4-5-20251001",
    "rate_limit_sec": 1.0,
    "max_retries": 4,
    "max_tokens": 4096,

    # Mục tiêu sinh dữ liệu (tổng ~2400)
    "target_mc": 900,
    "target_nli": 900,
    "target_syllogism": 600,

    # Few-shot config (Strategy A)
    "fewshot_n": 5,
    "samples_per_call": 6,

    # Checkpoint
    "checkpoint_every": 30,

    # Paths
    "output_dir": "data",
    "checkpoint_dir": "data/checkpoints",

    # Dataset nguồn (VLSP2025-LegalSML Public Test)
    "source_dataset": "VLSP2025-LegalSML/Public-Test",

    # Seed
    "seed": 99,
}

SYSTEM_PROMPT = "Bạn là một chuyên gia pháp luật Việt Nam."

# ═══════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════

random.seed(CONFIG["seed"])

for d in [CONFIG["output_dir"], CONFIG["checkpoint_dir"]]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# ANTHROPIC API CLIENT
# ═══════════════════════════════════════════════════════════════

print("Initializing Anthropic API...")

CLIENT = None
try:
    import anthropic

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("ANTHROPIC_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found! Set env var, .env, or Colab Secret.")

    CLIENT = anthropic.Anthropic(api_key=api_key)
    print(f"[OK] Anthropic API ready (model: {CONFIG['model']})")

except Exception as e:
    print(f"[ERROR] Failed to initialize Anthropic: {e}")
    CLIENT = None

# Rate limiting
_last_call_time = [0.0]

def claude_call(prompt: str, system: str = None) -> str | None:
    """Gọi Claude API với rate limiting và retry."""
    if CLIENT is None:
        return None

    for attempt in range(CONFIG["max_retries"]):
        try:
            elapsed = time.time() - _last_call_time[0]
            if elapsed < CONFIG["rate_limit_sec"]:
                time.sleep(CONFIG["rate_limit_sec"] - elapsed)
            _last_call_time[0] = time.time()

            kwargs = {
                "model": CONFIG["model"],
                "max_tokens": CONFIG["max_tokens"],
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            response = CLIENT.messages.create(**kwargs)
            return response.content[0].text

        except Exception as e:
            wait = CONFIG["rate_limit_sec"] * (2 ** attempt)
            print(f"  [Retry {attempt+1}/{CONFIG['max_retries']}] {type(e).__name__}: {e} -> waiting {wait:.0f}s")
            time.sleep(wait)

    return None

def extract_json(text: str) -> list | dict | None:
    """Trích JSON từ phản hồi API."""
    if not text:
        return None

    text = re.sub(r"```(?:json)?", "", text).strip("` \n")

    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return None

# ═══════════════════════════════════════════════════════════════
# DATA LOADING (từ VLSP2025-LegalSML Public Test)
# ═══════════════════════════════════════════════════════════════

print(f"\nLoading source data from {CONFIG['source_dataset']}...")

from datasets import load_dataset

mc_raw = load_dataset(CONFIG["source_dataset"], "multichoice_questions", split="train")
nli_raw = load_dataset(CONFIG["source_dataset"], "nli_questions", split="train")
syllo_raw = load_dataset(CONFIG["source_dataset"], "syllogism_questions", split="train")

print(f"  MC: {len(mc_raw)} | NLI: {len(nli_raw)} | Syllogism: {len(syllo_raw)}")

def format_mc(ex):
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"]))
    user = f"Câu hỏi: {ex['question']}\n\nCác lựa chọn:\n{opts}\n\nChỉ trả lời A, B, C hoặc D."
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": letters[ex["answer"]]},
        ],
        "task": "mc",
        "answer": ex["answer"],
    }

def format_nli(ex):
    user = f"Điều luật:\n{ex['legal_document']}\n\nCâu hỏi: {ex['specific_question']}\n\nĐiều luật có trả lời được câu hỏi không? Chỉ trả lời 'Có' hoặc 'Không'."
    answer = "Có" if ex["answer"] == 0 else "Không"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "task": "nli",
        "answer": ex["answer"],
    }

def format_syllo(ex):
    user = f"Tình huống pháp lý:\n{ex['question']}\n\nHãy phân tích theo cấu trúc tam đoạn luận:\nTiền đề lớn: ...\nTiền đề nhỏ: ...\nKết luận: ..."
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": ""},
        ],
        "task": "syllogism",
        "answer": None,
    }

mc_fmt = [format_mc(x) for x in mc_raw]
nli_fmt = [format_nli(x) for x in nli_raw]
syllo_fmt = [format_syllo(x) for x in syllo_raw]

print(f"  Formatted: MC={len(mc_fmt)}, NLI={len(nli_fmt)}, Syllo={len(syllo_fmt)}")

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def save_jsonl(records: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_user_text(rec: dict) -> str:
    return next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")

def get_assistant_text(rec: dict) -> str:
    return next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), "")

def fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

LETTER2IDX = {"A": 0, "B": 1, "C": 2, "D": 3}

def normalize_answer(task: str, assistant: str):
    """Kiểm tra & chuẩn hóa đáp án."""
    a = (assistant or "").strip()

    if task == "mc":
        if a in LETTER2IDX:
            return a, LETTER2IDX[a]
        m = re.search(r"ĐÁP\s*:\s*([ABCD])", a, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            return letter, LETTER2IDX[letter]
        m = re.match(r"^\(?([ABCD])[\.\):\s]", a)
        if m:
            letter = m.group(1).upper()
            return letter, LETTER2IDX[letter]
        return None

    if task == "nli":
        first = re.split(r"[\s,.\n:;]+", a)[0].lower() if a else ""
        if first == "có":
            return "Có", 0
        if first == "không":
            return "Không", 1
        return None

    return None

def strip_fewshot_labels(text: str) -> str:
    """Làm sạch nhãn few-shot."""
    if not text:
        return text
    cleaned = re.sub(r"VÍ\s*DỤ\s*\d+\s*:", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"Đ[ỀE]\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"ĐÁP\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

# ═══════════════════════════════════════════════════════════════
# STRATEGY A: FEW-SHOT SAMPLING (MinLegal approach)
# ═══════════════════════════════════════════════════════════════

def strategy_a_prompt(task: str, examples: list) -> str:
    task_desc = {
        "mc": "câu hỏi trắc nghiệm pháp luật Việt Nam với 4 lựa chọn (A, B, C, D) và đáp án đúng",
        "nli": "cặp (điều luật, câu hỏi) và nhãn 'Có' hoặc 'Không' cho biết điều luật có thể trả lời câu hỏi hay không",
    }[task]

    examples_text = "\n\n".join(
        f"VÍ DỤ {i+1}:\nĐỀ: {get_user_text(ex)}\nĐÁP: {get_assistant_text(ex)}"
        for i, ex in enumerate(examples)
    )

    return f"""Bạn là chuyên gia tạo dữ liệu pháp luật Việt Nam. Dưới đây là các ví dụ về {task_desc}.

{examples_text}

Hãy tạo {CONFIG['samples_per_call']} mẫu MỚI cùng định dạng, ĐA DẠNG lĩnh vực pháp luật (thuế, đất đai, lao động, giao thông, bảo hiểm xã hội, hình sự, dân sự, hôn nhân gia đình, doanh nghiệp...).

Yêu cầu:
- Nội dung phải chính xác theo pháp luật Việt Nam hiện hành
- Đa dạng về độ khó (dễ, trung bình, khó)
- Không trùng lặp với các ví dụ đã cho

Trả về JSON array, mỗi phần tử có 2 key:
- "user": nội dung đề bài
- "assistant": đáp án"""

def generate_strategy_a(task: str, pool: list, target: int) -> list:
    ckpt_path = f"{CONFIG['checkpoint_dir']}/strA_{task}.jsonl"
    output = load_jsonl(ckpt_path)

    print(f"\n[Strategy A / {task}]")
    print(f"  Resume: {len(output)} | Target: {target}")

    stale = 0
    rejected = 0
    while len(output) < target:
        examples = random.sample(pool, min(CONFIG["fewshot_n"], len(pool)))
        prompt = strategy_a_prompt(task, examples)

        response = claude_call(prompt)
        items = extract_json(response) or []

        added = 0
        for item in items:
            if isinstance(item, dict) and item.get("user") and item.get("assistant"):
                norm = normalize_answer(task, str(item["assistant"]))
                if norm is None:
                    rejected += 1
                    continue
                assistant_clean, answer = norm
                user_clean = strip_fewshot_labels(str(item["user"]))
                if not user_clean:
                    rejected += 1
                    continue
                output.append({
                    "user": user_clean,
                    "assistant": assistant_clean,
                    "task": task,
                    "answer": answer,
                    "source": "strategy_a",
                })
                added += 1

        if added == 0:
            stale += 1
            print(f"  [!] Empty batch ({stale})")
            if CLIENT is None or stale >= 5:
                break
        else:
            stale = 0

        if len(output) % CONFIG["checkpoint_every"] < CONFIG["samples_per_call"]:
            save_jsonl(output, ckpt_path)
            print(f"  Progress: {len(output)}/{target}")

    save_jsonl(output, ckpt_path)
    print(f"  [OK] Generated: {len(output)} samples (rejected {rejected})")
    return output[:target]

# ═══════════════════════════════════════════════════════════════
# STRATEGY B: ASPECT-BASED CoT (Bosch@AI approach)
# ═══════════════════════════════════════════════════════════════

ASPECT_SYSTEM = "Bạn là một chuyên gia pháp luật Việt Nam."

def aspect_extraction_prompt(legal_doc: str) -> str:
    return f"""Dựa trên văn bản pháp luật được cung cấp, hãy xác định 1-3 khía cạnh pháp lý riêng biệt, điểm chính hoặc chủ đề được bao quát (ví dụ: điều kiện áp dụng, mức xử phạt, phạm vi điều chỉnh, ngoại lệ).

Nội dung văn bản:
\"\"\"{legal_doc}\"\"\"

Trả về CHỈ MỘT đối tượng JSON: {{ "aspects": ["Khía cạnh 1 ngắn gọn", "Khía cạnh 2", ...] }}"""

def cot_syllogism_prompt(legal_doc: str, aspects: list) -> str:
    aspects_text = "; ".join(aspects)
    return f"""Bạn là chuyên gia pháp luật Việt Nam chuyên tổng hợp dữ liệu. Dựa trên văn bản pháp luật và các khía cạnh đã xác định, hãy tạo các ví dụ tam đoạn luận pháp lý.

Văn bản pháp luật:
\"\"\"{legal_doc}\"\"\"

Các khía cạnh pháp lý: {aspects_text}

Với MỖI khía cạnh, tạo một câu hỏi tình huống pháp lý mở và lời giải có cấu trúc:
- Phần phân tích chi tiết đặt trong thẻ <think>...</think>
- Sau đó lần lượt ghi:
  Tiền đề lớn: <quy phạm pháp luật chung>
  Tiền đề nhỏ: <tình huống cụ thể>
  Kết luận: <hệ quả pháp lý có căn cứ>

Yêu cầu: tình huống thực tế, lập luận logic chặt chẽ, bám sát văn bản được cung cấp.

Trả về JSON array, mỗi phần tử có 2 key:
- "question": tình huống pháp lý cần phân tích
- "reasoning": lời giải đầy đủ (gồm thẻ <think>...</think> rồi Tiền đề lớn/nhỏ/Kết luận)"""

def generate_strategy_b(nli_pool: list, target: int) -> list:
    ckpt_path = f"{CONFIG['checkpoint_dir']}/strB_syllogism.jsonl"
    output = load_jsonl(ckpt_path)

    print(f"\n[Strategy B / syllogism — aspect-based CoT]")
    print(f"  Resume: {len(output)} | Target: {target}")

    docs = [get_user_text(r) for r in nli_pool]
    random.shuffle(docs)
    doc_idx = 0
    stale = 0
    rejected = 0

    while len(output) < target and doc_idx < len(docs):
        legal_doc = docs[doc_idx]
        doc_idx += 1

        aspect_resp = claude_call(aspect_extraction_prompt(legal_doc), system=ASPECT_SYSTEM)
        aspect_obj = extract_json(aspect_resp) or {}
        aspects = aspect_obj.get("aspects") if isinstance(aspect_obj, dict) else None
        if not aspects:
            aspects = ["nội dung chính của quy định"]

        response = claude_call(cot_syllogism_prompt(legal_doc, aspects), system=ASPECT_SYSTEM)
        items = extract_json(response) or []

        added = 0
        for item in items:
            if isinstance(item, dict) and item.get("question") and item.get("reasoning"):
                reasoning = str(item["reasoning"]).strip()
                if not all(kw in reasoning for kw in ("Tiền đề lớn", "Tiền đề nhỏ", "Kết luận")):
                    rejected += 1
                    continue
                output.append({
                    "user": (
                        f"Tình huống pháp lý:\n{str(item['question']).strip()}\n\n"
                        "Hãy phân tích theo cấu trúc tam đoạn luận:\n"
                        "Tiền đề lớn: ...\nTiền đề nhỏ: ...\nKết luận: ..."
                    ),
                    "assistant": reasoning,
                    "task": "syllogism",
                    "answer": None,
                    "source": "strategy_b",
                    "aspects": aspects,
                })
                added += 1

        if added == 0:
            stale += 1
            print(f"  [!] Empty batch ({stale})")
            if CLIENT is None or stale >= 5:
                break
        else:
            stale = 0

        if len(output) % CONFIG["checkpoint_every"] < 4:
            save_jsonl(output, ckpt_path)
            print(f"  Progress: {len(output)}/{target}")

    save_jsonl(output, ckpt_path)
    print(f"  [OK] Generated: {len(output)} samples (rejected {rejected})")
    return output[:target]

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print("DATA GENERATION - SLM Vietnamese Legal")
    print(f"Model: {CONFIG['model']}")
    print(f"Source: {CONFIG['source_dataset']}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    gen_mc = generate_strategy_a("mc", mc_fmt, CONFIG["target_mc"])
    gen_nli = generate_strategy_a("nli", nli_fmt, CONFIG["target_nli"])
    gen_syllo = generate_strategy_b(nli_fmt, CONFIG["target_syllogism"])

    # Merge & deduplicate
    print(f"\n[Merge & Deduplicate]")
    seen_hashes = set()
    all_generated = []
    for records in [gen_mc, gen_nli, gen_syllo]:
        for r in records:
            h = fingerprint(r["user"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_generated.append(r)

    duplicates_removed = len(gen_mc) + len(gen_nli) + len(gen_syllo) - len(all_generated)
    print(f"  Total generated: {len(gen_mc) + len(gen_nli) + len(gen_syllo)}")
    print(f"  After dedup: {len(all_generated)} (removed {duplicates_removed})")

    # Convert to chat format
    def to_chat_format(rec):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": rec["user"]},
                {"role": "assistant", "content": rec["assistant"]},
            ],
            "task": rec["task"],
            "answer": rec.get("answer"),
        }

    gen_chat = [to_chat_format(r) for r in all_generated]
    random.shuffle(gen_chat)

    # Save by task
    print(f"\n[Save Output]")
    by_task = {"mc": [], "nli": [], "syllogism": []}
    for r in gen_chat:
        by_task[r["task"]].append(r)

    for task, records in by_task.items():
        path = f"{CONFIG['output_dir']}/train_{task}.jsonl"
        save_jsonl(records, path)
        print(f"  {path}: {len(records)} samples")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Duration: {duration}")
    print(f"Generated: MC={len(by_task['mc'])} NLI={len(by_task['nli'])} Syllo={len(by_task['syllogism'])}")
    print(f"Total: {len(gen_chat)} samples")
    print(f"Output: {CONFIG['output_dir']}/")
    print(f"\nEvaluation data: load from HuggingFace {CONFIG['source_dataset']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
