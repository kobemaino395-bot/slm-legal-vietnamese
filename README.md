# SLM Pháp Luật Việt Nam

Fine-tuning mô hình ngôn ngữ nhỏ (Qwen3-4B) cho các tác vụ suy luận pháp lý tiếng Việt.

## Giới Thiệu

Dự án nghiên cứu fine-tuning Small Language Model cho các tác vụ suy luận pháp lý tiếng Việt:

1. **Multiple Choice (MC)**: Trả lời câu hỏi trắc nghiệm pháp luật
2. **Natural Language Inference (NLI)**: Xác định điều luật có trả lời được câu hỏi không
3. **Syllogism (Tam đoạn luận)**: Phân tích tình huống pháp lý theo cấu trúc tam đoạn luận

## Phương Pháp

### Sinh Dữ Liệu Tổng Hợp

Sử dụng Claude Haiku 4.5 với 2 chiến lược:
- **Strategy A (MinLegal)**: Few-shot sampling
- **Strategy B (Bosch)**: Aspect-based Chain-of-Thought

### Fine-tuning 2 Giai Đoạn

1. **Giai đoạn 1**: Fine-tune riêng từng task (MC, NLI, Syllogism)
2. **Giai đoạn 2**: Gộp adapters và fine-tune trên tập dữ liệu hỗn hợp

Hyperparameters:
- LoRA rank: 16
- LoRA alpha: 32
- Stage 1: lr=2e-4, epochs=3
- Stage 2: lr=5e-5, epochs=2

## Kết Quả

| Task | Accuracy |
|------|----------|
| Multiple Choice | 86.99% |
| NLI | 89.33% |
| Syllogism | 56.70% |
| **Trung bình** | **77.67%** |

## Cài Đặt

```bash
pip install -r requirements.txt
```

## Cấu Trúc Thư Mục

```
├── src/
│   └── data_generation.py    # Script sinh dữ liệu
├── data/
│   ├── original/             # Dữ liệu gốc (evaluation)
│   └── generated/            # Dữ liệu tổng hợp (training)
├── notebooks/
│   └── kaggle/               # Notebooks training trên Kaggle
├── report/                   # Luận án LaTeX
└── requirements.txt
```

## Sử Dụng

### 1. Dữ liệu training

Đã có sẵn 2346 samples trong `data/generated/`:
- MC: 894 samples
- NLI: 900 samples
- Syllogism: 552 samples

Nếu muốn sinh lại hoặc thêm dữ liệu:
```bash
# Cần ANTHROPIC_API_KEY trong .env
python src/data_generation.py
```

### 2. Training

Xem notebook: `notebooks/qwen3-4b-legal-qlora-multitask-finetune.ipynb`

## Yêu Cầu

- Python 3.10+
- CUDA (cho training)
- Anthropic API key (cho sinh dữ liệu)

## License

Dự án phục vụ mục đích nghiên cứu và học thuật.
