# SLM Pháp Luật Việt Nam

Fine-tuning mô hình ngôn ngữ nhỏ (Qwen3-4B) cho các tác vụ suy luận pháp lý tiếng Việt.

## Giới Thiệu

Dự án nghiên cứu fine-tuning Small Language Model cho các tác vụ suy luận pháp lý tiếng Việt, dựa trên bộ dữ liệu [VLSP2025-LegalSML](https://huggingface.co/datasets/VLSP2025-LegalSML/Public-Test):

1. **Multiple Choice (MC)**: Trả lời câu hỏi trắc nghiệm pháp luật
2. **Natural Language Inference (NLI)**: Xác định điều luật có trả lời được câu hỏi không
3. **Syllogism (Tam đoạn luận)**: Phân tích tình huống pháp lý theo cấu trúc tam đoạn luận

## Tài Liệu Tham Khảo

Phương pháp của dự án được xây dựng dựa trên hai báo cáo dẫn đầu cuộc thi VLSP2025-LegalSML:

| Đội | Hạng | Score | Báo cáo |
|-----|------|-------|---------|
| Bosch@AI | 1 | 81.08% | [PDF](https://aclanthology.org/2025.vlsp-1.22.pdf) |
| MinLegal | 2 | 85.24% | [PDF](https://aclanthology.org/2025.vlsp-1.24.pdf) |

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
│   └── generated/            # Dữ liệu tổng hợp (training)
├── notebooks/
│   └── *.ipynb               # Notebooks training
├── report/                   # Báo cáo LaTeX
└── requirements.txt
```

## Sử Dụng

### 1. Dữ liệu

**Training data** (2346 samples sinh từ Claude Haiku 4.5):
- Kaggle: [wtihds4/vietnamese-legal-small-language](https://www.kaggle.com/datasets/wtihds4/vietnamese-legal-small-language)
- MC: 894 | NLI: 900 | Syllogism: 552

**Evaluation data** (440 samples từ cuộc thi):
- HuggingFace: [VLSP2025-LegalSML/Public-Test](https://huggingface.co/datasets/VLSP2025-LegalSML/Public-Test)

Nếu muốn sinh lại hoặc thêm dữ liệu training:
```bash
# Cần ANTHROPIC_API_KEY trong .env
python src/data_generation.py
```

### 2. Training

1. Tạo notebook trên Kaggle
2. Thêm dataset `wtihds4/vietnamese-legal-small-language`
3. Chạy notebook: `notebooks/qwen3-4b-legal-qlora-multitask-finetune.ipynb`

## Yêu Cầu

- Python 3.10+
- CUDA (cho training)
- Anthropic API key (cho sinh dữ liệu và đánh giá syllogism)

## License

Dự án phục vụ mục đích nghiên cứu và học thuật.
