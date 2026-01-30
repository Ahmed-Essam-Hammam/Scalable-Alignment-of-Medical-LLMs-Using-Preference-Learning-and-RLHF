# Scalable-Alignment-of-Medical-LLMs-Using-Preference-Learning-and-RLHF

A comprehensive research project comparing four modern alignment techniques for Large Language Models in medical question-answering, using Llama 3.2 3B as the base model.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models](#models)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements and compares **four distinct alignment techniques** for fine-tuning Large Language Models (LLMs) on medical question-answering tasks:

1. **Model A**: Supervised Fine-Tuning (SFT) - Baseline
2. **Model B**: Human-in-the-Loop (HITL) - Iterative feedback
3. **Model C**: Reinforcement Learning from Human Feedback (RLHF) with PPO
4. **Model D**: Direct Preference Optimization (DPO)

**Base Model**: Llama 3.2 3B  
**Domain**: Medical Question-Answering  
**Dataset Size**: 400K+ medical samples + preference pairs

---

## 📁 Project Structure

```
MODEL_B_HITL/
├── notebooks/
│   ├── lora-on-ultramedical_Model_A.ipynb          # Model A: SFT training
│   ├── rlhf-with-reward-model-ppo_Model_C.ipynb    # Model C: RLHF/PPO training
│   └── direct-preference-optimization-fine-tuning_Model_D.ipynb  # Model D: DPO training
│
├── model_b_hitl/                                    # Model B: HITL system
│   ├── config.py                                    # Configuration settings
│   ├── model_manager.py                             # Model loading & generation
│   ├── feedback_manager.py                          # Feedback collection & storage
│   ├── trainer.py                                   # Incremental training logic
│   ├── gradio_ui.py                                 # Web interface
│   ├── main.py                                      # Application entry point
│   └── requirements.txt                             # Python dependencies
│
├── models/                                          # Trained model checkpoints
│   ├── llama3.2-3b-ultramedical-lora/              # Model A (SFT baseline)
│   ├── Model_SFT_3B.zip                            # Compressed Model A
│   ├── Reward_Model_3B.zip                         # Trained reward model (for Model C)
│   └── [Model B/C/D outputs generated during training]
│
├── utils/
│   ├── check_hf_access.py                          # HuggingFace authentication helper
│   └── .env.example                                # Environment variables template
│
├── .env                                            # Environment variables (not in git)
├── .gitignore                                      # Git ignore rules
└── README.md                                       # This file
```

---

## 🤖 Models

### Model A: Supervised Fine-Tuning (SFT)
**Purpose**: Baseline model trained with standard supervised learning

**Training Method**:
- LoRA fine-tuning on Llama 3.2 3B
- Trained on UltraMedical instruction dataset
- Parameter-efficient: ~20M trainable parameters (0.5% of total)

**Notebook**: `lora-on-ultramedical_Model_A.ipynb`

**Key Features**:
- 4-bit quantization for memory efficiency
- Gradient checkpointing
- LoRA config: r=16, alpha=32, dropout=0.05

**Output**: `llama3.2-3b-ultramedical-lora/`

---

### Model B: Human-in-the-Loop (HITL)
**Purpose**: Iteratively improve Model A through direct human feedback

**Training Method**:
- Interactive web interface for feedback collection
- Human accepts or edits model responses
- Automatic incremental LoRA retraining every N samples
- Continuous alignment with human preferences

**Components**:
- `main.py`: Launch the HITL application
- `gradio_ui.py`: Web interface for feedback collection
- `feedback_manager.py`: Data storage and dataset management
- `trainer.py`: Incremental training logic

**Key Features**:
- Real-time feedback collection via Gradio interface
- Automatic retraining triggers (default: every 20 samples)
- Feedback stored in JSONL format for reproducibility
- Version control for each training iteration

**Usage**:
```bash
cd model_b_hitl
python main.py
```

**Output**: `model_b_hitl/llama3.2-3b-ultramedical-model-b/`

---

### Model C: RLHF with PPO
**Purpose**: Align model using reinforcement learning from human preferences

**Training Method**:
1. Train reward model on preference pairs (UltraMedical-Preferences)
2. Use PPO algorithm to optimize policy against reward model
3. KL divergence penalty to prevent drift from base model

**Notebook**: `rlhf-with-reward-model-ppo_Model_C.ipynb`

**Key Features**:
- Two-stage training: Reward model → PPO optimization
- Manual training loop for stability
- CPU offloading for memory efficiency
- Comprehensive logging and checkpointing

**Components**:
- Reward Model: `Reward_Model_3B.zip`
- Policy Model: Optimized with PPO
- Reference Model: Frozen copy for KL penalty
- Value Model: Estimates future rewards

**Output**: `model_c_ppo_final/`

---

### Model D: Direct Preference Optimization (DPO)
**Purpose**: Modern preference-based alignment without reward modeling

**Training Method**:
- Direct optimization on preference pairs
- No separate reward model needed
- More stable and efficient than PPO

**Notebook**: `direct-preference-optimization-fine-tuning_Model_D.ipynb`

**Key Features**:
- Single-stage training (simpler than RLHF)
- DPO loss with β temperature parameter
- Automatic reference model creation
- Mixed precision training (AMP)

**Advantages over PPO**:
- ✅ Faster training (30-60 min vs 2-3 hours)
- ✅ More stable gradients
- ✅ Lower memory footprint
- ✅ Often better final performance

**Output**: `model_d_dpo_final/`

---

## 📊 Datasets

### UltraMedical
**Purpose**: Instruction fine-tuning dataset for medical QA

**Size**: 400,000+ examples

**Format**:
```json
{
  "id": "sample_id",
  "type": "medical_qa",
  "conversations": [
    {"from": "human", "value": "Medical question..."},
    {"from": "gpt", "value": "Medical answer..."}
  ],
  "answer": "Correct answer",
  "score": 5.0
}
```

**Used in**: Model A (SFT), Model B (HITL)

**Source**: [Specify source if public, or note as proprietary]

---

### UltraMedical-Preferences
**Purpose**: Preference pairs for alignment training

**Size**: [Specify number of preference pairs]

**Format**:
```json
{
  "prompt": "Medical question...",
  "chosen": [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "better response"}
  ],
  "rejected": [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "worse response"}
  ]
}
```

**Used in**: Model C (RLHF), Model D (DPO)

**Source**: [Specify source if public, or note as proprietary]

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- HuggingFace account with Llama access

### Step 1: Clone Repository
```bash
git clone https://github.com/Ahmed-Essam-Hammam/Scalable-Alignment-of-Medical-LLMs-Using-Preference-Learning-and-RLHF.git
cd MODEL_B_HITL
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup HuggingFace Authentication
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your HuggingFace token
# HF_TOKEN=hf_your_token_here
```

Or use the helper script:
```bash
python check_hf_access.py
```

### Step 5: Accept Llama License
Visit https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and accept the license.

---

## 💻 Usage

### Training Model A (SFT)
```bash
# Open Jupyter notebook
jupyter notebook lora-on-ultramedical_Model_A.ipynb

# Or run in Kaggle/Colab with GPU enabled
```

**Expected Duration**: 2-3 hours on T4 GPU

---

### Training Model B (HITL)

#### Step 1: Start the Application
```bash
cd model_b_hitl
python main.py
```

#### Step 2: Access Web Interface
Open your browser to `http://localhost:7860`

#### Step 3: Collect Feedback
1. Enter a medical question
2. Generate response
3. Accept or edit the response
4. System auto-retrains after 20 samples (configurable)

#### Step 4: Monitor Progress
- Feedback saved to `human_feedback_data.jsonl`
- Model checkpoints: `model_b/llama3.2-3b-ultramedical-model-b_iter_N/`

**Configuration Options**:
```bash
python main.py --retrain-threshold 50  # Retrain every 50 samples
python main.py --no-auto-retrain       # Disable auto-retraining
python main.py --share                 # Create public Gradio link
```

---

### Training Model C (RLHF/PPO)
```bash
# Upload to Kaggle and run notebook
rlhf-with-reward-model-ppo_Model_C.ipynb
```

**Requirements**:
- GPU with 16GB+ VRAM
- Kaggle/Colab environment recommended

**Steps**:
1. Train reward model (~30-60 min)
2. Run PPO training (~1-2 hours)

**Expected Duration**: 2-3 hours total on T4 GPU

---

### Training Model D (DPO)
```bash
# Upload to Kaggle and run notebook
direct-preference-optimization-fine-tuning_Model_D.ipynb
```

**Requirements**:
- GPU with 16GB VRAM (single T4 sufficient)

**Expected Duration**: 30-60 minutes on T4 GPU

---

## 📈 Results

### Performance Comparison

| Model | Method | Training Time | GPU Memory | Trainable Params |
|-------|--------|---------------|------------|------------------|
| **A** | SFT | 7-9 hours | 8GB | 9M (0.5%) |
| **B** | HITL | Iterative | 8GB | 9M (0.5%) |
| **C** | RLHF/PPO | 2-3 hours | 16GB | 9M (0.5%) |
| **D** | DPO | 2-3 hours | 8GB | 9M (0.5%) |


**Recommended Metrics**:
- Medical accuracy
- Response quality (human evaluation)
- Safety/hallucination rate
- BLEU/ROUGE scores
- BERTScore

---

## 🔧 Technical Details

### Optimization Techniques

#### LoRA (Low-Rank Adaptation)
```python
LoraConfig(
    r=16,                # Rank
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.05,   # Dropout rate
    target_modules=[     # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

**Benefits**:
- 99%+ reduction in trainable parameters
- Faster training and inference
- Lower memory footprint
- Easy model merging and switching

---

#### 4-bit Quantization
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

**Benefits**:
- 75% memory reduction
- Enables 3B model on 16GB GPU
- Minimal performance degradation

---

#### Gradient Checkpointing
- Trades computation for memory
- Enables larger batch sizes
- Essential for training on consumer GPUs

---

### Model Architecture

**Base Model**: Llama 3.2 3B
- **Parameters**: 3.21 billion
- **Context Length**: 128K tokens
- **Vocabulary**: 128,256 tokens
- **Architecture**: Decoder-only transformer

**After LoRA**:
- **Trainable**: ~20 million (0.6%)
- **Frozen**: ~3.19 billion (99.4%)

---

### Training Configuration

#### Model A (SFT)
```python
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-4
lr_scheduler: "cosine"
warmup_ratio: 0.1
max_seq_length: 2048
```

#### Model B (HITL)
```python
num_train_epochs: 1  # Per iteration
learning_rate: 1e-4  # Lower for stability
retrain_threshold: 20  # Samples before retrain
```

#### Model C (RLHF)
```python
# Reward Model
learning_rate: 1e-5
beta: 0.1  # Temperature

# PPO
learning_rate: 1e-5
batch_size: 16
mini_batch_size: 1
```

#### Model D (DPO)
```python
learning_rate: 5e-6
beta: 0.1  # DPO temperature
max_length: 1024
warmup_ratio: 0.1
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Solution**:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable gradient checkpointing
- Use smaller `max_seq_length`

#### 2. HuggingFace Authentication Error
**Solution**:
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token in .env file
HF_TOKEN=hf_your_token_here
```

#### 3. Model Not Loading
**Solution**:
- Verify HuggingFace access to Llama 3.2
- Check `MODEL_SFT_PATH` in config
- Ensure model files exist and aren't corrupted

#### 4. Training Stuck/Not Starting
**Solution**:
- Check GPU is actually enabled (Kaggle/Colab settings)
- Verify data format matches expected structure
- Use manual training loops instead of Trainer API
- Monitor GPU usage with `nvidia-smi`

---

## 📚 References

### Papers
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **RLHF**: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **Llama**: [Llama 3 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

### Libraries
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd MODEL_B_HITL

# Create development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install pre-commit hooks (if available)
pre-commit install
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The Llama 3.2 model is subject to Meta's license agreement. Please review and comply with the [Llama 3 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

---

## 🙏 Acknowledgments

- **Meta AI** for the Llama 3.2 model
- **HuggingFace** for the Transformers, PEFT, and TRL libraries
- **UltraMedical Dataset** creators [Add attribution if known]
- **Research community** for alignment technique papers

---

## 📧 Contact

For questions or collaboration opportunities:
- **Email**: ahmedessamhamam2@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/ahmed-essam-a681a134b/

---

## 🔮 Future Work

- [ ] Quantitative evaluation on medical benchmarks
- [ ] Human evaluation study with medical professionals
- [ ] Comparison with GPT-4 on medical tasks
- [ ] Fine-tuning on domain-specific medical subfields
- [ ] Deployment as API service
- [ ] Integration with medical knowledge bases

---

**⭐ If you find this project useful, please consider giving it a star!**

---
