#  GPT-2 - Minimal Transformer Language Model


---

## Features

* Custom Transformer Decoder architecture (GPT-style)
*  Multi-head self-attention and positional encoding
*  Tokenization using HuggingFace GPT-2 tokenizer
*  Train/test split from local TinyStories dataset
*  Cross-entropy loss with padding token masking
* Gradient accumulation support
* Perplexity evaluation
*  Greedy text generation

---

##  Dataset

We use the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), a cleaned corpus of short, simple stories suitable for language modeling experiments.

You should first download and cache it locally using:

```python
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train")
dataset.save_to_disk("./tinystories_local")
```

Then in the main script:

```python
from datasets import load_from_disk
dataset = load_from_disk("./tinystories_local")
```

---

##  Training

To train the model on 10% of the dataset (90% train / 10% test split):

```bash
python gpt2_tinystories.py
```

**Training settings:**

* Epochs: 5
* Batch Size: 8
* Learning Rate: 3e-4
* Gradient Accumulation Steps: 2

After each epoch:

* Model checkpoint saved
* Sample text generated
* Training loss/accuracy plotted

---

## 📈 Evaluation

Model is evaluated using **perplexity**:

```python
def compute_perplexity(model, test_loader):
    ...
```

Lower perplexity means better language modeling performance.

---

##  Text Generation

Use greedy decoding or sampling:

```python
def generate_text(model, prompt, max_length=50, method="greedy"):
    ...
```

**Example:**

```
Prompt: "Once upon a time, in a small village"
Generated: "Once upon a time, in a small village, there was a little boy named Max who loved to build robots..."
```

---

## Model Architecture

* Embedding + Positional Encoding
* 2 Transformer Decoder Layers:

  * Multi-Head Self-Attention
  * Feed-Forward Network
  * LayerNorm + Residual
* Linear Output Head

Approx. 86M parameters (GPT-2 tiny config).

---

## Dependencies

Install required packages:

```bash
pip install torch transformers datasets matplotlib
```

---

##  Load + Generate

Use `gpt2_generate.py` to load a trained model and generate text:

```bash
python gpt2_generate.py
```

Edit the model path and prompts as needed.

---

## Results

Training metrics per epoch are plotted and saved automatically:

![Training Metrics](https://github.com/user-attachments/assets/c3ff5854-cbfd-4bc9-9b9d-2259162da87b)

You can also review generated text samples.

---

##  Notes

* This is a minimal, educational GPT-2-style model.
* Use HuggingFace Transformers for production-level deployment.
* Extend it with top-k sampling, more layers, or longer sequences.

---

## 📚 References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [GPT-2 Paper](https://openai.com/research/language-unsupervised)
* [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
* [HuggingFace Transformers](https://huggingface.co/transformers)
* [PyTorch Docs](https://pytorch.org/docs/stable/index.html)

---

## ✨ Author

**khaled ghalwash**
Computer and Data Sciences | Alexandria National University
AI/ML Enthusiast • Deep Learning • NLP • Transformers
[GitHub](https://github.com/mohamedashraff22) | [LinkedIn](https://www.linkedin.com/in/mohameed-ashraf/)

---

## 📝 License

This project is released under the MIT License.

