



````markdown
 GPT-2 Finance Q&A Model

This is a fine-tuned GPT-2 model for answering finance-related questions.  
It can be used for Q&A tasks.

---

 Model Details

- Model type: GPT-2 (decoder-only)
- Fine-tuned on: Finance Q&A dataset
- Tokenizer: GPT-2 tokenizer
- Hub repository: [https://huggingface.co/peaceboii/gpt2-finance-qa](https://huggingface.co/peaceboii/gpt2-finance-qa)
- License: apache2.0

---

 Installation

Install required packages:

```bash
pip install transformers torch streamlit
````

---

## Usage

### 1. Load the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "peaceboii/gpt2-finance-qa"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 2. Generate answers

```python
question = "What is quantitative easing?"

inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Answer:", answer)
```

---

## Interactive Demo (Streamlit)

1. Create a Python file `ui.py`:

```python
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "peaceboii/gpt2-finance-qa"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

st.title("Finance Q&A GPT-2 Demo")

question = st.text_input("Enter your finance question:")

if question:
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.markdown(f"**Answer:** {answer}")
```

2. Run the app locally:

```bash
streamlit run ui.py
```

3. Access the interactive interface at `https://gpt2-finance.streamlit.app/`.

---

## Notes

* The model is fine-tuned specifically on finance Q\&A, so accuracy is highest for finance-related queries.
* Large checkpoints are hosted on Hugging Face Hub;
* For private models, provide `use_auth_token=True` when loading the model.

---

## Contact

For issues or questions, contact **kumaravelu/ kumaravelu2003@gmail.com**.



