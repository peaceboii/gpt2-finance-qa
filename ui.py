import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource(show_spinner=False)
def load_model():
    model_name_or_path = "peaceboii/gpt2-finance-qa"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

model, tokenizer = load_model()

st.title("Finance Q&A GPT-2 Demo")

question = st.text_input("Enter your finance question:")

if question:
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.markdown(f"**Answer:** {answer}")
