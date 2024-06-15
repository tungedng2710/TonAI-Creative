import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
config = PeftConfig.from_pretrained("awaisakhtar/llama-2-7b-summarization-finetuned-on-xsum-lora-adapter")
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "awaisakhtar/llama-2-7b-summarization-finetuned-on-xsum-lora-adapter")

def format_chat(history, user_input):
    chat_template = ""
    for (role, text) in history:
        chat_template += f"{role}: {text}\n"
    chat_template += f"user: {user_input}\n"
    chat_template += "assistant: "
    return chat_template

def summarize_text(user_input, max_length=4096):
    history = [("system", "Summarize the text below in to a paragraph about 100 words")]
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          torch_dtype=torch.bfloat16,
                                          device_map = device)
    model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)
    chat_input = format_chat(history, user_input)
    inputs = tokenizer(chat_input, return_tensors="pt").to(device)
    outputs = model.generate(inputs['input_ids'],
                             max_length=max_length,
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = response.split("assistant: ")[-1].strip()
    return assistant_response

if __name__ == '__main__':
    with open("/root/tungn197/genAI/data/meeting_dialog/dummy_meeting_2.txt") as txtfile:
        text = txtfile.read()
    print(summarize_text(text))