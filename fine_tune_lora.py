import sqlite3
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def token_count(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def get_pair():
    ret = []
    conn = sqlite3.connect("output/alpaca.db")
    cur = conn.cursor()
    cur.execute("SELECT prompt, response FROM results WHERE run_index = 0 AND model = 'llama3.3'")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
    for prompt, response in cur.fetchall():
        training_prompt = (
            "You are given a prompt. Do not answer it. Predict the output token count that the Llama 3.3 LLM will produce when answering the prompt. Only output a number.\n\n"
            "## Prompt:\n"
            "```\n"
            f"{prompt}\n"
            "```\n\n"
            "## Output token count:\n"
        )
        expected = str(token_count(prompt, tokenizer))
        p = (training_prompt, expected)
        ret.append(p)
    print("Len: ", len(ret))
    return ret

#
# 1. Provide a generator or function that yields (input, output)
#
def data_generator():
    # Replace with your function
    for prompt, expected in get_pair():   # your own function
        yield {
            "prompt": prompt,
            "response": expected
        }

#
# 2. Build HF dataset
#
dataset = Dataset.from_generator(data_generator)

#
# 3. Load tokenizer + base model (Llama 3.2)
#
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def format_example(x):
    # Simple instruction-style format
    return f"<s>[INST] {x['prompt']} [/INST]\n{x['response']}</s>"

#def tokenize(batch):
#    formatted = [format_example(x) for x in batch["prompt"]]
#    return tokenizer(
#        formatted,
#        truncation=True,
#        max_length=2048,
#        padding="max_length"
#    )

def tokenize(batch):
    out = tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048,
        padding=False   # or True if you prefer fixed length
    )
    out["labels"] = out["input_ids"].copy()
    return out

dataset = dataset.map(lambda x: {
    "text": format_example(x)
})
tokenized = dataset.map(tokenize, batched=True)
#tokenized = dataset.map(
#    lambda x: tokenizer(
#        x["text"],
#        truncation=True,
#        max_length=2048
#    ),
#    batched=True
#)

#
# 4. Apply LoRA
#
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

#
# 5. Train
#
args = TrainingArguments(
    output_dir="lora-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    bf16=True,
    logging_steps=10,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("lora-out")

