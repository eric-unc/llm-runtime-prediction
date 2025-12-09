# LLM runtime prediction
This is my class project to predict the runtime/overall token output count of an LLM.

## Installation
### Ollama
```sh
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.12.6 sh
sudo systemctl disable ollama
```

### Python dependencies
```sh
python3 -m venv pred-Env
source pred-Env/bin/activate
pip install numpy pandas transformers matplotlib datasets peft
deactivate
```

Also, you will need [Hugging Face](https://huggingface.co/docs/huggingface_hub/quick-start). You will need to create a token with access to Meta's llama 3.2 and 3.3.

### Misc
This might be useful for testing.
```sh
sudo apt install sqlite3
```

## Usage
Basically for each script, you want to activate the virtual environment, run it, and then deactive it.
```sh
source pred-Env/bin/activate
python3 pearson-alpaca.py
deactivate
```

### Fine-tuning
```sh
source pred-Env/bin/activate
export HF_TOKEN=... # fill in your Hugging Face token here

python3 fine_tune_lora.py
./make_gguf.sh
./make_ollama.sh
# Can test with llama32-ft then
deactivate
```
