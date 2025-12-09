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

Also you need `convert_lora_to_gguf.py` from llama.cpp if you want to use `make_gguf.sh`:
```sh
# Move to parent directory of this one
cd ..
git clone git@github.com:rosalab/llama.cpp.git
cd llama.cpp
git switch pred-freeze
cd ../output-pred
```

## Usage
Basically for each script, you want to activate the virtual environment, run it, and then deactive it.
```sh
source pred-Env/bin/activate
python3 pearson-alpaca.py
deactivate
```

### Specific scripts
These are in order of how I would run them for reproducability.
* `bench-example.sh`: bench the two examples given in the intro of the paper; results are saved to `output/example.db`.
* `bench-alpaca.py`: bench LLama 3.2 and 3.3 on the first 1000 elements of Alpaca dataset; results are saved to `output/alpaca.db`.
* `plot-alpaca.py`: plot prompt length/output length v median runtime, or figure 1/2 in the paper.
* `pearson-alpaca.py`: get the Pearson coefficient for that data, mentioned in the text.
* `predict-alpaca.py`: use off-the-shelf models to predict output token lengths of various prompts. Results are saved to `output/alpaca-pred.db`.
* `plot-alpaca-predictions.py`: plot those prediction accuracies; figure 4/5 in the paper, plus some extra plots.
* `fine_tune_lora.py`: fine-tune LLama 3.2 using the data in `output/alpaca.db` to make it help predict those output token counts. The model is saved to a new `lora-out` directory.
* `make_gguf.sh`: produces an `output/ft.gguf` from the `lora-out`.
* `make_ollama.sh`: make that `ft.gguf` file available to ollama under the name `llama32-ft`. So, at this point you can simply do `ollama run llama32-ft` to access the fine-tuned model.
* `bench-alpaca2.py`: bench Llama 3.3 on the next 500 elements, adding to `output.db`. Since we use this for validation, you should do this after running `fine_tune_lora.py` if you're trying to reproduce the project.
* `predict-alpaca2.py`: use Llama 3.2 and the fine-tuned version of Llama 3.2 to predict output token lengths of various prompts. Results are saved to `output/alpaca-pred2.db`.
* `plot-alpaca-predictions2.py`: plot those prediction accuracies; figure ? in the paper, plus some extra plots.

Finally, `bench-prompt.py` is a helper for the benchmarks.

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
