prompts=(
"Calculate f(1) for f(x) = 3x^4 + 2x + 1/x - 5."
"Calculate the third integral of f(x) = 3x^4 + 2x + 1/x - 5."
)

ollama pull "llama3.3"

for prompt in "${prompts[@]}"; do
	python3 bench-prompt.py "llama3.3" "$prompt" output/example.db
done
