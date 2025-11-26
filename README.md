# qwen-gated-reasoning

Minimal experiment for testing whether a tiny gating policy can pick between
answering directly or using chain-of-thought (CoT) when solving GSM8K problems
with a frozen Qwen chat model.

## What it does
- Loads a Qwen chat model (default: `Qwen/Qwen1.5-1.8B-Chat`) without changing
  its weights.
- Runs two prompting strategies per question: direct answer and CoT.
- Trains a lightweight logistic regression gate on simple question features to
  select the best strategy.
- Reports accuracy and average token usage for always-direct, always-CoT, and
  the learned gate on a held-out subset.

## Setup
Install the required Python packages (a `requirements.txt` is included):

```bash
pip install -r requirements.txt
```

If your environment restricts outbound network access, pre-download wheels or
vendor the dependencies into your environment before running the script.

Make sure you can access the chosen Qwen model via Hugging Face (login may be
required).

## Running the experiment
Execute the gating experiment with a small subset for a quick sanity check:

```bash
python gating_qwen_gsm8k.py --n_train 40 --n_test 20
```

For a slightly more stable readout, use more samples:

```bash
python gating_qwen_gsm8k.py --n_train 80 --n_test 40
```

Key arguments:
- `--model_name`: Hugging Face model name (default: `Qwen/Qwen1.5-1.8B-Chat`).
- `--n_train`: Number of GSM8K examples to evaluate both prompts on for training
  the gate.
- `--n_test`: Number of held-out examples for evaluation.
- `--temperature`: Sampling temperature (set `0` for greedy decoding).
- `--max_new_tokens`: Maximum tokens the model can generate per call.

The script prints accuracy and average generated tokens for the two fixed
strategies and the learned gating policy so you can see whether per-question
prompt selection improves the accuracy–cost tradeoff.
