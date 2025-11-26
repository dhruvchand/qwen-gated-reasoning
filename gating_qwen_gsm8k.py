"""
Minimal experiment for policy-over-LM gating with Qwen on GSM8K.

This script:
- Loads a frozen Qwen chat model as the language engine.
- Defines two prompting strategies: direct answer and chain-of-thought (CoT).
- Evaluates both strategies for each question in a small GSM8K subset.
- Trains a lightweight logistic regression gate on simple question features to
  choose between direct and CoT per example.
- Reports accuracy and average token usage for always-direct, always-CoT,
  and the learned gate on a held-out test subset.
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLM:
    """Frozen Qwen chat model wrapper for simple generation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen1.5-1.8B-Chat",
        device: str | None = None,
        dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"[LM] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        ).to(self.device)
        self.model.eval()
        print("[LM] Loaded.")

    @torch.no_grad()
    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int]:
        """Generate text and return the output plus number of new tokens."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = output_ids[0][input_len:]
        num_tokens = gen_ids.shape[0]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text, int(num_tokens)


SYSTEM_PROMPT = (
    "You are a helpful math tutor. When asked to show reasoning, think step by "
    "step. When giving a final answer, use the format 'Answer: <number>'."
)


MULTI_STEP_WORDS = [
    "each",
    "altogether",
    "after",
    "then",
    "total",
    "together",
    "in all",
    "per",
    "every",
]


def build_prompt_direct(question: str) -> str:
    return (
        "Solve the following math problem and give ONLY the final numeric answer.\n\n"
        f"Problem:\n{question}\n\n"
        "Answer:"
    )


def build_prompt_cot(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Show your reasoning, then on the last line write:\n"
        "Answer: <number>\n\n"
        f"Problem:\n{question}\n"
    )


def parse_numeric_from_answer(answer_str: str) -> str:
    """Extract a numeric prediction from the model output."""

    matches = list(re.finditer(r"Answer\s*:\s*(.+)", answer_str, re.IGNORECASE))
    candidate = None
    if matches:
        candidate = matches[-1].group(1)
    else:
        nums = re.findall(r"[-+]?\d*\.?\d+", answer_str)
        candidate = nums[-1] if nums else ""

    candidate = candidate.strip()
    candidate = re.sub(r"[^\d\.\-]+$", "", candidate)
    return candidate


def parse_numeric_from_gsm8k_answer(answer_str: str) -> str:
    nums = re.findall(r"[-+]?\d*\.?\d+", answer_str)
    if nums:
        return nums[-1]
    return answer_str.strip()


def extract_features(question: str) -> np.ndarray:
    q_lower = question.lower()
    word_count = len(question.split())
    num_numbers = len(re.findall(r"\d+", question))
    has_multi = int(any(w in q_lower for w in MULTI_STEP_WORDS))
    return np.array([word_count, num_numbers, has_multi], dtype=np.float32)


@dataclass
class ExampleResult:
    question: str
    gold_answer_raw: str
    gold_num: str
    direct_pred: str
    direct_correct: bool
    direct_tokens: int
    cot_pred: str
    cot_correct: bool
    cot_tokens: int
    features: np.ndarray


def evaluate_example_with_both(lm: QwenLM, question: str, gold_answer_raw: str) -> ExampleResult:
    gold_num = parse_numeric_from_gsm8k_answer(gold_answer_raw)

    direct_prompt = build_prompt_direct(question)
    direct_out, direct_tokens = lm.generate(SYSTEM_PROMPT, direct_prompt)
    direct_pred = parse_numeric_from_answer(direct_out)
    direct_correct = direct_pred == gold_num

    cot_prompt = build_prompt_cot(question)
    cot_out, cot_tokens = lm.generate(SYSTEM_PROMPT, cot_prompt)
    cot_pred = parse_numeric_from_answer(cot_out)
    cot_correct = cot_pred == gold_num

    features = extract_features(question)

    return ExampleResult(
        question=question,
        gold_answer_raw=gold_answer_raw,
        gold_num=gold_num,
        direct_pred=direct_pred,
        direct_correct=direct_correct,
        direct_tokens=direct_tokens,
        cot_pred=cot_pred,
        cot_correct=cot_correct,
        cot_tokens=cot_tokens,
        features=features,
    )


def run_experiment(
    model_name: str,
    n_train: int,
    n_test: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    lm = QwenLM(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print("[Data] Loading GSM8K (train split)...")
    ds = load_dataset("gsm8k", "main")["train"]
    print(f"[Data] Total GSM8K train examples: {len(ds)}")

    indices = list(range(len(ds)))
    random.shuffle(indices)
    subset_indices = indices[: n_train + n_test]
    subset = ds.select(subset_indices)

    train_subset = subset.select(range(n_train))
    test_subset = subset.select(range(n_train, n_train + n_test))

    print(f"[Data] Using {len(train_subset)} for gating train, {len(test_subset)} for test.")

    train_results: List[ExampleResult] = []
    print("\n[Train] Evaluating LM on train subset...")
    for i, ex in enumerate(train_subset):
        print(f"[Train] Example {i + 1}/{len(train_subset)}")
        train_results.append(
            evaluate_example_with_both(
                lm,
                question=ex["question"],
                gold_answer_raw=ex["answer"],
            )
        )

    X_train: List[np.ndarray] = []
    y_train: List[int] = []

    for res in train_results:
        if res.direct_correct and not res.cot_correct:
            X_train.append(res.features)
            y_train.append(0)
        elif res.cot_correct and not res.direct_correct:
            X_train.append(res.features)
            y_train.append(1)
        else:
            continue

    X_train_arr = np.array(X_train, dtype=np.float32)
    y_train_arr = np.array(y_train, dtype=np.int64)

    print(f"\n[Gate] Training set size for gating: {len(y_train_arr)} examples.")
    if len(y_train_arr) == 0:
        print(
            "[Gate] No informative training examples (direct vs CoT never differ). "
            "Increase n_train to gather more data."
        )
        return

    gate = LogisticRegression()
    gate.fit(X_train_arr, y_train_arr)
    gate_train_acc = accuracy_score(y_train_arr, gate.predict(X_train_arr))
    print(f"[Gate] Training accuracy on informative cases: {gate_train_acc:.3f}")

    print("\n[Test] Evaluating on test subset...")
    test_results: List[ExampleResult] = []
    for i, ex in enumerate(test_subset):
        print(f"[Test] Example {i + 1}/{len(test_subset)}")
        test_results.append(
            evaluate_example_with_both(
                lm,
                question=ex["question"],
                gold_answer_raw=ex["answer"],
            )
        )

    metrics = compute_metrics(test_results, gate)

    print("\n=== Results on test subset ===")
    print(
        f"Always direct: acc={metrics['direct_acc']:.3f}, "
        f"avg_tokens={metrics['direct_tokens']:.1f}"
    )
    print(
        f"Always CoT  : acc={metrics['cot_acc']:.3f}, "
        f"avg_tokens={metrics['cot_tokens']:.1f}"
    )
    print(
        f"Gating policy: acc={metrics['gate_acc']:.3f}, "
        f"avg_tokens={metrics['gate_tokens']:.1f}"
    )
    frac_cot = np.mean(np.array(metrics["gate_choices"]) == 1)
    print(f"Gate chose CoT on {frac_cot * 100:.1f}% of test questions.")


def compute_metrics(results: List[ExampleResult], gate: LogisticRegression) -> dict:
    direct_corrects = [int(r.direct_correct) for r in results]
    direct_tokens = [r.direct_tokens for r in results]

    cot_corrects = [int(r.cot_correct) for r in results]
    cot_tokens = [r.cot_tokens for r in results]

    gate_corrects: List[int] = []
    gate_tokens: List[int] = []
    gate_choices: List[int] = []

    for res in results:
        choice = int(gate.predict(res.features.reshape(1, -1))[0])
        gate_choices.append(choice)
        if choice == 0:
            gate_corrects.append(int(res.direct_correct))
            gate_tokens.append(res.direct_tokens)
        else:
            gate_corrects.append(int(res.cot_correct))
            gate_tokens.append(res.cot_tokens)

    return {
        "direct_acc": float(np.mean(direct_corrects)),
        "direct_tokens": float(np.mean(direct_tokens)),
        "cot_acc": float(np.mean(cot_corrects)),
        "cot_tokens": float(np.mean(cot_tokens)),
        "gate_acc": float(np.mean(gate_corrects)),
        "gate_tokens": float(np.mean(gate_tokens)),
        "gate_choices": gate_choices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gating experiment for Qwen on GSM8K")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen1.5-1.8B-Chat",
        help="Hugging Face model name for Qwen.",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=80,
        help="Number of GSM8K train examples for gating train subset.",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=40,
        help="Number of GSM8K train examples for test subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens for LM generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0 for greedy).",
    )
    args = parser.parse_args()

    run_experiment(
        model_name=args.model_name,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
