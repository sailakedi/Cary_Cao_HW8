import os
import io
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import arxiv
import fitz  # PyMuPDF
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from trl import RewardTrainer
import evaluate

# ---------------- CONFIG ---------------- #

# ArXiv config
ARXIV_QUERY = "cs.LG"            # change to anything you want
NUM_TRAIN_PAPERS = 10
NUM_EVAL_PAPERS = 10

# Paths
RAW_PDF_DIR = "pdfs"
TRAIN_META_JSON = "train_papers.json"
EVAL_META_JSON = "eval_papers.json"
SUMMARY_JSON = "summaries_train.json"
REWARD_JSONL = "reward_data.jsonl"

# Summarization / LM config
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # TODO: replace with your LLaMA 3 7B/8B path/checkpoint
LLAMA_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 256

# Reward model config
REWARD_MODEL_NAME = "microsoft/deberta-v3-base"
REWARD_OUT_DIR = "reward_model"
REWARD_NUM_EPOCHS = 3
REWARD_BATCH_SIZE = 4

# Evaluation config
RESULTS_JSON = "evaluation_results.json"

# ---------------- DATA STRUCTURES ---------------- #

@dataclass
class PaperData:
    arxiv_id: str
    title: str
    abstract: str
    text: str
    figure_captions: List[str]

# ---------------- ARXIV + PDF PARSING ---------------- #

def download_arxiv_pdfs(query: str, num_papers: int, out_dir: str) -> List[PaperData]:
    os.makedirs(out_dir, exist_ok=True)

    search = arxiv.Search(
        query=f"cat:{query}",
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: List[PaperData] = []
    for result in search.results():
        arxiv_id = result.get_short_id()
        title = result.title
        abstract = result.summary
        pdf_path = os.path.join(out_dir, f"{arxiv_id}.pdf")

        print(f"Downloading {arxiv_id}: {title[:80]}...")
        result.download_pdf(filename=pdf_path)

        text, figure_captions = extract_text_and_figures(pdf_path)
        papers.append(
            PaperData(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                text=text,
                figure_captions=figure_captions,
            )
        )

    return papers


def extract_text_and_figures(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Very simple PDF -> text + "figure captions" extractor using PyMuPDF.
    We treat any line starting with 'Figure' or 'Fig.' as a caption.
    This is *not* perfect, but good enough for a teaching pipeline.
    """
    doc = fitz.open(pdf_path)
    all_text_parts = []
    figure_captions = []

    for page in doc:
        t = page.get_text("text")
        all_text_parts.append(t)

        for line in t.splitlines():
            stripped = line.strip()
            if stripped.startswith("Figure ") or stripped.startswith("Fig. "):
                figure_captions.append(stripped)

    doc.close()
    full_text = "\n".join(all_text_parts)
    return full_text, figure_captions


def save_papers_to_json(papers: List[PaperData], path: str):
    data = []
    for p in papers:
        data.append(
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": p.abstract,
                "text": p.text,
                "figure_captions": p.figure_captions,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_papers_from_json(path: str) -> List[PaperData]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    papers = []
    for d in data:
        papers.append(
            PaperData(
                arxiv_id=d["arxiv_id"],
                title=d["title"],
                abstract=d["abstract"],
                text=d["text"],
                figure_captions=d["figure_captions"],
            )
        )
    return papers

# ---------------- SUMMARIZATION MODEL ---------------- #

def load_llama():
    print(f"Loading LLaMA model: {LLAMA_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)
    # Important for LLaMA chat-style models:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        torch_dtype=torch.float16 if LLAMA_DEVICE == "cuda" else torch.float32,
        device_map="auto" if LLAMA_DEVICE == "cuda" else None,
    )
    return model, tokenizer


def build_multimodal_text(paper: PaperData, max_chars: int = 6000) -> str:
    """
    Construct a text input that includes title, abstract, some body text,
    and figure captions as 'multimodal' cues.
    """
    figs = "\n".join(f"- {c}" for c in paper.figure_captions[:5])  # up to 5 captions
    body = paper.text
    if len(body) > max_chars:
        body = body[:max_chars]

    multimodal = (
        f"Title: {paper.title}\n\n"
        f"Abstract:\n{paper.abstract}\n\n"
        f"Selected figure captions:\n{figs}\n\n"
        f"Paper excerpt:\n{body}\n"
    )
    return multimodal


def llama_generate_summary(
    model,
    tokenizer,
    multimodal_text: str,
    prompt_style: str = "generic",
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """
    Generate a single summary from LLaMA using a chat-style prompt.
    Different prompt_style strings change the instruction.
    """
    if prompt_style == "generic":
        system_prompt = (
            "You are an expert research assistant. Summarize the following research paper "
            "clearly and concisely for a graduate student."
        )
    elif prompt_style == "figure_aware":
        system_prompt = (
            "You are an expert research assistant. Summarize the following research paper, "
            "explicitly incorporating the information conveyed by the figures and their captions."
        )
    else:
        system_prompt = (
            "You are a helpful academic assistant. Provide a detailed yet concise summary of the paper."
        )

    # Simple chat-format: you can adjust to the exact chat template of your LLaMA variant
    prompt = f"<|system|>{system_prompt}\n<|user|>Summarize the following research paper excerpt:\n\n{multimodal_text}\n<|assistant|>"

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(LLAMA_DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Extract only content after last <|assistant|>
    if "<|assistant|>" in decoded:
        summary = decoded.split("<|assistant|>")[-1].strip()
    else:
        summary = decoded.replace(prompt, "").strip()

    return summary

# ---------------- BUILD SUMMARY PAIRS + REWARD DATA ---------------- #

def generate_summaries_for_papers(papers: List[PaperData]) -> List[Dict[str, Any]]:
    """
    For each paper, produce two summaries with different prompts / sampling.
    We'll also keep the abstract as a 'reference' for ROUGE/BERTScore.
    """
    model, tokenizer = load_llama()
    results = []

    for idx, paper in enumerate(papers):
        print(f"\n=== Summarizing paper {idx+1}/{len(papers)}: {paper.arxiv_id} ===")
        multimodal_text = build_multimodal_text(paper)

        # Summary 1: generic prompt, low temperature
        s1 = llama_generate_summary(
            model,
            tokenizer,
            multimodal_text,
            prompt_style="generic",
            temperature=0.3,
            top_p=0.9,
        )

        # Summary 2: figure-aware prompt, higher temp
        s2 = llama_generate_summary(
            model,
            tokenizer,
            multimodal_text,
            prompt_style="figure_aware",
            temperature=0.8,
            top_p=0.95,
        )

        results.append(
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "summary_1": s1,
                "summary_2": s2,
            }
        )

        print("Summary 1 (generic):", s1[:200], "...")
        print("Summary 2 (figure-aware):", s2[:200], "...")

    # Save raw summaries
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def auto_label_preferences_with_rouge(summary_records: List[Dict[str, Any]]) -> None:
    """
    Use ROUGE-L against the abstract as a weak preference label:
    chosen = summary with higher ROUGE-L vs abstract, rejected = the other.
    Writes reward_data.jsonl with 'chosen' and 'rejected' fields.
    """
    rouge = evaluate.load("rouge")

    data_for_jsonl = []

    for rec in summary_records:
        ref = rec["abstract"]
        s1 = rec["summary_1"]
        s2 = rec["summary_2"]

        scores = rouge.compute(
            predictions=[s1, s2],
            references=[ref, ref],
        )
        # 'rougeL' is aggregated; we need per-summary, so recompute separately
        s1_score = rouge.compute(predictions=[s1], references=[ref])["rougeL"]
        s2_score = rouge.compute(predictions=[s2], references=[ref])["rougeL"]

        if s1_score >= s2_score:
            chosen, rejected = s1, s2
        else:
            chosen, rejected = s2, s1

        data_for_jsonl.append(
            {
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    with open(REWARD_JSONL, "w", encoding="utf-8") as f:
        for item in data_for_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote preference data to {REWARD_JSONL} ({len(data_for_jsonl)} pairs).")

# ---------------- REWARD MODEL TRAINING ---------------- #

def load_reward_dataset(path: str) -> Dataset:
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset


def train_reward_model():
    print("Loading reward model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_NAME,
        num_labels=1,
    )

    dataset = load_reward_dataset(REWARD_JSONL)

    def preprocess(examples):
        # Pairwise encoding: separate chosen and rejected
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }

        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tok_chosen = tokenizer(
                chosen,
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            tok_rejected = tokenizer(
                rejected,
                truncation=True,
                padding="max_length",
                max_length=512,
            )

            new_examples["input_ids_chosen"].append(tok_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tok_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tok_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tok_rejected["attention_mask"])

        return new_examples

    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=REWARD_OUT_DIR,
        per_device_train_batch_size=REWARD_BATCH_SIZE,
        num_train_epochs=REWARD_NUM_EPOCHS,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Training reward model...")
    trainer.train()
    trainer.save_model(REWARD_OUT_DIR)
    tokenizer.save_pretrained(REWARD_OUT_DIR)
    print("Reward model saved to", REWARD_OUT_DIR)

# ---------------- EVALUATION ---------------- #

def score_summaries_with_reward_model(
    summaries: List[str],
    reward_model,
    reward_tokenizer,
) -> List[float]:
    """
    Compute scalar reward scores for each summary.
    (Single-input scoring: we just feed the text and take the output logit.)
    """
    reward_model.eval()
    scores = []

    for s in summaries:
        inputs = reward_tokenizer(
            s,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        ).to(LLAMA_DEVICE if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            out = reward_model(**inputs)
            # out.logits shape: [batch, 1]
            score = out.logits.squeeze().item()
        scores.append(score)

    return scores


def evaluate_on_new_papers():
    # Load reward model
    print("Loading reward model for evaluation...")
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_OUT_DIR)
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_OUT_DIR)
    reward_model.to(LLAMA_DEVICE if torch.cuda.is_available() else "cpu")

    # 1) Collect new papers
    print(f"Downloading {NUM_EVAL_PAPERS} new evaluation papers from arXiv...")
    eval_papers = download_arxiv_pdfs(ARXIV_QUERY, NUM_EVAL_PAPERS, os.path.join(RAW_PDF_DIR, "eval"))
    save_papers_to_json(eval_papers, EVAL_META_JSON)

    # 2) Generate two summaries per paper
    print("Generating summaries for evaluation papers...")
    eval_summaries = generate_summaries_for_papers(eval_papers)

    # 3) Compute ROUGE & BERTScore against abstracts
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    results = []

    for rec in eval_summaries:
        ref = rec["abstract"]
        s1 = rec["summary_1"]
        s2 = rec["summary_2"]

        # ROUGE
        r1 = rouge.compute(predictions=[s1], references=[ref])
        r2 = rouge.compute(predictions=[s2], references=[ref])

        # BERTScore
        b1 = bertscore.compute(
            predictions=[s1],
            references=[ref],
            lang="en",
        )
        b2 = bertscore.compute(
            predictions=[s2],
            references=[ref],
            lang="en",
        )

        # Reward model scores
        scores = score_summaries_with_reward_model(
            [s1, s2],
            reward_model,
            reward_tokenizer,
        )
        rm1, rm2 = scores

        result_entry = {
            "arxiv_id": rec["arxiv_id"],
            "title": rec["title"],
            "abstract": ref,
            "summary_1": s1,
            "summary_2": s2,
            "rouge_1": r1,
            "rouge_2": r2,
            "bertscore_1": b1,
            "bertscore_2": b2,
            "reward_1": rm1,
            "reward_2": rm2,
        }
        results.append(result_entry)

        print("\n=== Paper", rec["arxiv_id"], "===")
        print("Reward scores: S1 =", rm1, " | S2 =", rm2)
        print("ROUGE-L: S1 =", r1["rougeL"], " | S2 =", r2["rougeL"])
        print("BERTScore F1: S1 =",
              b1["f1"][0],
              "| S2 =",
              b2["f1"][0])

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nSaved detailed evaluation results to", RESULTS_JSON)
    print("You can now inspect where reward scores agree/disagree with ROUGE/BERTScore.")

# ---------------- PHASE DRIVER ---------------- #

def build_data_phase():
    # 1. Collect 10 papers with PDFs, text + figure captions
    print(f"Downloading {NUM_TRAIN_PAPERS} training papers from arXiv...")
    train_papers = download_arxiv_pdfs(ARXIV_QUERY, NUM_TRAIN_PAPERS, os.path.join(RAW_PDF_DIR, "train"))
    save_papers_to_json(train_papers, TRAIN_META_JSON)

    # 2. Generate two summaries per paper using LLaMA
    print("Generating summaries for training papers...")
    summary_records = generate_summaries_for_papers(train_papers)

    # 3. Build reward modeling data with chosen/rejected
    print("Building reward modeling dataset via ROUGE-based preference labels...")
    auto_label_preferences_with_rouge(summary_records)


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal summarization + reward modeling pipeline (arXiv + LLaMA + DeBERTa)."
    )
    parser.add_argument(
        "--phase",
        choices=["build_data", "train_reward", "evaluate", "all"],
        default="all",
        help="Which phase to run.",
    )
    args = parser.parse_args()

    if args.phase in ("build_data", "all"):
        print("=== PHASE 1–3: Data collection, summarization, preference dataset ===")
        build_data_phase()

    if args.phase in ("train_reward", "all"):
        print("\n=== PHASE 4: Reward model training ===")
        train_reward_model()

    if args.phase in ("evaluate", "all"):
        print("\n=== PHASE 5–6: Evaluation and comparison ===")
        evaluate_on_new_papers()


if __name__ == "__main__":
    main()
