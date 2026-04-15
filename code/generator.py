"""
generator.py — flan-t5-small LLM Generator (CPU-only)

Builds a prompt from retrieved chunks and generates an answer.
No GPU, no external API required.
"""

import os
import sys
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class Generator:
    """
    Wraps google/flan-t5-small for CPU inference.
    flan-t5 is instruction-tuned → understands "Answer based on context" prompts well.
    """

    def __init__(self):
        print(f"  Loading generator: {config.GENERATOR_MODEL} …")
        self.tokenizer = T5Tokenizer.from_pretrained(config.GENERATOR_MODEL)
        self.model     = T5ForConditionalGeneration.from_pretrained(config.GENERATOR_MODEL)
        self.model.eval()
        print("  ✓ Generator ready (CPU)\n")

    def generate(self, query: str, chunks: list) -> str:
        context = "\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in chunks
        )
        prompt = config.RAG_PROMPT_TEMPLATE.format(
            context=context, question=query
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            num_beams=config.NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
