"""
VLA-PEFT Inference Server
FastAPI server that exposes the fine-tuned LLaMA model for GeoGuesser explanations.
"""

import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration - can be overridden via environment variables
BASE_MODEL_NAME = os.environ.get("VLA_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LORA_CHECKPOINT_DIR = os.environ.get("VLA_LORA_CHECKPOINT", "./checkpoints/checkpoint-1494")

app = FastAPI(title="VLA-PEFT GeoGuesser Server", version="1.0.0")
tokenizer = None
model = None

class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

class ChatReq(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

@app.on_event("startup")
def _load():
    global tokenizer, model
    print("🚀 Loading VLA-PEFT model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # Try to load LoRA checkpoint if available
    try:
        if LORA_CHECKPOINT_DIR and os.path.exists(LORA_CHECKPOINT_DIR):
            print(f"  📦 Attempting to load LoRA from {LORA_CHECKPOINT_DIR}...")
            base = PeftModel.from_pretrained(base, LORA_CHECKPOINT_DIR)
            print("  ✅ LoRA loaded successfully")
    except Exception as e:
        print(f"  ⚠️  WARNING: Could not load LoRA checkpoint: {e}")
        print("  ℹ️  Continuing with base model only")

    model = base.eval()
    device = next(model.parameters()).device
    print(f"✅ Model loaded and ready on {device}!")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "ok": True,
        "device": str(next(model.parameters()).device) if model else "not loaded",
        "model": BASE_MODEL_NAME,
        "lora_checkpoint": LORA_CHECKPOINT_DIR if os.path.exists(LORA_CHECKPOINT_DIR or "") else None,
    }

@app.post("/chat")
@torch.inference_mode()
def chat(req: ChatReq):
    """
    Generate a response given a chat history.
    
    Args:
        req: ChatReq with messages list and generation parameters
        
    Returns:
        dict with "assistant" key containing the generated text
    """
    # Convert Pydantic objects into plain dicts
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    input_ids = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.temperature > 0,
        temperature=req.temperature,
        top_p=req.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Important: decode ONLY the newly generated tokens
    new_tokens = out[0, input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {"assistant": text}
