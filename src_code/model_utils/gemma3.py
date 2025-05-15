from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

MODEL_ID = "google/gemma-3-4b-it"
CACHE_DIR = "../../../pip_cache/"

class Gemma3Model:
    def __init__(self):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto", cache_dir=CACHE_DIR
        ).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    def inference(self, prompt, img = None):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are path planner."}]
            },
            {
                "role": "user",
                "content": []
            }
        ]
        if img:
            messages[1]["content"].append({"type": "image", "image": img})
        
        messages[1]["content"].append({"type": "text", "text": f"{prompt}"})
        return self.generate(messages)

    def generate(self, messages):
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=True)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

