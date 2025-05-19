from transformers import AutoModelForCausalLM, AutoTokenizer


class AceMath:
    def __init__(self):
        device = "cuda"
        model_name = "nvidia/AceMath-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    def inference(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
                                                messages,
                                                tokenize=False,
                                                add_generation_prompt=True
                                            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


