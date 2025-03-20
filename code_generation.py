import torch


class CodeGenStopCriteria(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated_text = ""
        self.previous_newline_index = -1
        self.remove_last_newline = False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        new_token = self.tokenizer.decode(input_ids[0][-1])
        if "<|end_of_text|>" in new_token:
            new_token = new_token.replace("<|end_of_text|>", "")
            self.generated_text += new_token
            return True

        self.generated_text += new_token
        if "\n" in new_token:
            if self.previous_newline_index != -1:
                current_newline_index = len(self.generated_text) - 1
                substring = self.generated_text[
                    self.previous_newline_index + 1 : current_newline_index
                ]

                if substring.startswith(" ") or substring.startswith("\t"):
                    self.previous_newline_index = current_newline_index
                    return False
                else:
                    # print(self.previous_newline_index, current_newline_index, substring)
                    self.remove_last_newline = True
                    self.generated_text = self.generated_text[
                        : self.previous_newline_index
                    ]
                    self.generated_text = self.generated_text.replace("\n\n", "\n")
                    return True
            else:
                self.previous_newline_index = len(self.generated_text) - 1
            return False

        return False


def generate_solution(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    stop_criteria = CodeGenStopCriteria(tokenizer)

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[stop_criteria],
        )

    # _ = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = stop_criteria.generated_text
    return new_tokens
