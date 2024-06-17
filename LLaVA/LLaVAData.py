import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoProcessor
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_value: torch.Tensor
    a_input_ids: torch.Tensor


class LLaVADataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        super().__init__()
        self.chat_data, self.image_dir = self.build_dataset(data_dir=data_dir)

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_dir = data_dir.joinpath("chat.json")
        img_dir = data_dir.joinpath("images_dl")
        chat_data = pd.read_json(chat_dir).to_dict(orient="records")

        return chat_data, img_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        human_input = cur_data["conversations"][0]["value"]
        gpt_output = cur_data["conversations"][1]["value"]
        img_dir = self.image_dir.joinpath(cur_data.get("image"))

        return human_input, gpt_output, img_dir


def builde2tensor(processor: AutoProcessor, q_text: str, a_text: str, image_path: Path):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(image_path)

    inputs = processor(prompt, image, return_tensors="pt")

    answer = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]

    return QaImageOutput(
        q_input_ids=inputs["input_ids"],
        pixel_value=inputs["pixel_values"],
        a_input_ids=answer,
    )


class TrainLLaVAModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int) -> None:
        self.processor = processor
        self.ignore_index = IGNORE_INDEX

    def convert_one_piece(
        self,
        q_input_idx: torch.Tensor,
        a_input_idx: torch.Tensor,
    ):
        input_ids = torch.concat(
            tensors=[
                q_input_idx,
                a_input_idx,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        labels = torch.concat(
            tensors=[
                torch.full_like(q_input_idx, fill_value=self.ignore_index),
                a_input_idx,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )

        return input_ids, labels

    def __call__(self, features: List) -> torch.Any:
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        max_input_len_list = []

        for feature in features:
            qaimg = builde2tensor(
                processor=self.processor,
                q_text=feature[0],
                a_text=feature[1],
                image_path=feature[2],
            )
            input_ids, labels = self.convert_one_piece(
                q_input_idx=qaimg.q_input_ids, a_input_idx=qaimg.a_input_ids
            )

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            pixel_values_list.append(qaimg.pixel_value)
            max_input_len_list.append(input_ids.shape[1])

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    tensors=[
                        torch.full(
                            size=(1, max_input_len - max_input_len_list[index]),
                            fill_value=self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )

        final_labels = torch.concat(
            [
                torch.concat(
                    tensors=[
                        torch.full(
                            size=(1, max_input_len - max_input_len_list[index]),
                            fill_value=self.ignore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )

        final_pixel_values = torch.concat(tensors=pixel_values_list, axis=0)

        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }
