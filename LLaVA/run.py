import torch
from dataclasses import dataclass, field
from LLaVAData import LLaVADataset, TrainLLaVAModelCollator
import transformers
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "the path to save data"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ckpt/model_001")
    train_type: str = field(
        default="none",
        metadata={
            "help": """
                使用lore训练: use_lora;
                全量训练: none;
                冻结vision_tower训练: freeze_vision;
        """
        },
    )


def load_dataset(dataargs: DataArguments):
    dataset = LLaVADataset(dataargs.data_path)
    return dataset


def load_model_processor(modelargs: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        tensor_type=torch.bfloat16,
        low_cpu_mem_use=True,
    )

    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)

    if modelargs.train_type == "use_lora":
        from peft import LoraConfig, get_peft_model

        Lora_r = 8
        Lora_alpha = 16
        Lora_dropout = 0.1
        target_moduel = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=Lora_r,
            lora_alpha=Lora_alpha,
            lora_dropout=Lora_dropout,
            target_modules=target_moduel,
        )
        model = get_peft_model(model, config)
    elif modelargs.train_type == "none":
        pass
    elif modelargs.train_type == "freeze_vision":
        for param in model.vision_tower.parameters():
            param.requirs_grad = False

    return model, processor


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    modelargs, dataargs, trainargs = parser.parse_args_into_dataclasses
    model, processor = load_model_processor(modelargs)
    dataset = load_dataset(dataargs)
    collator = TrainLLaVAModelCollator(processor, -100)

    trainer = Trainer(
        model=model,
        args=trainargs,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=trainargs.output_dir)


# python run.py --output_dir "output/" --per_device_train_batch_size 4
