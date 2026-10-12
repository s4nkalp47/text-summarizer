import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from textSummarizer.constants import TOKENIZER_NAME_FILE
from textSummarizer.entity import ModelTrainerConfig
from textSummarizer.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

    def check_data_matches_checkpoint(self) -> None:
        """Token IDs only mean something to the tokenizer that made them, so refuse mismatched data."""
        name_file = self.config.data_path / TOKENIZER_NAME_FILE
        tokenized_with = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else "unknown"
        if tokenized_with != self.config.model_ckpt:
            raise ValueError(
                f"Data in {self.config.data_path} was tokenized with '{tokenized_with}', but training uses "
                f"'{self.config.model_ckpt}'. Rerun the transformation stage in the same smoke mode."
            )

    def build_training_args(self) -> Seq2SeqTrainingArguments:
        fp16 = self.config.fp16
        if fp16 and not torch.cuda.is_available():
            logger.warning("fp16 requested but no CUDA GPU is available; training in fp32 instead")
            fp16 = False
        return Seq2SeqTrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            logging_first_step=True,
            fp16=fp16,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            eval_strategy="no",  # we evaluate once, after training
            save_strategy="no",  # we save the final model ourselves
            report_to="none",
        )

    def train(self) -> None:
        self.check_data_matches_checkpoint()
        dataset = load_from_disk(self.config.data_path)

        logger.info(f"Loading model and tokenizer: {self.config.model_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)

        # Pads each batch to its longest example; label padding becomes -100 so the loss ignores it.
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        training_args = self.build_training_args()
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
        )
        # T5 and Pegasus accept **kwargs in forward(), so Trainer assumes they normalize the loss over the whole
        # accumulated batch and skips dividing by gradient_accumulation_steps. They actually return a per-batch
        # mean, which would make the loss and gradients gradient_accumulation_steps times too large.
        trainer.loss_is_scaled_for_ga = False

        effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        logger.info(
            f"Training on {training_args.device} | {len(dataset['train'])} train rows | "
            f"effective batch size {effective_batch} | max_steps {training_args.max_steps}"
        )
        train_result = trainer.train()
        logger.info(
            f"Training finished: {train_result.global_step} steps, "
            f"average train loss {train_result.training_loss:.4f}"
        )

        eval_metrics = trainer.evaluate()
        logger.info(f"Validation loss: {eval_metrics['eval_loss']:.4f} ({len(dataset['validation'])} rows)")

        model.save_pretrained(self.config.model_path)
        tokenizer.save_pretrained(self.config.tokenizer_path)
        logger.info(f"Saved model to {self.config.model_path} and tokenizer to {self.config.tokenizer_path}")
