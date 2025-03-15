import torch
from transformers import ViTForImageClassification, TrainingArguments, ViTImageProcessor
import evaluate
from transformers import Trainer
import numpy as np


model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

class Vit:
    def __init__(self, train_dataset, valid_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=7
        )

        self.training_args = TrainingArguments(
            output_dir="./vit-base-beans",
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            num_train_epochs=4,
            fp16=True,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to='tensorboard',
            load_best_model_at_end=True,
            lr_scheduler_type="cosine", 
            warmup_ratio=0.1
        )

        def collate_fn(batch):
            return {
                'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
                'labels': torch.stack([x['labels'] for x in batch])
            }

        metric = evaluate.load("f1")

        def compute_metrics(p):
            return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average='micro')

        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=processor,
        )

    def train(self):

        train_results = self.trainer.train()
        self.trainer.save_model()
        self.trainer.log_metrics("train", train_results.metrics)
        self.trainer.save_metrics("train", train_results.metrics)
        self.trainer.save_state()

    def test(self):
        test_results = self.trainer.evaluate(self.test_dataset)
        print("Model f1 accuracy in the test dataset is:", test_results['eval_f1'])