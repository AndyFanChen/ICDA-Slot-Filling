import argparse
import logging
import os

import datasets
import torch
from datasets import ClassLabel, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

import warnings
warnings.filterwarnings("ignore")
import csv

#%%


#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name_empty",
        type=str,
        required=True,
        # default="FanChen0116/19100_chat_50x_slot_limit_empty",
        help="The name of the empty data dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_name_syn",
        type=str,
        required=True,
        # default="FanChen0116/19100_chat_50x_slot_limit_empty",
        help="The name of the synthetic data dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="None",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
        # default="FanChen0116/few_64_train_SpanBERT"
    )
    parser.add_argument(
        "--model_name_or_path_empty",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
        # default="FanChen0116/empty_64_train_SpanBERT"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the final model.")
    parser.add_argument("--output_pvi", type=str, required=True, help="Where to store the training pvi result.")
    parser.add_argument("--output_valid_pvi", type=str, required=True, help="Where to store the valid set pvi result.")

    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()
    # args = []
    return args
# args = parse_args()
# print(args)

def main():
    torch.cuda.empty_cache()
    args = parse_args()

    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # set multiple data!
    raw_datasets_empty = load_dataset(args.dataset_name_empty, args.dataset_config_name)
    raw_datasets_syn = load_dataset(args.dataset_name_syn, args.dataset_config_name)

    column_names = raw_datasets_syn["train"].column_names
    features = raw_datasets_syn["train"].features

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[1]

    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f"{args.task_name}_tags" in column_names:
        label_column_name = f"{args.task_name}_tags"
    else:
        label_column_name = column_names[2]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets_syn["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)
    print(f"label_list {label_list}")

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config_empty = AutoConfig.from_pretrained(args.model_name_or_path_empty, num_labels=num_labels)

    tokenizer_name_or_path = args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    model_empty = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path_empty,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config_empty,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}
    model_empty.config.label2id = {l: i for i, l in enumerate(label_list)}
    model_empty.config.id2label = {i: l for i, l in enumerate(label_list)}

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False
    special_tokens = ['<people>', '<date>', '<time>', '<first_name>', '<last_name>', '<None>']
    special_tokens_dict = {
        'additional_special_tokens': ['<people>', '<date>', '<time>', '<first_name>', '<last_name>', '<None>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        modified_text = []
        for example_idx, example in enumerate(examples[text_column_name]):
            this_example = example + list(
                set(["<" + requested + ">" for requested in examples['request_slot'][example_idx]]))
            modified_text.append(this_example)  # modified_text = []

        tokenized_inputs = tokenizer(
            modified_text,
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            add_special_tokens=False
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            # try:
            this_text = modified_text[i]
            none_list = []
            for text_idx, text_word in enumerate(this_text):
                if text_word in special_tokens:
                    none_list.append(this_text.index(text_word))
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None or word_idx in none_list:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    inner1 = label[word_idx]
                    inner2 = label_to_id[inner1]
                    label_ids.append(inner2)
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
            # except IndexError:
            #     print(modified_text[i])
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():

        processed_raw_datasets_empty = raw_datasets_empty.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets_empty["train"].column_names,
            desc="Running tokenizer on dataset",
        )

        processed_raw_datasets_syn = raw_datasets_syn.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets_syn["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset_empty = processed_raw_datasets_empty["train"]
    train_dataset_syn = processed_raw_datasets_syn["train"]

    valid_dataset_empty = processed_raw_datasets_empty["validation"]
    valid_dataset_syn = processed_raw_datasets_syn["validation"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader_empty = DataLoader(
        train_dataset_empty, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    train_dataloader_syn = DataLoader(
        train_dataset_syn, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    valid_dataloader_empty = DataLoader(
        valid_dataset_empty, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    valid_dataloader_syn = DataLoader(
        valid_dataset_syn, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)
    model_empty.to(device)

    # Prepare everything with our `accelerator`.
    model, model_empty, train_dataloader_empty, train_dataloader_syn, valid_dataloader_empty, valid_dataloader_syn = accelerator.prepare(
        model, model_empty, train_dataloader_empty, train_dataloader_syn, valid_dataloader_empty, valid_dataloader_syn
    )

    print("first training")
    # logger.info("***** Running training *****")

    pvi_syn_list = []
    # model.train()
    train_bar_empty = tqdm(enumerate(train_dataloader_empty), total=len(train_dataloader_empty), desc="PVI Calculate")
    valid_bar_empty = tqdm(enumerate(valid_dataloader_empty), total=len(valid_dataloader_empty), desc="PVI Calculate")
    iterator_syn = iter(train_dataloader_syn)
    valid_iterator_syn = iter(valid_dataloader_syn)

    loss_fn = torch.nn.CrossEntropyLoss()
    for step, batch_empty in train_bar_empty:
        with torch.no_grad():
            # put empty data into empty model, get empty loss(check if it is cross_entropy)
            outputs_empty = model_empty(**batch_empty)
            loss_empty = outputs_empty.loss
            loss_empty = loss_empty.item()

            batch_syn = next(iterator_syn)
            outputs_syn = model(**batch_syn)
            loss_syn = outputs_syn.loss
            loss_syn = loss_syn.item()

            pvi_syn = (loss_empty - loss_syn)
            pvi_syn_list.append(pvi_syn)

    valid_pvi_syn_list = []
    for step, batch_empty in valid_bar_empty:
        with torch.no_grad():
            # put empty data into empty model, get empty loss(check if it is cross_entropy)
            outputs_empty = model_empty(**batch_empty)
            loss_empty = outputs_empty.loss

            loss_empty = loss_empty.item()
            batch_syn = next(valid_iterator_syn)
            outputs_syn = model(**batch_syn)

            loss_syn = outputs_syn.loss
            loss_syn = loss_syn.item()
            pvi_syn = (loss_empty - loss_syn)
            valid_pvi_syn_list.append(pvi_syn)

    with open(os.path.join(args.output_dir, args.output_pvi), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(pvi_syn_list)

    with open(os.path.join(args.output_dir, args.output_valid_pvi), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(valid_pvi_syn_list)


if __name__ == "__main__":
    main()