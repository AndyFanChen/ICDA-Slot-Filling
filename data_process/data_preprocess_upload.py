import json
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import argparse
import nltk

nltk.download('punkt')


def parse_args():
    parser = argparse.ArgumentParser(description="Process the PVI read and select.")
    parser.add_argument("--train_file", type=str, required=True, help="training json file")
    parser.add_argument("--valid_file", type=str, required=True, help="validation json file")
    parser.add_argument("--test_file", type=str, required=True, help="testing json file")
    parser.add_argument("--push_huggingface_data", type=str, required=True, help="PVI threshold value")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Split train valid ratio")
    parser.add_argument(
        "--empty",
        action='store_true',
        help="upload empty data or not"
    )

    args = parser.parse_args()
    return args


def slot_not_in_list(labels, slot_list):
    out_range_label = False
    for label in labels:
        if label["slot"] not in slot_list:
            out_range_label = True
    # any(label["slot"] not in slot_list for label in labels)

    return out_range_label


def nltk_offset_mapping(text, tokens):
    offsets = []
    start = 0

    for token in tokens:
        start = text.find(token, start)
        end = start + len(token)
        offsets.append((start, end))
        start = end

    return offsets


def set_start_end(label, offset_mapping):
    try:
        start = label["valueSpan"]["startIndex"]
    except KeyError:
        start = 0
    end = label["valueSpan"]["endIndex"]

    # Find the start token
    start_token = None
    for m, offset in enumerate(offset_mapping):
        if start >= offset[0] and start < offset[1]:
            start_token = m
            break

    # Find the end token
    end_token = None
    for m, offset in enumerate(offset_mapping):
        if end > offset[0] and end <= offset[1]:
            end_token = m
            break

    return start_token, end_token


def ner_tags_set(data, slot_list, dataset_dict, tag_names):
    # Update the ner_tags based on the valueSpan information
    for i, d in enumerate(data):
        text = d["userInput"]["text"]
        tokens = word_tokenize(text)
        tags = ["O"] * len(tokens)
        # set tags to correspond slot
        if "labels" in d:
            labels = d["labels"]
            if slot_not_in_list(labels, slot_list):
                continue
            # Get the offset mappings
            offset_mapping = nltk_offset_mapping(text, tokens)
            for label in labels:
                slot = label["slot"]
                start_token, end_token = set_start_end(label, offset_mapping)
                # Update the tags based on the start and end tokens
                if start_token is not None and end_token is not None:
                    tags[start_token] = f"B-{slot}"
                    for m in range(start_token + 1, end_token + 1):
                        tags[m] = f"I-{slot}"
        # set ner_tags as tags
        for j in range(len(tags)):
            dataset_dict["ner_tags"][i][j] = tag_names.index(tags[j])
    return dataset_dict


def train_val_dataset(dataset_dict, split_ratio):
    id_train, id_valid, token_train, token_valid, tag_train, tag_valid, req_slot_train, req_slot_valid \
        = train_test_split(dataset_dict["id"], dataset_dict["tokens"], dataset_dict["ner_tags"],
                           dataset_dict["request_slot"], test_size=split_ratio, random_state=42)
    train_dataset_dict = {"id": id_train, "tokens": token_train,
                          "ner_tags": tag_train, 'request_slot': req_slot_train}
    valid_dataset_dict = {"id": id_valid, "tokens": token_valid,
                          "ner_tags": tag_valid, 'request_slot': req_slot_valid}

    return train_dataset_dict, valid_dataset_dict


def dataset_dict_to_dataset(dataset_dict, tag_names):
    dataset = Dataset.from_dict(dataset_dict)
    # Add tag names to the dataset
    class_label = ClassLabel(names=tag_names)
    dataset = dataset.rename_column("ner_tags", "labels")
    new_features = Features({
        'id': dataset.features['id'],
        'tokens': dataset.features['tokens'],
        "labels": Sequence(class_label),
        'request_slot': dataset.features['request_slot']
    })
    dataset = dataset.cast(new_features)

    return dataset


def datasetCreat(file, val_ratio, empty=False, request=True):
    with open(file, "r", encoding='utf-8-sig') as f:
        data = json.load(f)

    dataset_dict = {
        "id": list(range(len(data))),
        "tokens": [word_tokenize(d["userInput"]["text"]) for d in data],
        "ner_tags": [[0] * len(word_tokenize(d["userInput"]["text"])) for d in data],
        "request_slot": [
            [] if "context" not in d.keys() or "requestedSlots" not in d["context"].keys() else d["context"][
                "requestedSlots"] for d in data]
    }
    tag_names = ['O', 'I-time', 'B-date', 'B-last_name', 'B-people', 'I-date', 'I-people',
                 'I-last_name', 'I-first_name', 'B-first_name', 'B-time']
    slot_list = ['time', 'people', 'date', 'first_name', 'last_name']
    dataset_dict = ner_tags_set(data, slot_list, dataset_dict, tag_names)

    if empty:
        for sen_idx, sentence in enumerate(dataset_dict['tokens']):
            dataset_dict['tokens'][sen_idx] = ['z' for token in sentence]
            dataset_dict["request_slot"][sen_idx] = ['None']
    # Create the Hugging Face dataset
    if val_ratio != 0.0:
        train_dataset_dict, valid_dataset_dict = train_val_dataset(dataset_dict, val_ratio)
        train_dataset = dataset_dict_to_dataset(train_dataset_dict, tag_names)
        valid_dataset = dataset_dict_to_dataset(valid_dataset_dict, tag_names)

        return train_dataset, valid_dataset
    else:
        dataset = dataset_dict_to_dataset(dataset_dict, tag_names)

        return dataset


def main():
    args = parse_args()
    print(f"data_preprocess args:{args}")

    if args.val_ratio != 0.0:
        trainData, validData = datasetCreat(args.train_file, val_ratio=args.val_ratio, empty=args.empty,
                                            request=not args.empty)
        print("valid split by train not valid file")
    else:
        trainData = datasetCreat(args.train_file, val_ratio=args.val_ratio, empty=args.empty, request=not args.empty)
        validData = datasetCreat(args.valid_file, val_ratio=args.val_ratio, empty=args.empty, request=not args.empty)
    print(f"trainData feature {trainData.features}")
    # test
    testData = datasetCreat(args.test_file, val_ratio=args.val_ratio, empty=False)
    dataset = DatasetDict({"train": trainData, "validation": validData, "test": testData})
    dataset.push_to_hub(args.push_huggingface_data)
    # print("push success!")


if __name__ == "__main__":
    main()