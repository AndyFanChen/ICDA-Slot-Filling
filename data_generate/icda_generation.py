import json
import time
import copy
from tqdm import tqdm
import random
from langchain import PromptTemplate
from langchain.llms import OpenAI
from ICDA_config import *
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

config = Config()
OPENAI_API_KEY = "YOUR OPENAI_API_KEY"


def open_ai_set():

    if OPENAI_API_KEY != "":
        api_key = OPENAI_API_KEY
    else:
        api_key = os.environ["OPENAI_API_KEY"]
    openai = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=api_key
    )
    openai.temperature = config.temperature
    openai.frequency_penalty = config.rep_penalty

    return openai

def hugging_model(prompt, model_name): # if want to use LLM form huggingface instead of OpenAI

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    outputs_word = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return outputs_word

def file_set(if_shuffle=False): # Use restaurant8k data set to make prompt

    file = config.prompt_file
    print(f"prompt form file: {config.prompt_file}")
    with open(file, "r") as f:
        data = json.load(f)
    print(f"Total prompt examples num {len(data)}")
    if if_shuffle:
        random.shuffle(data)

    return data

def txt_insert_slot(text, labels):

    total_add_idx = 0
    for label in labels:
        slot = label["slot"]
        start_index = label["valueSpan"]["startIndex"] + total_add_idx
        end_index = label["valueSpan"]["endIndex"] + total_add_idx
        text = (text[:start_index] + f"<{slot}> " + text[start_index:end_index] +
                f" </{slot}>" + text[end_index:])
        total_add_idx += len(f"<{slot}> ") + len(f" </{slot}>")
    return text


def data_to_example(data): # extract data and transfer it to specific form

    examples = []
    for i, entry in enumerate(data):
        if entry.get("labels") is not None:
            # set label then insert label into text
            labels = entry["labels"]
            for label in labels:
                label['valueSpan'].setdefault("startIndex", 0)
            overlap = any(
                idx != 0 and
                label['valueSpan']["endIndex"] >= labels[0]['valueSpan']["startIndex"] and
                label['valueSpan']["startIndex"] <= labels[0]['valueSpan']["endIndex"]
                for idx, label in enumerate(labels)
            )
            if overlap:
                # print(f"overlap idx={i} skip data!")
                continue
            labels.sort(key=lambda x: x["valueSpan"]["startIndex"])
            text = txt_insert_slot(entry["userInput"]["text"], labels)
            examples.append(text)
    return examples


def create_example_prompt(all_examples, index, prompt_volume, suffix, prefix):

    example_start = index * prompt_volume
    example_end = (index + 1) * prompt_volume
    examples = all_examples[example_start:example_end]  # choose slice use this round
    examples = [f"Example{idx + 1}: " + example for idx, example in enumerate(examples)] # add Example{}:

    example_prompt = PromptTemplate.from_examples(
        examples=examples,
        suffix=suffix,
        input_variables=[],
        example_separator="\n\n",
        prefix=prefix
    )

    return example_prompt, examples


def process_item(item): # Classify text and return text, start_index, end_index
    # Regex pattern to find text between Example{idx}: and the next Example{idx}: or the end of the string
    itme = item.strip()
    pattern = r"(?:Example\d+:)?(.+?)(?=Example\d+:|$)"

    # Extract text portions into a list
    output_list = re.findall(pattern, itme, re.DOTALL)

    # Remove leading and trailing whitespace from each item in the list
    output_list = [text.strip() for text in output_list]

    return output_list
    # print(output_list)


def list_to_data_form(ai_text_list): # input the string llm generated out put asked data form

    output_data = []
    for line in ai_text_list:
        labels = []
        slots = re.finditer(r'<(\w+)>(.*?)</\1>', line)
        total_slot_len = 0
        for slotIdx, slot in enumerate(slots):
            slot_group = slot.group(1)
            total_slot_len += len(slot_group)
            slot_start = slot.start(2)
            slot_end = slot.end(2)
            label = {
                "slot": slot_group.lower(),
                "valueSpan": {
                    "startIndex": slot_start - total_slot_len - 2 - 7 * (slotIdx),
                    "endIndex": slot_end - total_slot_len - 4 - 7 * (slotIdx)
                }
            }
            labels.append(label)
            total_slot_len += len(slot.group(1))

        pattern_1 = r'<(\w+)>(.*?)</\1>'
        pattern_2 = r'<(\w+)>\s*(.*?)\s*</\1>'
        text = re.sub(pattern_2, r'\2', line)
        if len(labels) != 0:
            output_data.append({
                "userInput": {
                    "text": text.strip()
                },
                "labels": labels
            })
        else:
            output_data.append({
                "userInput": {
                    "text": text.strip()
                },
                "labels": labels
            })
    real_data_num = len(output_data)
    return output_data, real_data_num

def prefix_gene(config, slots_ask):

    if len(slots_ask) != 0:
        new_prefix = config.prefix_1 + str(slots_ask) + config.prefix_2
    else:
        new_prefix = config.default_prefix

    return new_prefix


def prompt_renew(config, this_slot, rest_slot, output_num_dict, examples, prefix, now_example_prompt):

    output_num_dict[f'{this_slot}_num'] += 1
    if output_num_dict[f'{this_slot}_num'] > config.slot_ask_num[config.slot_ask.index(this_slot)]:
        rest_slot.remove(this_slot)
        # print(f"slot_ask after remove: {slot_ask}")
        new_prefix = prefix_gene(config, rest_slot)
        example_prompt = PromptTemplate.from_examples(
            examples=examples,
            suffix=config.suffix,
            input_variables=[],
            example_separator="\n\n",
            prefix=new_prefix
        )
        print(f"{this_slot} limit arrived! new prompt: ")
        print(example_prompt.format())
        return example_prompt, output_num_dict, new_prefix

    return now_example_prompt, output_num_dict, prefix

def renew_num_prompt_filter_text(config, ai_text, rest_slot, output_num_dict, examples, prefix, example_prompt):

    filtered_ai_text = []

    for text in ai_text:
        slots = re.finditer(r'<(\w+)>(.*?)</\1>', text)
        slots_names = [slot.group(1).lower() for slot in slots]
        for slot_name in slots_names:
            if slot_name in config.slot_ask:
                example_prompt, output_num_dict, prefix = prompt_renew(config, slot_name, rest_slot, output_num_dict,
                                                                       examples, prefix, example_prompt)
        if all(slot_name in config.slot_ask for slot_name in slots_names):
            filtered_ai_text.append(text)

    return example_prompt, output_num_dict, prefix, filtered_ai_text


def process_save_check_ai_text(config, ai_text_list, real_output):

    all_slot_ai_text, real_data_num = list_to_data_form(ai_text_list)

    with open(config.check_point_file, 'w', encoding='utf-8-sig') as json_file:
        json.dump(all_slot_ai_text, json_file, ensure_ascii=False, indent=2)

    print(f"{real_output} check point!")

    if real_output != 2:
        time.sleep(60)

    return all_slot_ai_text, real_data_num


def generate_data(config, all_examples, round_num, rest_slot, prefix, openai=None):

    # count num of each slot generated
    output_num_dict = {'real_output_num': 0,
                       'people_num': 0,
                       'date_num': 0,
                       'time_num': 0,
                       'first_name_num': 0,
                       'last_name_num': 0}

    # before training setting
    real_output = 0  # record now accumulated data generated
    if_output_enough = False
    ai_text_list = []  # list put all generated data

    while True:
        print("new round!")
        totalBar = tqdm(range(round_num), total=round_num,
                        desc=f"LLM Generate Processing Target Output: {config.total_data_gene}")

        for i in totalBar:
            example_prompt, examples = create_example_prompt(all_examples, i, config.prompt_vol, config.suffix, prefix)

            if i % 100 == 0:
                print(f"idx:{i}\n{example_prompt.format()}")  # check prompt

            if config.hugging_face_llm is not None:
                ai_text = hugging_model(example_prompt.format(), "google/flan-t5-xxl")
            else:
                ai_text = openai(example_prompt.format()) # generate data

            ai_text = process_item(ai_text)
            real_output += len(ai_text)
            output_num_dict['real_output_num'] += len(ai_text)

            example_prompt, output_num_dict, prefix, ai_text = renew_num_prompt_filter_text(config, ai_text, rest_slot,
                                                                                           output_num_dict, examples,
                                                                                           prefix, example_prompt)

            totalBar.set_postfix(output_num_dict, refresh=True)
            ai_text_list += ai_text
            if real_output == config.total_data_gene:
                if_output_enough = True
                break

            if (real_output % 100 == 0 and real_output >= 100) or real_output == 2:
                process_save_check_ai_text(config, ai_text_list, real_output)

            if if_output_enough:
                break
        if if_output_enough:
            print(f"data generate over!")
            break
    return ai_text_list

def main():
    # print(os.environ.keys())
    openai = open_ai_set() if config.hugging_face_llm is None else None # setting llm model
    data = file_set() # prepare raw training data
    # data setting
    all_examples = data_to_example(data) # transfer raw training data to specific form
    if not config.use_all_data:
        all_examples = all_examples[:config.use_data_num]
    print(f"input data num: {len(all_examples)}")

    # training information
    round_num = len(all_examples) // config.prompt_vol + 1 # loops will run if using all example
    print(f"Examples using each round: {config.prompt_vol}")
    rest_slot = copy.deepcopy(config.slot_ask)
    prefix = config.default_prefix
    # generate
    ai_text_list = generate_data(config, all_examples, round_num, rest_slot, prefix, openai)

    # transfer ai_text_list to json data form
    all_slot_ai_text, real_data_num  = list_to_data_form(ai_text_list)
    print(f"final data generated: {real_data_num}")

    # save output
    with open(config.outputFile, 'w', encoding='utf-8-sig') as json_file: # generate json file for all slot mixed
        json.dump(all_slot_ai_text, json_file, ensure_ascii=False, indent=2)
    print("generated save end!")

if __name__ == "__main__":
    main()
