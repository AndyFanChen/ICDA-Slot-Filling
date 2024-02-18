import json
from langchain.llms import OpenAI
from langchain import PromptTemplate
import time
from tqdm import tqdm
import csv
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic_file",
        type=str,
        required=True,
        help="The name of the empty data dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The name of the empty data dataset to use (via the datasets library).",
    )

    args = parser.parse_args(args=[])

    return args

def save_list(list_to_save, file_name):
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list_to_save)


def read_list(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            list_from_file = row
    list_from_file = [int(num) for num in list_from_file]
    return list_from_file


def drop_data(data, drop_list):
    rest_data = []
    for data_idx, new_data in enumerate(data):
        if data_idx not in drop_list:
            rest_data.append(new_data)
    return rest_data

def save_data(data, name):
    with open(name, 'w', encoding='utf-8-sig') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=2)

def basic_drop(data):
    now_drop_list = []
    for d_idx, d in enumerate(data):
        text = d["userInput"]["text"].lower()
        if "<" in text:
            now_drop_list.append(d_idx)
        elif ">" in text:
            now_drop_list.append(d_idx)
        elif "slot" in text:
            now_drop_list.append(d_idx)
        elif "request" in text:
            now_drop_list.append(d_idx)
        elif len(d["labels"]) == 0:
            now_drop_list.append(d_idx)
    rest_data = drop_data(data, now_drop_list)
    return rest_data

def openAI_set():
    api_key = os.environ["OPENAI_API_KEY"]
    openai = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=api_key
    )
    return openai


def trans_ans(text):

    if "yes" in text.lower():
        return "yes"
    elif "no" in text.lower():
        return "no"


def word_and_slot_get(one_d, order=0):
    text = one_d['userInput']['text']
    try:
        word_start = one_d['labels'][order]["valueSpan"]["startIndex"]
    except KeyError:
        word_start = 0
    word_end = one_d['labels'][order]["valueSpan"]["endIndex"]
    word = text[word_start:word_end]
    slot = one_d['labels'][order]["slot"]

    return word, slot

def question_prompt_assign(one_d):
    slot_num = len(one_d["labels"])
    if slot_num == 1:
        q1 = "text: “{text}”\n\n" \
             "Is only tag “{word1}” as “{slot1}” slot"\
             " and not tag any other word as “{slot1}” slot in this text correct?"\
             " Just answer Yes or No."
        prompt_q1 = PromptTemplate(
            input_variables=["text", "word1", "slot1"],
            template=q1
        )
        text = one_d['userInput']['text']
        word1, slot1 = word_and_slot_get(one_d, 0)

        prompt_q1_word = prompt_q1.format(text=text, word1=word1, slot1=slot1)

    elif slot_num == 2:
        q1 = "text: “{text}”\n\n"\
             "Is only tag “{word1}” as “{slot1}” slot, only tag “{word2}” as “{slot2}” slot"\
             " and not tag any other word as “{slot1}” and “{slot2}” slot in this text correct?"\
             " Just answer Yes or No."
        prompt_q1 = PromptTemplate(
            input_variables=["text", "word1", "slot1", "word2", "slot2"],
            template=q1
        )
        text = one_d['userInput']['text']
        word1, slot1 = word_and_slot_get(one_d, 0)
        word2, slot2 = word_and_slot_get(one_d, 1)

        prompt_q1_word = prompt_q1.format(text=text, word1=word1, slot1=slot1, word2=word2, slot2=slot2)
    elif slot_num == 3:
        q1 = "text: “{text}”\n\n"\
             "Is only tag “{word1}” as “{slot1}” slot, only tag “{word2}” as “{slot2}” slot, only tag “{word3}” as “{slot3}” slot"\
             " and not tag any other word as “{slot1}”, {slot2} and “{slot3}” slot in this text correct?"\
             " Just answer Yes or No."
        prompt_q1 = PromptTemplate(
            input_variables=["text", "word1", "slot1", "word2", "slot2", "word3", "slot3"],
            template=q1
        )
        text = one_d['userInput']['text']
        word1, slot1 = word_and_slot_get(one_d, 0)
        word2, slot2 = word_and_slot_get(one_d, 1)
        word3, slot3 = word_and_slot_get(one_d, 2)

        prompt_q1_word = prompt_q1.format(text=text, word1=word1, slot1=slot1, word2=word2, slot2=slot2, word3=word3, slot3=slot3)
    elif slot_num == 4:
        q1 = "text: “{text}”\n\n"\
             "Is only tag “{word1}” as “{slot1}” slot, only tag “{word2}” as “{slot2}” slot, only tag “{word3}” as “{slot3}” slot"\
             ", only tag “{word4}” as “{slot4}” slot"\
             " and not tag any other word as “{slot1}”, “{slot2}”, “{slot3}” and “{slot4}” slot in this text correct?"\
             " Just answer Yes or No."
        prompt_q1 = PromptTemplate(
            input_variables=["text", "word1", "slot1", "word2", "slot2", "word3", "slot3", "word4", "slot4"],
            template=q1
        )
        text = one_d['userInput']['text']
        word1, slot1 = word_and_slot_get(one_d, 0)
        word2, slot2 = word_and_slot_get(one_d, 1)
        word3, slot3 = word_and_slot_get(one_d, 2)
        word4, slot4 = word_and_slot_get(one_d, 3)

        prompt_q1_word = prompt_q1.format(text=text, word1=word1, slot1=slot1, word2=word2, slot2=slot2, word3=word3, slot3=slot3
                                          , word4=word4, slot4=slot4)
    elif slot_num == 5:
        q1 = "text: “{text}”\n\n"\
             "Is only tag “{word1}” as “{slot1}” slot, only tag “{word2}” as “{slot2}” slot, only tag “{word3}” as “{slot3}” slot"\
             ", only tag “{word4}” as “{slot4}” slot, only tag “{word5}” as “{slot5}” slot"\
             " and not tag any other word as “{slot1}”, “{slot2}”, “{slot3}”, “{slot4}” and “{slot5}” slot in this text correct?"\
             "Just answer Yes or No."
        prompt_q1 = PromptTemplate(
            input_variables=["text", "word1", "slot1", "word2", "slot2", "word3", "slot3", "word4", "slot4", "word5", "slot5"],
            template=q1
        )

        text = one_d['userInput']['text']
        word1, slot1 = word_and_slot_get(one_d, 0)
        word2, slot2 = word_and_slot_get(one_d, 1)
        word3, slot3 = word_and_slot_get(one_d, 2)
        word4, slot4 = word_and_slot_get(one_d, 3)
        word5, slot5 = word_and_slot_get(one_d, 4)

        prompt_q1_word = prompt_q1.format(text=text, word1=word1, slot1=slot1, word2=word2, slot2=slot2, word3=word3, slot3=slot3
                                          , word4=word4, slot4=slot4, word5=word5, slot5=slot5)
    else:
        q1 = """text: “{text}”\n\n
                Is no label for this text correct?."""
        prompt_q1 = PromptTemplate(
            input_variables=["text"],
            template=q1
        )
        text = one_d['userInput']['text']
        prompt_q1_word = prompt_q1.format(text=text)
        print("error slot!")

    return prompt_q1_word

def chat_drop(data, openai):
    q1_check_name = r'./syn_data/syn_few7_req_1000_check_q1.json'
    q2_check_name = r'./syn_data/syn_few7_req_1000_check_q2.json'
    all_check_name = r'./syn_data/syn_few7_req_1000_check_all.json'
    q1_drop_list_name = r"./DropList/q1_list_req_1000_0602.csv"
    q2_drop_list_name = r"./DropList/q2_list_req_1000_0602.csv"
    all_drop_list_name = r"./DropList/all_list_req_1000_0602.csv"
    # if use check point:
    # q1_drop_list = read_list(q1_drop_list_name)
    # q2_drop_list = read_list(q2_drop_list_name)
    # all_drop_list = read_list(all_drop_list_name)
    # else
    q1_drop_list = []
    q2_drop_list = []
    all_drop_list = []
    q2 = "text: “{text}”\n\n"\
         "Is this text is normal and no weird marks in it?"\
         " Just answer Yes or No."
    prompt_q2 = PromptTemplate(
        input_variables=["text"],
        template=q2
    )
    data_bar = tqdm(enumerate(data), total=len(data), desc="Checking data")
    for d_idx, d in data_bar:
        # use for interrupt accident
        # if d_idx <= 1300:
        #     continue
        prompt_q1_word = question_prompt_assign(one_d=d)
        prompt_q2_word = prompt_q2.format(text=d['userInput']['text'])
        if d_idx % 100 == 0: # check first
            print(f"id {d_idx}")
            print(f"Q1 prompt:\n{prompt_q1_word}")
            print(f"Q2 prompt:\n{prompt_q2_word}")
        yes_no_text_1 = openai(prompt_q1_word)
        yes_no_text_1 = trans_ans(yes_no_text_1)
        if (yes_no_text_1 != "yes") and (yes_no_text_1 != "no"):
            print(f"Q1 error! output word: {yes_no_text_1}")
        yes_no_text_2 = openai(prompt_q2_word)
        yes_no_text_2 = trans_ans(yes_no_text_2)
        if (yes_no_text_2 != "yes") and (yes_no_text_2 != "no"):
            print(f"Q2 error! output word: {yes_no_text_2}")
        if not yes_no_text_1 == "yes":
            q1_drop_list.append(d_idx)
            print(f"q1 word: {d['userInput']['text']} id: {d_idx}")
        if not yes_no_text_2 == "yes":
            q2_drop_list.append(d_idx)
            print(f"q2 word: {d['userInput']['text']} id: {d_idx}")
        if not (yes_no_text_1 == "yes" and yes_no_text_2 == "yes"):
            all_drop_list.append(d_idx)
        if (d_idx > 0 and d_idx % 100 == 0) or d_idx == 2:
            q1_check_data = drop_data(data, q1_drop_list)
            q2_check_data = drop_data(data, q2_drop_list)
            all_check_data = drop_data(data, all_drop_list)
            save_data(q1_check_data, q1_check_name)
            save_list(q1_drop_list, q1_drop_list_name)
            print("q1 check!")
            print(f"q1 length {len(q1_check_data)}")
            print(f"q1_drop_list: {q1_drop_list}")
            save_data(q2_check_data, q2_check_name)
            save_list(q2_drop_list, q2_drop_list_name)
            print("q2 check!")
            print(f"q2 length {len(q2_check_data)}")
            print(f"q2_drop_list: {q2_drop_list}")
            save_data(all_check_data, all_check_name)
            save_list(all_drop_list, all_drop_list_name)
            print("all check!")
            print(f"all length {len(all_check_data)}")
            print(f"all_drop_list: {all_drop_list}")
            if d_idx != 2:
                time.sleep(100)

    q1_rest_data = drop_data(data, q1_drop_list)
    q2_rest_data = drop_data(data, q2_drop_list)
    all_rest_data = drop_data(data, all_drop_list)
    return q1_rest_data, q2_rest_data, all_rest_data

# main
def main():
    args = parse_args()
    with open(args.synthetic_file, 'r', encoding='utf-8-sig') as file:
        raw_data = json.load(file)
    openai = openAI_set()

    basic_drop_data = basic_drop(raw_data)
    q1_rest_data, q2_rest_data, all_rest_data = chat_drop(basic_drop_data, openai)

    print(f"origin len {len(raw_data)}")
    print(f"after basic drop len {len(basic_drop_data)}")


    basic_output_file = '{}_basic.json'.format(args.output_file)
    save_data(basic_drop_data, basic_output_file)
    print("basic save!")

    q1_output_file = '{}_q1.json'.format(args.output_file)
    save_data(q1_rest_data, q1_output_file)
    print("q1 save!")

    q2_output_file = '{}_q2.json'.format(args.output_file)
    save_data(q2_rest_data, q2_output_file)
    print("q2 save!")


    all_output_file = '{}_chat.json'.format(args.output_file)
    save_data(all_rest_data, all_output_file)
    print("chat save!")


if __name__ == "__main__":
    main()