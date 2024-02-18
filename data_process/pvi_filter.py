import argparse
import json
import pandas as pd
import numpy as np
import random
import csv
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Process the PVI read and select.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the JSON file")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file")

    # parser.add_argument("--drop_data_vol", type=int, required=True, help="drop data volume")
    parser.add_argument("--drop_data_ratio", type=float, required=True, help="drop data ratio")
    parser.add_argument("--output_dir", type=str, required=True, help="drop data ratio")
    parser.add_argument("--new_file_name", type=str, required=True, help="json file saved")
    parser.add_argument(
        "--pvi_high",
        action='store_true',
        help="Drop high or low PVI"
    )
    parser.add_argument(
        "--if_drop_vol",
        action='store_true',
        help="drop data volume or ratio"
    )

    # parser = argparse.ArgumentParser(description="Process the PVI read and select.")
    # parser.add_argument("--json_file", type=str, default=r"C:\Users\Andy Chen\PycharmProjects\HuggingPVITest\src\few_PVI\syn_few7_7100_chat.json", help="Path to the JSON file")
    # parser.add_argument("--csv_file", type=str, default=r"C:\Users\Andy Chen\PycharmProjects\HuggingPVITest\src\few_PVI\syn_data_pvi_7100.csv", help="Path to the CSV file")
    # parser.add_argument("--if_drop_vol", type=bool, default=False, help="drop data volume or ratio")
    # parser.add_argument("--drop_data_vol", type=int, default=0, help="drop data volume")
    # parser.add_argument("--drop_data_ratio", type=float, default=0.1, help="drop data ratio")
    # parser.add_argument("--new_file_name", type=str, default="new_test.json", help="Path to the CSV file")
    # parser.add_argument("--pvi_high", type=bool, default=False, help="Drop high or low PVI")

    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    print(f"pvi filter args:{args}")
    def pvi_read_select(file, pvi_file, new_file_name, pvi_threshold=0, pvi_up=True):
        with open(file, "r", encoding='utf-8-sig') as f:
            data = json.load(f)
        pvi_scores = pd.read_csv(pvi_file, header=None)
        # Convert the DataFrame to a NumPy array
        pvi_scores = np.array(pvi_scores)
        pvi_scores = pvi_scores.reshape(-1)

        drop_ratio = args.drop_data_ratio
        drop_vol = int(args.drop_data_ratio * len(pvi_scores))

        # Get the indices that would sort the pvi_scores array
        sorted_indices = np.argsort(pvi_scores)

        # Remove the indices of the 30 lowest scores
        # pvi_high drop high pvi
        if args.pvi_high:
            indices_to_keep = sorted_indices[:len(pvi_scores) - drop_vol]
            pvi_threshold = pvi_scores[indices_to_keep[-1]]
        else:
            indices_to_keep = sorted_indices[drop_vol:]
            pvi_threshold = pvi_scores[indices_to_keep[0]]

        # Index into data with these indices
        filtered_data = [data[i] for i in indices_to_keep]

        ori_vol = len(pvi_scores)
        rest_vol = len(pvi_scores) - drop_vol
        drop_ratio = drop_ratio
        pvi_tsd = pvi_threshold

        if args.pvi_high:
            csv_name = os.path.join(args.output_dir, f'ori_vol_{ori_vol} drop_ratio_{drop_ratio} High pvi_drop_file.csv')
        else:
            csv_name = os.path.join(args.output_dir, f'ori_vol_{ori_vol} drop_ratio_{drop_ratio} Low pvi_drop_file.csv')

        with open(csv_name, 'w', newline='') as csvfile:
            fieldnames = ['ori_vol', 'rest_vol', 'drop_ratio', 'pvi_tsd']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({'ori_vol': ori_vol, 'rest_vol': rest_vol, 'drop_ratio': drop_ratio, 'pvi_tsd': pvi_tsd})

        if args.pvi_high:
            json_name = os.path.join(args.output_dir, f"drop_ratio_{drop_ratio} High {new_file_name}")
        else:
            json_name = os.path.join(args.output_dir, f"drop_ratio_{drop_ratio} Low {new_file_name}")

        with open(json_name, 'w',
                  encoding='utf-8-sig') as json_file:
            json.dump(filtered_data, json_file, ensure_ascii=False, indent=2)
        print(f"ori_vol:{len(pvi_scores)} rest_vol:{len(pvi_scores) - drop_vol} drop_ratio:{drop_ratio}")
        print(f"new data saved! file name: {json_name}")
        return filtered_data

        # # look if difficulty related to pvi
        # # pvi_scores_bar = enumerate(pvi_scores)
        # idx_to_drop = []
        # for pvi_idx, pvi in enumerate(pvi_scores):
        #     if pvi_up == "Up":
        #         if pvi >= pvi_threshold:
        #             idx_to_drop.append(pvi_idx)
        #     else:
        #         if pvi <= pvi_threshold:
        #             idx_to_drop.append(pvi_idx)

        # new_data = [item for i, item in enumerate(data) if i not in idx_to_drop]
        # print(f"drop vol: {len(idx_to_drop)} new data vol: {len(new_data)}")
        # with open(new_file_name, 'w', encoding='utf-8-sig') as json_file:  # generate json file for all slot mixed
        #     json.dump(new_data, json_file, ensure_ascii=False, indent=2)
        # print("new data saved!")
        # return new_data

    random.seed(42)
    new_data_pvi = pvi_read_select(args.json_file, args.csv_file, args.new_file_name, pvi_up=args.pvi_high)

if __name__ == "__main__":
    main()