from transformers import TrainerCallback
import csv
import os

from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import jsonlines
import torch


# def get_num_devices():
#     if torch.cuda.is_available():
#         return torch.cuda.device_count()
#     else:
#         return 1  # Assumes at least one CPU is available if no GPUs are found


class MetricsLoggingCallback(TrainerCallback):
    def __init__(self, pvi_info="", drop_data_ratio=""):

        self.best_f1 = -1
        self.best_epoch = -1
        # self.best_eval_metrics = None
        # self.pvi_info = pvi_info
        # self.drop_data_ratio = drop_data_ratio

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):

        if metrics['eval_f1'] > self.best_f1:
            self.best_f1 = metrics['eval_f1']
            self.best_epoch = metrics['epoch']


    # def on_train_end(self, args, state, control, **kwargs):
    #     print(f"best_eval_metrics {self.best_eval_metrics}")
    #     if self.pvi_info != "":
    #         with open(self.pvi_info, 'r') as csvfile:
    #             reader = csv.DictReader(csvfile)
    #             row = next(reader)
    #             rest_vol = int(row['rest_vol'])
    #             pvi_tsd = float(row['pvi_tsd'])

    #         with jsonlines.open(os.path.join(args.output_dir, f'result_best_epoch_{self.drop_data_ratio} rest_vol_{rest_vol} pvi_tsd{round(pvi_tsd, 2)}_valid.json'), 'w') as writer:
    #             writer.write(self.best_eval_metrics)
    #         print("save success")

    
    # def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        
    #     # with jsonlines.open(os.path.join(args.output_dir, f'result_best_epoch_{args.drop_data_ratio}_test.json'), 'w') as writer:
    #     #     writer.write(metrics)
    #     return super().on_predict(args, state, control, metrics, **kwargs)