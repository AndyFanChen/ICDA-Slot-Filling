# In-Context Data Augmentation For Slot Filling
* Applied in-context learning approach with ChatGPT to generate synthetic data which can augment scarce datasets and improved model performance by 65%
* Establish PVI score that represents the meaning of the synthetic data to the model, selecting synthetic data with the optimal PVI allocation and fine-tune SpanBERT model with selected datasets
## Result Table
* Use synthetic data to impove f1 score from 40.0 to 57.4  
* Use LLM filter to impove f1 score from 57.4 to 61.4  
* Use PVI filter to impove f1 score from 61.4 to 65.9  

<img src="https://github.com/AndyFanChen/ICDA-Slot-Filling/blob/main/ICDA_result_table.png">

For more details, please refer to ICDA_Slot_Filling_Poster.pdf which is the poster for this project.
## Execute step
* First step: Generate synthetic data.
* Second step: Filter out synthetic data by LLM and PVI filter.
* Third step: Mix synthetic data and original data to train.
## Generate Data
### Generate Synthetic Data
`python3 icda_generation.py`

You have to set your openai key in environment and set PATH as "OPENAI_API_KEY"
Can set all parameters in ICDA_config.py

## Processing Data
### Filter out by LLM filter
* `synthetic_file`: original synthetic data
* `output_file`: output data name
```
`python3 llm_filter.py --synthetic_file \
--output_file`
```
### Calculate pvi value
* `dataset_name_empty`: empty feature dataset(huggingface dataset)
* `dataset_name_syn`: synthetic dataset (huggingface dataset)
* `output_dir`: the path where the output files save 
* `output_pvi`: output pvi value file name (csv)
* `output_valid_pvi`: output pvi value file name (csv)
* `model_name_or_path`: model trained by original data (huggingface model)
* `model_name_or_path_empty`: model trained by empty data (correspond to original data) (huggingface model)
```
python3 pvi_calculate.py --dataset_name_empty \
--dataset_name_syn \
--output_dir \
--output_pvi \
--output_valid_pvi \
--model_name_or_path \
--model_name_or_path_empty
```
### Filter out by PVI filter
* `json_file`: synthetic data need to filter out
* `csv_file`: pvi score file correspond to synthetic data
* `new_file_name`: file output after filter out
* `drop_data_ratio`: how much proportion of data want to filter out this time
* `pvi_high`: filter out by High or Low pvi (If want to filter out high use this)

```
`python3 pvi_filter.py --json_file \
--csv_file \
--new_file_name \
--drop_data_ratio \
--output_dir \
--pvi_high \
```
### Mix data(preprocessed data and original data)
* `data1`: first file want to be merged
* `data2`: second file want to be merged
* `output`: path of merged data
```
python3 merge_data.py --data1 \
--data2 \
--output
```
* `train_file`: training set of the data
* `valid_file`: validation set of the data
* `test_file`: testing set of the data
* `push_huggingface_data`: name for output dataset on hugging face hub
```
python3 data_preprocess_upload.py  --train_file \
--valid_file \
--test_file \
--push_huggingface_data
```

## Train and Evaluate
Example:   

```
python3 slot_filling_train_trainer.py \
 --dataset_name "FanChen0116/19100_chat_8x_slot_pvi"\
 --output_dir "/ssd/andychen/dataset/trainer/few_8x" \
 --model_name_or_path "SpanBERT/spanbert-base-cased"\
 --per_device_train_batch_size 8\
 --per_device_eval_batch_size 8\
 --learning_rate 5e-5\
 --weight_decay 1e-6\
 --lr_scheduler_type "constant" \
 --num_train_epochs 10 \
 --request_slots_list people date time first_name last_name \
 --overwrite_output_dir \
 --do_train \
 --do_eval \
 --do_predict \
 --logging_steps 0.2\
 --eval_steps 0.2\
 --evaluation_strategy steps \
 --save_strategy steps \
 --load_best_model_at_end \
 --save_total_limit 2 \
 --metric_for_best_model f1 \
 --greater_is_better True \
 --drop_data_ratio "0.1_High_8x"\
```
