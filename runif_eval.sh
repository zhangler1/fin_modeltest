#!/bin/bash

/home/llm/mambaforge/envs/py310-torch23-cu121/bin/python -m instruction_following_eval.evaluation_main \
--input_data="instruction_following_eval/data/input_data.jsonl" \
--input_response_data="instruction_following_eval/data/input_response_data_spark_lite_2024-07-25 17:46:18.jsonl" \
--output_dir="instruction_following_eval/data"
exit 0
