#!/bin/bash

/home/llm/mambaforge/envs/py310-torch23-cu121/bin/python -m finDatasets.instruction_following_eval.evaluation_main \
--input_data="finDatasets/instruction_following_eval/data/input_data.jsonl" \
--input_response_data="finDatasets/instruction_following_eval/data/input_response_data_spark_lite_2024-07-26 14:27:13.jsonl" \
--output_dir="finDatasets/instruction_following_eval/data"
exit 0
