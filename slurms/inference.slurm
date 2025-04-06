#!/bin/sh
#python dsgi.py --model-name $MODEL_NAME --prompt-type $PROMPT_TYPE --p --prod
python dsgi.py --model-name $MODEL_NAME --prompt-type $PROMPT_TYPE --n --prod --s 3 --t 0.7
