import json
from tqdm import tqdm
import openai
import unicodedata
from openai import OpenAI
import base64
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
import time
import os
import argparse
parser = argparse.ArgumentParser()


parser.add_argument("--benchmark", default='mathverse', type=str)
parser.add_argument("--model", default='qwen2vl-7b', type=str)

args = parser.parse_args()

answer_path = f"/DATA/Evaluation/model_answer/{args.benchmark}/{args.model}_answer.json"
save_path = f"/DATA/Evaluation/judge/{args.benchmark}/{args.model}_answer.json"


with open(answer_path, 'r', encoding='utf-8') as f:
    data_list = json.load(f)
correct_cnt = 0

sampling_params = SamplingParams(max_tokens=2048,temperature=0)

llm = LLM(model='/DATA/models/Qwen2.5-32B-Instruct',tensor_parallel_size=2)
tokenizer = AutoTokenizer.from_pretrained('/DATA/models/Qwen2.5-32B-Instruct')
prompt_template = "Your task is to judge whether the response expresses the same meaning as the answer of a question.\nThe question is: {question}\nThe answer is: {gt}\nThe response is: {response}\nPlease check and compare them and then judge. If the response is correct, your output should be Yes. Otherwise, your output should be No. Directly give me your output."
prompt_lists = []
inst_ls = [] 


for i, item in enumerate(tqdm(data_list)):
    question = item['query'].replace('<image>','')
    model_answer = item['model_answer']
    if '<answer>' in model_answer:
        model_answer[model_answer.find('<answer>'):model_answer.find('</answer>')].replace('<answer>','').replace('</answer>','')
    elif '<CONCLUSION>' in model_answer:
        model_answer = model_answer[model_answer.find('<CONCLUSION>'):model_answer.find('</CONCLUSION>')].replace('<CONCLUSION>','').replace('</CONCLUSION>','')
    elif 'boxed' in model_answer:
        anchor = model_answer.split('\\boxed{')[-1]
        end = -anchor[::-1].find('}')-1
        model_answer = anchor[0:end].replace('\\boxed{','')
    else:
        model_answer = '\n'.join(model_answer.split('\n')[-3:])
    gt = item['response']
    messages = [{"role": "user", "content": prompt_template.format(gt=gt,response=model_answer,question=question)}]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    prompt_lists.append(text)
    inst_ls.append(prompt_template.format(gt=gt,response=model_answer,question=question))

finals = []
outputs = llm.generate(prompt_lists, sampling_params)
num = 0

for i in range(len(data_list)):
    response_text = outputs[i].outputs[0].text
    item = data_list[i]
    item['judge'] = response_text
    item['inst'] = inst_ls[i]
    finals.append(item)
    if 'Yes' in item['judge'] or 'yes' in item['judge']:
        num += 1

os.makedirs(f"/DATA/Evaluation/judge/{args.benchmark}/", exist_ok=True)
print(args.benchmark, num/len(data_list))
with open(save_path, 'w', encoding='utf-8') as out_file:
    json.dump(finals, out_file, ensure_ascii=False, indent=4)

os.makedirs(f"/DATA/Evaluation/results/{args.benchmark}/", exist_ok=True)
accuracy = num/len(data_list)
final_save_path = (
    f"/DATA/Evaluation/results/{args.benchmark}/{args.model}_accuracy_{accuracy:.4f}.json"
)

with open(final_save_path, 'w', encoding='utf-8') as out_file:
    json.dump({"accuracy": accuracy}, out_file, ensure_ascii=False, indent=4)