import json
from tqdm import tqdm
import re
import os
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default='/HOME/paratera_xy/pxy368/HDD_POOL/yuting/inference/sample/math-8k/qwen2.5vl-7b-math8k-sample16_rejection_sampling.json', type=str)
parser.add_argument("--model_path", default='/HOME/paratera_xy/pxy368/HDD_POOL/yuting/models/Qwen2.5-32B-Instruct', type=str)
parser.add_argument("--out_dir", default='/HOME/paratera_xy/pxy368/HDD_POOL/yuting/inference/sample', type=str)
parser.add_argument("--dataset", default='math-8k', type=str)
parser.add_argument("--model", required=True, type=str, help="Qwen2.5VL-7B-Instruct")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
OUT_PATH = f'{args.out_dir}/{args.dataset}/{args.model}_judged-answers.json'

with open(args.dataset_path, 'r', encoding='utf-8') as f:
    data_list = json.load(f)

llm = LLM(model=args.model_path, tensor_parallel_size=2)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)


def extract_boxed_answer(response):
    match = re.search(r'\\boxed\{(.*?)\}', response)
    if match:
        return match.group(1).strip()
    return response.strip()

## Math-8k
def extract_gt(response):
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

prompt_template = (
    "Your task is to determine whether the response expresses the same meaning as the answer of a question.\n"
    "Question: {question}\n"
    "Ground Truth Answer: {gt}\n"
    "Model Response: {response}\n"
    "Instructions:\n"
    "- If the response is correct, output 'Yes'.\n"
    "- If the response is incorrect, output 'No'.\n"
    "Output only 'Yes' or 'No'."
)


for item in tqdm(data_list):
    query = item['query'].replace('<image>', '')  
    gt = extract_boxed_answer(item['response']) 
    model_answers = item['model_answers']  
    
    model_answer_judge = []
    
    for model_answer in model_answers:
        extracted_answer = extract_boxed_answer(model_answer)
        messages = [{"role": "user", "content": prompt_template.format(question=query, gt=gt, response=extracted_answer)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampling_params = SamplingParams(max_tokens=2048, temperature=0)  
        output = llm.generate([text], sampling_params)[0].outputs[0].text.strip()
        
        judgment = "Yes" if "Yes" in output else "No"
        model_answer_judge.append({
            "model_answer": extracted_answer,
            "judge": judgment
        })
    item['model_answer_judge'] = model_answer_judge

with open(OUT_PATH, 'w', encoding='utf-8') as out_file:
    json.dump(data_list, out_file, ensure_ascii=False, indent=4)

print(f"Processed {len(data_list)} items and saved results to {OUT_PATH}")