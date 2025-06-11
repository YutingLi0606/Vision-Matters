from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
import re, json, torch
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()    


parser.add_argument("--benchmark", default='mathverse', type=str)
parser.add_argument("--root", default='/DATA/Evaluation/math-datasets', type=str)
parser.add_argument("--model_path", default='/DATA/models/QVQ-72B-Preview', type=str)
parser.add_argument("--model", default='qwen2vl-7b', type=str)
parser.add_argument("--out_dir", default='/DATA/Evaluation/model_answer', type=str)

args = parser.parse_args()

MODEL_PATH = args.model_path
if args.benchmark == 'wemath':
    BENCHMARK_PATH = os.path.join(args.root, 'We-Math/we-math.json')
if args.benchmark == 'mathvista':
    BENCHMARK_PATH = os.path.join(args.root, 'MathVista/mathvista.json')
if args.benchmark == 'mathverse':
    BENCHMARK_PATH = os.path.join(args.root, 'MathVerse/mathverse.json')
if args.benchmark == 'mathvision':
    BENCHMARK_PATH = os.path.join(args.root, 'MathVision/mathvision.json')


out_dir = os.path.join(args.out_dir, args.benchmark)
os.makedirs(out_dir, exist_ok=True)
OUT_PATH = f'{out_dir}/{args.model}_answer.json'


def extract_qwen_bbox(response):
    response = response.split()
    response = [s for s in response if '\\boxed' in s]
    if len(response) < 1:
        response = ''
    else:
        response = response[0]
        response = response.split('{')[-1]
        response = response.split('}')[0]
    return response

def extract_qwen(response):
    try: 
        for res in response.split('\n'):
            if 'Answer' in res or 'answer' in res:
                return res[-2]
    except:
        return ''

if '72b' in args.model:
    tensor_parallel_size = 4
    gpu_memory_utilization = 0.7
elif 'qvq' in args.model:
    tensor_parallel_size = 2
    gpu_memory_utilization = 0.9
else:
    tensor_parallel_size = 2
    gpu_memory_utilization = 0.7
    
llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
    dtype=torch.bfloat16, 
    gpu_memory_utilization=gpu_memory_utilization, 
    enforce_eager=True, 
    tensor_parallel_size=tensor_parallel_size,
    # max_model_len = 2048,
    # max_num_seqs = 32
)

sampling_params = SamplingParams(
    # max_tokens=4096,
    max_tokens=16384,
    temperature=0.2,
    stop_token_ids=[],
)
# sampling_params = SamplingParams(
#     temperature=temp,
#     top_p=0.001,
#     repetition_penalty=1.05,
#     max_tokens=2048,
#     stop_token_ids=[],
# )

processor = AutoProcessor.from_pretrained(MODEL_PATH,max_pixels = 1204224)

with open(BENCHMARK_PATH, 'r', encoding='utf-8') as f:
    total_data = json.load(f)

# total_data = total_data[:100]

outputs = []
correct_cnt = 0
for index in range(int(len(total_data)/5000)+1):
    if 5000*(index+1)<len(total_data):
        inputs_data = total_data[5000*index:5000*(index+1)]
    else:
        inputs_data = total_data[5000*index:]
    prompt_list = []
    # inputs_data = inputs_data[:10]
    with torch.no_grad():
        with tqdm(inputs_data) as progress:
            store_images = []
            for i, item in enumerate(progress):
                img = item['images'][0]
                inst = item['query'].replace('<image>','') + "\nPlease reason step by step, and put your final answer within \\boxed{}"
                p0 = "Please reason step by step to get the final answer"
                messages = [
                    # {"role": "system", "content": "Please reason step by step to get the answer"},
                    #{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "system", "content": "You are a helpful assistant."},
                    # {"role": "system", "content":"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img,
                                "min_pixels": 224 * 224,
                                "max_pixels": 1280 * 28 * 28,
                            },
                            {"type": "text", "text": f'{inst}'},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                llm_inputs = {
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                }
                prompt_list.append(llm_inputs)
                store_images.append(img)
        llm_outputs = llm.generate(prompt_list, sampling_params=sampling_params)
        # llm_outputs = llm.generate(prompt_list)

    with tqdm(inputs_data) as progress:
        for i, _ in enumerate(progress):
            response = llm_outputs[i].outputs[0].text
            # print(response)
            # response = extract_qwen(response)
            # print(response)
            # response = response.strip()
            
            o = inputs_data[i]
            o['model_answer'] = response
            outputs.append(o)
            
            # gt = inputs_data[i]['response']
            
            # if response == gt:
            #     correct_cnt += 1
            
                    
    print(f'index:{index}')

# print(correct_cnt / len(total_data))
with open(OUT_PATH, 'w', encoding='utf-8') as out_file:
    json.dump(outputs, out_file, ensure_ascii=False, indent=4)
