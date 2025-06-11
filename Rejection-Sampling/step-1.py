from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
import re
import json
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import Counter

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to the input dataset")
    parser.add_argument("--generation_model_path", required=True, type=str, help="Path to the generation model")
    parser.add_argument("--judgment_model_path", required=True, type=str, help="Path to the judgment model")
    parser.add_argument("--out_dir", required=True, type=str, help="Base output directory for all results")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name (e.g., math-8k)")
    parser.add_argument("--model", required=True, type=str, help="Qwen2.5VL-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples per input")
    return parser.parse_args()

def generate_model_answers(args):
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model}_{args.dataset}-rejection_sampling.json")

    llm = LLM(
        model=args.generation_model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        tensor_parallel_size=2,
    )

    processor = AutoProcessor.from_pretrained(args.generation_model_path, max_pixels=1204224)

    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        total_data = json.load(f)

    outputs = []
    for index in range((len(total_data) // 5000) + 1):
        start = index * 5000
        end = min((index + 1) * 5000, len(total_data))
        inputs_data = total_data[start:end]

        for seed in range(args.num_samples):
            sampling_params = SamplingParams(
                max_tokens=16384,
                temperature=0.6,
                seed=seed,
                stop_token_ids=[],
            )

            prompt_list = []
            for item in inputs_data:
                img = item['images'][0]
                inst = item['query'].replace('<image>', '')
                messages = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img,
                                "min_pixels": 224 * 224,
                                "max_pixels": 1280 * 28 * 28,
                            },
                            {"type": "text", "text": inst},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                mm_data = {"image": image_inputs} if image_inputs else {}
                prompt_list.append({"prompt": prompt, "multi_modal_data": mm_data})

            llm_outputs = llm.generate(prompt_list, sampling_params=sampling_params)

            for i, item in enumerate(inputs_data):
                response = llm_outputs[i].outputs[0].text
                if seed == 0:
                    o = item.copy()
                    o['model_answers'] = [response]
                    outputs.append(o)
                else:
                    outputs[start + i]['model_answers'].append(response)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    return out_path

def judge_model_answers(args, input_path):
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "judged-answers.json")

    llm = LLM(model=args.judgment_model_path, tensor_parallel_size=2)
    tokenizer = AutoTokenizer.from_pretrained(args.judgment_model_path)

    def extract_boxed_answer(response):
        match = re.search(r'\\boxed\{(.*?)\}', response)
        return match.group(1).strip() if match else response.strip()

    def extract_gt(response):
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

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

    with open(input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

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

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    return out_path

def filter_answers(args, input_path):
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}-RL-pos-neg.json")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = []
    for item in data:
        judges = item.get("model_answer_judge", [])
        if not judges:
            continue

        has_yes = any(j["judge"] == "Yes" for j in judges)
        has_no = any(j["judge"] == "No" for j in judges)
        if not (has_yes and has_no):
            continue

        yes_answers = []
        no_answers = []
        for judge, answer in zip(judges, item['model_answers']):
            if judge["judge"] == "Yes":
                yes_answers.append(answer)
            else:
                no_answers.append(answer)

        positive = max(yes_answers, key=len) if yes_answers else None
        negative = min(no_answers, key=len) if no_answers else None

        filtered_data.append({
            "query": item.get("query"),
            "response": positive,
            "rejected_response": negative,
            "images": item.get("images"),
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    return out_path

def calculate_scores_and_plot(args, input_path):
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    scored_json_path = os.path.join(out_dir, "scored.json")
    plot_path = os.path.join(out_dir, "score_distribution.png")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input JSON file not found at: {input_path}")

    with open(input_path, "r") as f:
        data = json.load(f)

    scores = []
    for item in data:
        if "captions" in item:
            del item["captions"]
        
        yes_count = sum(1 for judge in item["model_answer_judge"] if judge["judge"] == "Yes")
        no_count = sum(1 for judge in item["model_answer_judge"] if judge["judge"] == "No")
        total = yes_count + no_count
        
        score = yes_count / total if total != 0 else 0.0
        item["score"] = round(score, 2)
        scores.append(score)

    # 生成图表
    score_counts = Counter([round(s, 2) for s in scores])
    plt.figure(figsize=(12, 6))
    bars = plt.bar(score_counts.keys(), score_counts.values(), width=0.05, color='skyblue', edgecolor='black')
    
    plt.title("Score Distribution", fontsize=14)
    plt.xlabel("Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks([i/20 for i in range(0, 21)], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    with open(scored_json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    return scored_json_path, plot_path

if __name__ == "__main__":
    args = parse_arguments()

    step1_output = generate_model_answers(args)
    print(f"Step 1 completed: {step1_output}")
