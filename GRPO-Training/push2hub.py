import os
import json
from PIL import Image
from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData

def generate_dataset(data_path):
    base_dir = os.path.dirname(data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for item in data:
        problem = item.get("problem") or item.get("query") or item.get("question") or ""
        answer = item.get("answer") or item.get("response") or item.get("solution") or ""

        image_rel_path = item["images"][0]
        image_abs_path = os.path.join(base_dir, image_rel_path)
        
        if not os.path.exists(image_abs_path):
            raise FileNotFoundError(f"Missing image: {image_abs_path}")
            
        image = Image.open(image_abs_path, "r")
        yield {
            "images": [image],
            "problem": problem.strip(),
            "answer": answer.strip()
        }

def main():
    datasets_config = {
        "geoqa-r1v-8k-rotated": {
            "train": ["/HOME/paratera_xy/pxy368/HDD_POOL/yuting/datasets/geoqa-r1v-8k-rotated/geoqa-r1v-8k-rotated.json"],
            "test": ["/HOME/paratera_xy/pxy368/HDD_POOL/CODE/scripts/math_dataset/MathVision/mathvision-1.json"]
        },
        
        
    }

    for ds_name, splits in datasets_config.items():
        dataset_splits = {}
        for split, paths in splits.items():
            if not paths:
                continue
                
            datasets_list = []
            for path in paths:
                ds = Dataset.from_generator(
                    generate_dataset,
                    gen_kwargs={"data_path": path}
                )
                datasets_list.append(ds)
            
            if datasets_list:
                dataset_splits[split] = Dataset.from_list(
                    [item for ds in datasets_list for item in ds]
                )

        if dataset_splits:
            dataset = DatasetDict(dataset_splits).cast_column(
                "images", Sequence(ImageData())
            )

            repo_name = f"Yuting6/{ds_name}"

            dataset.push_to_hub(repo_name)
            print(f"数据集 {ds_name} 已上传到 {repo_name}")

if __name__ == "__main__":
    main()