import os
from datasets import Dataset

DEFAULT_BASE_DIR = "./data/evaluation"

math_token = 8192
med_token = 1500

math_batch = 5
med_batch = 15
other_batch = 10

math_datasets = ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
med_datasets = ["medqa", "medmcqa", "pubmedqa", "clinical_knowledge", "college_biology",
                "college_medicine", "medical_genetics", "professional_medicine", "anatomy", "entropy_bench"]

max_token_dataset = {ds: math_token for ds in math_datasets}
max_token_dataset.update({ds: med_token for ds in med_datasets})

max_batch_size = {ds: math_batch for ds in math_datasets}
max_batch_size.update({ds: med_batch for ds in med_datasets})

is_multi_choice = set(med_datasets)


def detect_question_type(question: str, answer: str = None) -> str:
    choice_indicators = [
        "A)", "B)", "C)", "D)", "E)", "F)",
        "A.", "B.", "C.", "D.", "E.", "F.",
        "A\u3001", "B\u3001", "C\u3001", "D\u3001", "E\u3001", "F\u3001",
        "A\uff0e", "B\uff0e", "C\uff0e", "D\uff0e", "E\uff0e", "F\uff0e",
        "Choose", "Select", "Which", "What is the correct",
        "\u9009\u62e9", "\u9009\u51fa", "\u54ea\u4e2a", "\u54ea\u9879", "\u6b63\u786e\u7b54\u6848",
    ]
    question_lower = question.lower()
    for indicator in choice_indicators:
        if indicator.lower() in question_lower:
            return "multiple_choice"
    if answer:
        if answer.strip().upper() in ["A", "B", "C", "D", "E", "F"]:
            return "multiple_choice"
    return "open_ended"


def standardize_dataset_format(dataset, dataset_name: str = None) -> list:
    if hasattr(dataset, 'to_list'):
        data_list = dataset.to_list()
    else:
        data_list = list(dataset)

    standardized_data = []
    for i, item in enumerate(data_list):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item at index {i} is not a dictionary: {type(item)}")

        question = item.get('question', item.get('Question', item.get('prompt', '')))
        answer = item.get('answer', item.get('Answer', item.get('response', '')))

        if not question:
            raise ValueError(f"Dataset item at index {i} missing 'question' field: {item}")

        question_type = detect_question_type(question, answer)
        standardized_item = {
            'question': question,
            'answer': answer or '',
            'type': question_type,
        }
        for key, value in item.items():
            if key not in ('question', 'answer', 'type'):
                standardized_item[key] = value

        standardized_data.append(standardized_item)

    return standardized_data


def load_dataset_from_exploration(path: str, default_segmentation='I need to carefully'):
    dataset = Dataset.from_json(path)
    new_dataset = []
    for item in dataset:
        for gen in item['responses']:
            new_dataset.append({
                'question': item['question'],
                'answer': item['answer'],
                'prompt': item['prompt'] + gen.split(default_segmentation)[0] + "</EXPLORATION>",
            })
    dataset_name = path.split("/")[-2]
    return new_dataset, dataset_name


def load_dataset_by_name(name: str, base_dir: str = None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    if name.endswith(".parquet"):
        dataset_path = name
        dataset_name = os.path.splitext(os.path.basename(name))[0]
    else:
        dataset_path = os.path.join(base_dir, f"{name}.parquet")
        dataset_name = name
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    raw_dataset = Dataset.from_parquet(dataset_path)
    standardized_dataset = standardize_dataset_format(raw_dataset, dataset_name)
    return standardized_dataset, dataset_name


def save_as_dataset(data, save_path):
    newdataset = []
    for item in data:
        for gen in item['responses']:
            newdataset.append({
                'question': item['question'],
                'answer': item['answer'],
                'prompt': item['prompt'] + gen,
            })
    dataset = Dataset.from_list(newdataset)
    dataset.to_parquet(save_path)
