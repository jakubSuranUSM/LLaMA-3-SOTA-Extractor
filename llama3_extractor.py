import os
import token
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from latex_extractor import extract_paths_and_labels
import json
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score

load_dotenv()

# configure environment variables
access_token = os.getenv('ACCESS_TOKEN')
hf_home = os.getenv('HF_HOME')

# set hugingface home directory
if not hf_home or not access_token:
    raise OSError("Please set the HF_HOME and ACCESS_TOKEN environment variables")
else:
    os.environ['HF_HOME'] = hf_home


if torch.cuda.is_available():
    torch.set_default_device("cuda")
    device = "cuda"
else:
    torch.set_default_device("cpu")
    device = "cpu"

summarization_system_prompt = """
You are part of a information extraction system that extracts tasks, datasets, metrics, and scores from AI research papers.
You will be provided a section from an AI research paper as latex source code.
Summarize the section, with the focus on tasks, datasets, metrics, and scores.
Scores are most likely to be numerical and found in LaTeX tables.
Do not provide any information that is not mentioned in the section.
Only talk about the most important tasks, datasets, metrics, and scores based on the context of the section.
"""

summarization_user_prompt = """
Summarize the following section with an emphasis on identifying and detailing any tasks, datasets, metrics, and scores mentioned:
"""

final_extraction_system_prompt = """
You are an information extraction model.
You will be provided a summary of sections from an AI research paper.
Based on the summary, extract leaderboards in a form of task, dataset, metric, and score (TDMS) tuples.
Scores should be extracted as numbers.
If the summary does not contain enough information for leaderboard construction, write an empty array.
If the summary contains leaderboards (TDMS tuples), provide them in a structured format as an array of python dictionary, one for each leaderboard.
This is an exmaple of an output: [{'LEADERBOARD': {'Task': 'Arithmetic Reasoning', 'Dataset': 'GSM8K', 'Metric': 'Accuracy', 'Score': '87.3'}}]
Extract only zero or one TDMS tuple, in a rare case, extract 2 or 3 TDMS tuples.
Write ONLY the array of dictionaries as showed in the example or an empty array, do not provide any extra words outsite of the array.
"""

final_extraction_user_prompt = """
Based on the following summaries, extractleaderboards (task, dataset, metric, and score tuples) that you believe are the most relevant, if any are present. 
Write an array of dictionaries or an empty array in not TDMS tuples are present.
This is an example output: [{'LEADERBOARD': {'Task': 'Arithmetic Reasoning', 'Dataset': 'GSM8K', 'Metric': 'Accuracy', 'Score': '87.3'}}] 
Sumaries:
"""


def summarize_section(section, tokenizer, model):
    """
    Summarize a section of a research paper.
    
    :param section: The section to summarize.
    :param tokenizer: The tokenizer to use.
    :param model: The model to use.
    
    :return: The summary of the section.
    """
    message = [{"role": "system", "content": summarization_system_prompt},
                {"role": "user", "content": summarization_user_prompt + section},]
    return pass_message_through_llama3(message, tokenizer, model)


def final_extraction(summary, tokenizer, model):
    """
    Extract task, dataset, metric, and score tuples from a summary.
    
    :param summary: The summary to extract from.
    :param tokenizer: The tokenizer to use.
    :param model: The model to use.
    
    :return: The extracted task, dataset, metric, and score tuples.
    """
    message = [{"role": "system", "content": final_extraction_system_prompt},
                {"role": "user", "content": final_extraction_user_prompt + summary},]
    return pass_message_through_llama3(message, tokenizer, model)


def pass_message_through_llama3(messages, tokenizer, model):
    """
    Pass a message through the Meta-Llama-3-8B-Instruct model.
    
    :param messages: The messages to pass through the model.
    :param tokenizer: The tokenizer to use.
    :param model: The model to use.
    
    :return: The response from the model.
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
     
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    return response
    

def predict_validation_set():    
    """
    Predict the TDMS tuples from the validation set using the Meta-Llama-3-8B-Instruct model.
    """
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        token=access_token
    )
    df = pd.read_csv("validation_sections.tsv", sep="\t")
    papers, _ = extract_paths_and_labels(False) 
    for paper in tqdm(papers, total=len(papers)):
        sections = df[df["paper"] == paper]["context"]

        summary = []
        for section in sections:
            summary.append(summarize_section(section, tokenizer, model))

        summary = "".join(summary)
        tdms = final_extraction(summary, tokenizer, model)
        tdms = tdms.replace("\n", "")
        
        with open("results.txt", "a") as f:
            f.write(tdms + "\n")
            f.close()
    

def evaluate_predictions():
    """
    Evaluate the predictions from the Meta-Llama-3-8B-Instruct model.
    """
    _, labels = extract_paths_and_labels(False)
    
    predictions = []
    with open("results.txt", "r") as f:
        for line in f:
            if line.startswith("[]"):
                line = "unanswerable\n"
            line.replace("\n", "")
            predictions.append(line)
            
    labels = [str(label) for label in labels[:7]]
    
    binary_labels = [1 if label.startswith('unpredictable') else 0 for label in labels]
    binary_predictions = [1 if prediction.startswith('unpredictable') else 0 for prediction in predictions]

    # Calculate precision, recall, and F1 score for the first task
    precision = precision_score(binary_labels, binary_predictions)
    recall = recall_score(binary_labels, binary_predictions)
    f1 = f1_score(binary_labels, binary_predictions)

    print("Metrics for recugnising if a paper reports scores or not (binary classification)")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    
    # Calculate precision, recall, and F1 score for the second task (TDMS extraction)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')

    print("Macro Precision:", precision)
    print("Macro Recall:", recall)
    print("Macro F1 Score:", f1)
    
            


def main():
    predict_validation_set()
    evaluate_predictions()
    
        
        
if __name__ == "__main__":
    main()
    
    