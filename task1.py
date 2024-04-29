"""
This script determines whether an AI paper reports TDMS scores or not using the abstract.
"""
import os
import pandas as pd
import regex as re
from tqdm import tqdm
os.environ['HF_HOME'] = '/mnt/netstore1_home/jakub.suran/.cache'
import transformers
import torch

DATASET_ROOT = 'sota/dataset'

torch.set_default_dtype(torch.float16)

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("cuda")
else:
    torch.set_default_device("cpu")


def extract_paths_and_labels(train=False):
    if train:
        root = os.path.join(DATASET_ROOT, 'train')
    else:
        root = os.path.join(DATASET_ROOT, 'validation')
    
    documents = os.listdir(root)
    latex_files = []
    labels = []

    for doc in documents:
        if os.path.exists(f'{root}/{doc}/{doc}.tex'):
            latex_files.append(f'{root}/{doc}/{doc}.tex')
            with open(f'{root}/{doc}/annotations.json', 'r') as f:
                content = f.read()
            if content.startswith("unanswerable"):
                labels.append(False)
            else:
                labels.append(True)
        else:
            raise FileNotFoundError(f'{root}/{doc}/{doc}.tex')

    print(f'Processed {len(latex_files)} latex files')
    assert len(latex_files) == len(labels)
    return latex_files, labels


def extract_abstract(doc, patterns):
    abstract = []
    for section_start, section_end in patterns:
        abstract.extend(re.findall(section_start + r'(.*?)' + section_end, doc, re.S))
    return "".join(abstract)



def get_summaries(path, df):
    sys_instructions = """
    You are part of a information extraction system that extracts tasks, datasets, metrics, and scores from AI research papers.
    You will be provided a latex source code of a section from an AI research paper.
    Based on this source code, summarize the section and focus on tasks, datasets, metrics, and scores.
    Only provide information that is mentioned in the section and be truthful about your response.
    """
    user_instructions = """
    Summarize the following section with an emphasis on identifying and detailing any tasks, datasets, metrics, and scores mentioned. Highlight the context in which these elements are discussed and any results or evaluations presented. Provide one short paragraph.
    Section:
    """

    summaries = []
    for section in df[df["paper"] == path]["context"]:
        messages = [{"role": "system", "content": sys_instructions},
                    {"role": "user", "content": user_instructions + section},]
        
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = pipeline(prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
        summary = outputs[0]["generated_text"][len(prompt):]
        summaries.append(summary)

    return "".join(summaries)
    

def determine_tdms(pipeline, summaries):
    messages = [{"role": "system", "content": "You are a helpful assistant that can determine whether an AI research paper reports task, dataset, metrics, and scores or not using the summaries of the sections of the paper. Based on these summaries, determine whether the paper reports TDMS scores or not. If the paper reports TDMS scores, provide a boolean value of True, otherwise provide a boolean value of False. Do not write any other information, do not provide and introduction, just write the boolean value."},
                {"role": "user", "content": "Does the AI paper report task, dataset, metric, scores tuples based on the following section summaries? Section summaries:" + summaries},]

    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(prompt, max_new_tokens=10000, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
    tdms = outputs[0]["generated_text"][len(prompt):]
    return tdms



model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
device="cuda")

files, labels = extract_paths_and_labels(True)
 
# patterns = [
#         (r'\\begin{abstract}', r'\\end{abstract}'),  # Standard abstract block
#         (r'\\abstract{', r'\\section'),  # Alternative abstract command
#         (r'\\section{Abstract}', r'\\section')  # Abstract as a section
# ]

df = pd.read_csv("wasp_sections.tsv", sep="\t")


results = []
for path, label in tqdm(zip(files[:100], labels[:100]), total=100):
    summaries = get_summaries(path, df)
    print(summaries[:100])
    tdms = determine_tdms(pipeline, summaries)
    results.append(tdms)
    # print(f"Label: {label}, TDMS: {tdms}, Paper: {path}")

print(f"Number of papert reporting TDMS - predicted: {sum([1 for r in results if r])}")
print(f"Number of papert reporting TDMS - true: {sum([1 for r in labels if r])}")

pd.DataFrame({"path": files[:100], "label": labels[:100], "tdms": results}).to_csv("results.tsv", index=False, sep="\t")