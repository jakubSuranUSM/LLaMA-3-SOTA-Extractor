import os
from numpy import average
import regex as re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_ROOT = 'sota/dataset'
access_token = "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"


def extract_paths(train=False):
    if train:
        root = os.path.join(DATASET_ROOT, 'train')
    else:
        root = os.path.join(DATASET_ROOT, 'validation')
    
    documents = os.listdir(root)
    latex_files = []

    for doc in documents:
        if os.path.exists(f'{root}/{doc}/{doc}.tex'):
            latex_files.append(f'{root}/{doc}/{doc}.tex')
        else:
            raise FileNotFoundError(f'{root}/{doc}/{doc}.tex')

    print(f'Processed {len(latex_files)} latex files')
    return latex_files

    
def extract_content(path):
    with open(path, 'r', encoding="ISO-8859-1") as f:
        content = f.read()
    
    parts = re.split(r'\\begin\{figure|\\end\{figure', content)
    result = ''
    for i, part in enumerate(parts):
        if i % 2 != 0:
            result += extract_caption(part) + '\n'
        else:
            result += part + '\n'
    
    return result


def extract_caption(string):
    parts = re.split(r'\\caption', string)
    caption = ''
    if len(parts) > 1:
        parts = parts[1:]
        for part in parts:
            result = ''
            stack = []
            for c in part:
                if c == '{': stack.append(c)
                elif c == '}': stack.pop()
                
                if stack: result += c
                else: break
            caption += '\\caption' + result.strip() + '\n'
        
    return caption


def extract_all_sections(doc):
    sections = re.split(r'\\section|\\subsection', doc, re.S)
    return sections


def get_section_name(section):
    stack = []
    if section.startswith('{'):
        stack.append('{')
        section = section[1:]
        i = 0
        while stack and i < len(section):
            if section[i] == '{':
                stack.append('{')
            elif section[i] == '}':
                stack.pop()
            i += 1
        return section[:i-1]
    
    return ""

def get_section_length(section):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)
    
    input_ids = tokenizer.encode(section, return_tensors="pt")
    return input_ids.shape[1]


import queue  # or queue in Python 3
import threading

class PrintThread(threading.Thread):
    def __init__(self, queue, total):
        threading.Thread.__init__(self)
        self.total = total
        self.queue = queue

    def printfiles(self, rows):
        with open('wasp_sections.tsv', 'a') as output:
            for row in rows:
                output.write(row + '\n')

    def run(self):
        for i in tqdm(range(self.total)):
            result = self.queue.get()
            self.printfiles(result)
            self.queue.task_done()

class ProcessThread(threading.Thread):
    def __init__(self, in_queue, out_queue):
        threading.Thread.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            path = self.in_queue.get()
            result = self.process(path)
            self.out_queue.put(result)
            self.in_queue.task_done()

    def process(self, path):
        content = extract_content(path)
        content = content.replace('\t', ' ')
        content = content.replace('\n', ' ')
        sections = extract_all_sections(content)
        result = []
        for section in sections:
            section_name = get_section_name(section)
            section_length = get_section_length(section)
            result.append(f"{path}\t{section_name}\t{section_length}\t{section}")
        return result
        

pathqueue = queue.Queue()
resultqueue = queue.Queue()
paths = extract_paths(True)

with open('wasp_sections.tsv', 'w') as output:
    output.write('paper\tsection_name\tsection_length\tcontext\n')


num_cpus = os.cpu_count()
if num_cpus is None:
    num_cpus = 1
print("Number of CPUs:", num_cpus)
# spawn threads to process
for i in range(0, num_cpus):
    t = ProcessThread(pathqueue, resultqueue)
    t.setDaemon(True)
    t.start()

# spawn threads to print
t = PrintThread(resultqueue, len(paths))
t.setDaemon(True)
t.start()

# add paths to queue
for path in paths:
    pathqueue.put(path)

# wait for queue to get empty
pathqueue.join()
resultqueue.join()


# doc_paths = extract_paths(True)

# df = pd.DataFrame(columns=['paper', 'section_name', 'section_length', 'context'])

# for doc_path in tqdm(doc_paths):
#     content = extract_content(doc_path)
#     content = content.replace('\t', ' ')
#     content = content.replace('\n', ' ')
#     sections = extract_all_sections(content)
#     for section in sections:
#         section_name = get_section_name(section)
#         section_length = get_section_length(section)
#         df.loc[-1] = [doc_path, section_name, section_length, section]
#         df.index = df.index + 1 
        
# df.to_csv('train_sections.csv', index=False, sep='\t')

df = pd.read_csv('train_sections.csv', sep='\t')

max_length_row = df.loc[df['section_length'].idxmax()]
print("Row with max section length:")
print(max_length_row)

# Print average section length
average_length = df['section_length'].mean()
print("Average section length:", average_length)

# Total number of sections
total_sections = len(df)
print("Total number of sections:", total_sections)

# Number of sections with lengths longer than 4000
long_sections = len(df[df['section_length'] > 4000])
print("Number of sections with lengths longer than 4000:", long_sections)