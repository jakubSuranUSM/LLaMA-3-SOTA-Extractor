import os
import regex as re
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from dotenv import load_dotenv
import queue
import threading

load_dotenv()

access_token = os.getenv('ACCESS_TOKEN')
hf_home = os.getenv('HF_HOME')
# set hugingface home directory
if not hf_home or not access_token:
    raise OSError("Please set the HF_HOME and ACCESS_TOKEN environment variables")
else:
    os.environ['HF_HOME'] = hf_home

model = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET_ROOT = 'sota/dataset'

def extract_paths_and_labels(train=False):
    """
    Extracts paths and labels for all latex files in the dataset for specific split.
    
    :param train: If True, extracts paths for training split, otherwise for validation split.
    
    :return: (list of paths to latex files, list of labels to the corresponding files)
    """
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
            labels.append(content)
        else:
            raise FileNotFoundError(f'{root}/{doc}/{doc}.tex')

    print(f'Processed {len(latex_files)} latex files')
    assert len(latex_files) == len(labels)
    return latex_files, labels

    
def extract_content(path):
    """
    Extracts content of the LaTeX file by removing LaTeX code of figures
    which take up a lot of space, while keeping the caption of the figure.
    
    :param path: Path to the LaTeX file
    :return: Content of the LaTeX file
    """
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
    """
    Extracts caption from the LaTeX figure.
    """
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
    """
    Extracts all sections from the LaTeX document, if the document does not contain any 
    sections, tries to extract subsections.
    
    :param doc: LaTeX document
    :return: List of sections and subsections
    """
    sections = re.split(r'\\section', doc)
    if len(sections) <= 1:
        sections = re.split(r'\\subsection', doc)
    return sections


def get_section_name(section):
    """
    Extracts the name of the section.
    
    :param section: Section from the LaTeX document
    :return: Name of the section
    """
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
    """
    Calculates the length of the section using a specific tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
    
    input_ids = tokenizer.encode(section, return_tensors="pt")
    return input_ids.shape[1]


class PrintThread(threading.Thread):
    """
    Thread that prints the results to the file.
    Uses a queue to get the results from the processing threads.
    """
    def __init__(self, queue, total, filename):
        threading.Thread.__init__(self)
        self.total = total
        self.queue = queue
        self.filename = filename

    def printfiles(self, rows):
        with open(self.filename, 'a') as output:
            for row in rows:
                output.write(row + '\n')

    def run(self):
        for _ in tqdm(range(self.total)):
            result = self.queue.get()
            self.printfiles(result)
            self.queue.task_done()


class ProcessThread(threading.Thread):
    """
    Thread that processes the paths from the queue.
    Extracts extracts all sections and subsections of the LaTeX document and writes them to the file.
    Writes the information about the section to the file in format: paper, section_name, section_length, context.
    Uses a queue to get the paths from the main thread and to put the results to the printing thread.
    """
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
        


def extract_secitons_of_all_papers(filename, train=False):
    """
    Extracts all sections and subsections of the LaTeX documents for all papers in the dataset.
    Writes the information about the section to the file in format: paper, section_name, section_length, context.
    Uses threads to process the paths and to print the results for faster results.
    """
    # create the queues
    pathqueue = queue.Queue()
    resultqueue = queue.Queue()
    paths, _ = extract_paths_and_labels(train)

    # Write the header to the file.
    with open(filename, 'w') as output:
        output.write('paper\tsection_name\tsection_length\tcontext\n')

    # get number of CPUs
    num_cpus = os.cpu_count()
    if num_cpus is None:
        num_cpus = 1
    print("Number of CPUs:", num_cpus)
    
    # spawn threads to process
    for _ in range(0, num_cpus):
        t = ProcessThread(pathqueue, resultqueue)
        t.setDaemon(True)
        t.start()

    # spawn threads to print
    t = PrintThread(resultqueue, len(paths), filename)
    t.setDaemon(True)
    t.start()

    # add paths to queue
    for path in paths:
        pathqueue.put(path)

    # wait for queue to get empty
    pathqueue.join()
    resultqueue.join()


def print_stats(filename): 
    """
    Reads the file with extracted sections and prints statistics.
    :param filename: Name of the file with extracted sections.
    """
    df = pd.read_csv(filename, sep='\t')

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
    print("Number of sections with lengths longer than 8000 (LLaMA 3 context window size):", long_sections)
    
    
def main():
    filename = 'validation_sections.tsv'
    extract_secitons_of_all_papers(filename, False)
    print_stats(filename)
    
    
if __name__ == "__main__":
    main()
    # no_tdms_path = 'sota/dataset/validation/0705.1367/0705.1367.tex'
    # tdms_path = 'sota/dataset/validation/2110.01997v1/2110.01997v1.tex'
    # tdms_annotations = 'sota/dataset/validation/2110.01997v1/annotations.json'

    # content = extract_content(tdms_path)
    # content = content.replace('\t', ' ')
    # content = content.replace('\n', ' ')
    # sections = extract_all_sections(content)   
    # for section in sections:
    #     print(section[:100])