import os
os.environ['HF_HOME'] = '/mnt/netstore1_home/jakub.suran/.cache'
import pandas as pd
import torch
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_default_dtype(torch.float16)

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("cuda")
else:
    torch.set_default_device("cpu")

llama2 = "meta-llama/Llama-2-7b-chat-hf"
llama3 = "meta-llama/Meta-Llama-3-8B"


tokenizer = AutoTokenizer.from_pretrained(llama3)
model = AutoModelForCausalLM.from_pretrained(llama3)


def extract_sota():
    print('here1')
    input_text = '''
        <s>[INST] <<SYS>>
        You will be provided a text from scientific paper.
        Based on this text, extract task, dataset, metric, and score that is the most relevant to the text.
        Only extract one task, one dataset, one metric, and one score.
        Don't include any other information in your response. 
        <</SYS>>
        
        Intelligent transportation \cite{5959985} has been receiving increasing attention recently, and for the applications, such as assisted driving, violation detection, and congestion forecasting, accurate and efficient cognition and reasoning over the traffic events captured by video cameras is extremely important. As shown by previous works \cite{10.3115/1073012.1073017,35179}, well-designed datasets are often crucial for the development, adaptation and evaluation of different data-driven approaches. 
        This indicates the significance of creating comprehensive and challenging benchmarks for video causal reasoning
        and cognitive development of models, 
        that explore the underlying causal structures of various traffic events. To this end, we introduce a novel dataset, SUTD-TrafficQA (Traffic Question Answering), to facilitate the research of causal reasoning in complex traffic scenarios.

        In our dataset, to help develop models for addressing several major and concerning issues in intelligent transportation, we design 6 challenging reasoning tasks, which require exploring the complex causal structures within the inference process of the traffic events. As shown in Figure \ref{fig:example},
        these tasks correspond to various traffic scenarios involving both road-agents and surroundings, and the models are required to forecast future events, infer past situations, explain accident causes, provide preventive advice, and so on. 

        To present these reasoning tasks, video question answering \cite{zhu2017uncovering} is a natural and effective choice, and is used for our dataset construction, since to accurately answer the given questions, the models need to acquire strong capabilities of performing various levels of logical reasoning and spatio-temporal cognition for the events.

        In some application scenarios, (e.g., assisted driving), the computational resource and energy budget can be constrained.
        Thus 
        both the inference accuracy and the computation efficiency are important for video event reasoning in these scenarios.
        Existing video QA methods \cite{Kim_2020_CVPR,Le_2020_CVPR,lei-etal-2020-tvqa} mainly focus on strengthening the reasoning accuracy without emphasizing much efficiency, and most of
        these works apply fixed computation pipelines 
        to answer different questions, while ignoring to conduct adaptive and efficient computation resource allocation based on the logic structure behind reasoning over video events. 

        In this paper, to achieve reliable and efficient video reasoning, we propose \textbf{Eclipse}, an \textbf{E}ffi\textbf{c}ient g\textbf{li}m\textbf{pse} network.
        Specifically, considering there is often large redundancy among video frames, via dynamic inference, our network adaptively determines where to skip and glimpse at each step, and what computation granularity needs to be allocated for the glimpsed frame. Such a dynamic reasoning scheme avoids feature extraction for the irrelevant segments in the video, and hence significantly reduces the overall computation cost towards 
        reliable and efficient reasoning.
        It is noteworthy that both the determination of selecting a glimpse frame and the decision of computation granularity for each glimpse are essentially discrete operations, which are 
        not trivial to optimize. To handle this issue, an effective joint Gumbel-Softmax mechanism is also introduced in this paper, which makes our Eclipse framework fully differentiable and end-to-end trainable. 

        To the best of our knowledge, this is the first work that simultaneously performs adaptive frame localization and feature granularity determination in a novel dynamic reasoning process for reliable and efficient causal reasoning and video QA. A joint Gumbel-Softmax operation is also introduced in this work to optimize the two decisions jointly.
        [/INST]
    '''
    
    input_text = '''
    <s> [INST] <<SYS>>
    You will be provided a table in tex format.
    Extract the data from the table and provide it in a structured format.
    Extract the best score from the table.
    <</SYS>>
        
    \begin{table}[t]
    \vspace{-0.2cm}
    \caption{Results on SUTD-TrafficQA dataset. 
    }
    \setlength\abovecaptionskip{-2cm}
    \setlength\belowcaptionskip{-2cm}
    \setlength{\tabcolsep}{10pt}
    \scriptsize
    \begin{center}
    \begin{tabular}{l|ccc}
    \hline
    Models & Setting-1/4 & Setting-1/2 & GFLOPs  \\\hline
    Q-type (random)& 25.00 & 50.00 & - \\ 
    QE-LSTM & 25.21 & 50.45 & - \\ 
    QA-LSTM & 26.65 & 51.02 & -  \\ \hline
    Avgpooling & 30.45& 57.50 & 252.69 \\ 
    CNN+LSTM & 30.78 & 57.64 & 252.95  \\ 
    I3D+LSTM & 33.21 & 54.67 & 108.72  \\
    VIS+LSTM \cite{ren2015exploring} & 29.91 & 54.25 & 252.80\\
    BERT-VQA \cite{Yang_2020_WACV} & 33.68 & 63.50 & 266.77 \\ 
    TVQA \cite{lei2018tvqa} & 35.16 & 63.15 & 252.11  \\ 
    HCRN \cite{Le_2020_CVPR} & 36.49 & 63.79 & 2051.04 \\ 
    \textbf{Eclipse}  & \textbf{37.05} & \textbf{64.77} & \textbf{28.14} \\ \hline
    \textit{Human}    &  95.43    & 96.78 & -\\ \hline
    \end{tabular}
    \end{center}
    \label{table:baseline_comparison}
    \vspace{-0.5cm}
    \end{table}

    [/INST]
    '''
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    print('here2')

    # Generate simplified text
    output_ids = model.generate(input_ids, max_length=2000)
    print('here3')
    simplified_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(simplified_text)


def main():
    extract_sota()

if __name__ == '__main__':
    main()