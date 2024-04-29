# LLaMA 3 SOTA Extractor

This repository contains codes for the LLaMA 3 SOTA extractor, which solves [Task 4 in CLEF 2024 SimpleText Lab](https://sites.google.com/view/simpletext-sota/home)

## Task overview from Task 4 website

The **SOTA?** shared task is defined on a dataset of Artificial Intelligence scholarly articles. There are two kinds of articles: one reporting (Task, Dataset, Metric, Score) tuples and another kind that do not report the TDMS tuples. For the articles reporting TDMS tuples, all the reported TDMS annotations are provided in a separate file accompanying the scraped full-text of the articles. The extraction task is defined as follows.

_Develop a machine learning model that can distinguish whether a scholarly article provided as input to the model reports a TDMS or not. And for articles reporting TDMSs, extract all the relevant ones._

Given the recent upsurge in the developments in generative AI in the form of Large Language Models (LLMs), creative LLM-based solutions to the task are particularly invited. The task does not place any restrictions on the application of open-sourced versus closed-sourced LLMs. Nonetheless, development of open-sourced solutions are encouraged.

For more background information on this task, we recommend the following publications:

- Salomon Kabongo, Jennifer D'Souza and Sören Auer (2023). Zero-Shot Entailment of Leaderboards for Empirical AI Research. In: ACM/IEEE Joint Conference on Digital Libraries. JCDL 2023.Santa Fe, NM, USA, 2023, pp. 237-241. https://doi.org/10.1109/JCDL57899.2023.00042 (Pre-print available at https://arxiv.org/abs/2303.16835)

- Salomon Kabongo, Jennifer D'Souza, & Sören Auer (2023). ORKG-Leaderboards: a systematic workflow for mining leaderboards as a knowledge graph. International Journal on Digital Libraries (2023). https://doi.org/10.1007/s00799-023-00366-1
