# Search-o1 testing and analysis

This repository contains our implementation and evaluation of the **Search-o1** framework, which is a search-enhanced, multi-step reasoning agent.  We used and modified a version of the original codebase and evaluated it on a small HotpotQA subset already included in this repository, and ready to run and test. The following are a brief explanation on how to run, use, and interpret results.



## Introduction
**Search-o1** is a reasoning framework that enhances LLMs by integrating:
- web search  
- document inspection  
- multi-step reasoning  
- search queries  

Unlike standard RAG systems, Search-o1:
- searches **multiple times**
- refines queries
- extracts relevant evidence
- updates its reasoning at each step

This repo uses UCSD's DSMLP-cluster implementation and our modifications to the search backend we used Serper.dev instead of Bing.


## Repository Structure

-  `scripts/` is where  `run_search_o1.py` is stored, which handles query generation, retrieval, and model reasoning.  
- The `data/QA_Datasets` is where our data is stored under the file name `hotpotqa.json` 
- The `models/` directory defines model loading and inference utilities for running **Qwen** or any other supported LLM.   
- All produced outputs—such as **JSONL logs** of model thoughts, search calls, and final answers—are automatically stored in the `outputs/` directory for each dataset split.  

Together, this structure provides a clean separation between code, configuration, and generated results so you can easily navigate and modify any part of the system.
If you



## Instructions 

### 1. SSH into the Cluster
Connect to your remote cluster where the model will run:

```bash
ssh username@your-cluster-address
cd /path/to/repository
```

###2 Launch the environment with A30 GPU

```bash
launch-scipy-ml.sh -W DSC180A_FA25_A00 -c 8 -m 32 -g 1 -v a30
```

###3 Install Dependencies

```bash
pip install -r requirements.txt
```

###4 Set up your API keys
bash
```
export JINA_API_KEY="your_jina_key"
export SERPER_API_KEY="your_serper_key"
```

###5 Run the Search-o1 script
bash
```
python scripts/run_search_o1.py \
  --dataset_name hotpotqa \
  --split test \
  --subset_num 10 \
  --max_search_limit 5 \
  --max_turn 10 \
  --top_k 10 \
  --max_doc_len 3000 \
  --use_jina True \
  --model_path "Qwen/Qwen2.5-3B-Instruct" \
  --jina_api_key $JINA_API_KEY \
  --bing_subscription_key $SERPER_API_KEY \
  --bing_endpoint "https://google.serper.dev/search"
```

###6 Check Outputs Saved to `outputs/`









## 📄 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{Search-o1,
  author       = {Xiaoxi Li and
                  Guanting Dong and
                  Jiajie Jin and
                  Yuyao Zhang and
                  Yujia Zhou and
                  Yutao Zhu and
                  Peitian Zhang and
                  Zhicheng Dou},
  title        = {Search-o1: Agentic Search-Enhanced Large Reasoning Models},
  journal      = {CoRR},
  volume       = {abs/2501.05366},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2501.05366},
  doi          = {10.48550/ARXIV.2501.05366},
  eprinttype    = {arXiv},
  eprint       = {2501.05366},
  timestamp    = {Wed, 19 Feb 2025 21:19:08 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2501-05366.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```



## LICENSE 
MIT License

Copyright (c) 2025 Xiaoxi Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

