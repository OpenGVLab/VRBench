# VRBench
A Benchmark for Multi-Step Reasoning in Long Narrative Videos

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2506.10857-b31b1b.svg)](https://arxiv.org/abs/2506.10857) -->

## Updates

[06/2025] VRBench has been accepted to ICCV2025!

## Overview

![overall_structure](./assets/teaser_arxiv.png)
We propose VRBench, the first long narrative video benchmark crafted for evaluating large models' multi-step reasoning capabilities, addressing limitations in existing evaluations that overlook temporal reasoning and procedural validity. It comprises 1,010 long videos (with an average duration of 1.6 hours), along with 9,468 human-labeled multi-step question-answering pairs and 30,292 reasoning steps with timestamps. These videos are curated via a multi-stage filtering process including expert inter-rater reviewing to prioritize plot coherence. We develop a human-AI collaborative framework that generates coherent reasoning chains, each requiring multiple temporally grounded steps, spanning seven types (e.g., event attribution, implicit inference). VRBench designs a multi-phase evaluation pipeline that assesses models at both the outcome and process levels. Apart from the MCQs for the final results, we propose a progress-level LLM-guided scoring metric to evaluate the quality of the reasoning chain from multiple dimensions comprehensively. Through extensive evaluations of 12 LLMs and 16 VLMs on VRBench, we undertake a thorough analysis and provide valuable insights that advance the field of multi-step reasoning.

## Citation

If you find our repo useful for your research, please consider citing our paper:

        @article{yu2025vrbench,
          title={VRBench: A Benchmark for Multi-Step Reasoning in Long Narrative Videos},
          author={Yu, Jiashuo and Wu, Yue and Chu, Meng and Ren, Zhifei and Huang, Zizheng and Chu, Pei and Zhang, Ruijie and He, Yinan and Li, Qirui and Li, Songze and others},
          journal={arXiv preprint arXiv:2506.10857},
          year={2025}
        }
