# DATA304 Final Project: Hierarchical Multi-Label Text Classification

**Student ID:** 2025952015  
**Course:** DATA304 - Big Data Analysis (Korea University)  
**Date:** December 20, 2025

## 📌 Project Overview
This repository contains the full implementation for the final project on **Hierarchical Multi-Label Text Classification**. The solution leverages a Graph Neural Network (GNN) combined with a BERT-based encoder. The training pipeline utilizes **Silver Label Generation** (inspired by TaxoClass) and **Self-Training** strategies to handle the lack of labeled training data.

## 🛠️ Prerequisites & Installation

The code is implemented in Python 3. It requires **PyTorch** and the **Hugging Face Transformers** library.

### Install Dependencies
Before running the pipeline, ensure all necessary packages are installed:

torch>=2.0.0

transformers>=4.30.0

networkx>=3.0

google-generativeai>=0.3.0 # not necessary if the llm_annotate.py file is not executed

scipy>=1.10.0

numpy>=1.24.0

pandas>=2.0.0

tqdm>=4.65.0

langdetect>=1.0.9
