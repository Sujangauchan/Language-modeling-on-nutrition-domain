# Nutrition Chatbot Project

## Abstract

In the age of algorithmically mediated information, public perceptions of nutrition are increasingly shaped by unverified digital narratives. This project aims to build a chatbot from scratch that delivers reliable information on personal nutritional requirements based on individual profile features such as age, BMI, daily activity, and medical conditions. It implements Natural Language Processing techniques to train bigram and transformer-based language models on the OpenWebText corpora dataset, and textual data from 57 government-issued and academic nutrition PDF documents. By training a Transformer-based language model on both corpora, the developed chatbot is intended to provide nutritional recommendations tailored to a person's specific characteristics based on individual queries. This presents a valuable opportunity in creating a global impact by utilizing a large volume of existing text-based academic and governmental research documents to provide reliable nutritional guidance for the public, especially in developing countries where nutritional deficiency has been a major issue.

## Introduction

Language model pretraining has shown to be effective in various natural language processing tasks from natural language inference, question answering to paraphrasing.  This project explores the evolution of language models, from statistical models to the transformer architecture, and their application in providing reliable nutritional information.  The transformer architecture, with its ability to capture rich relationships between distant words through the self-attention mechanism, forms the basis for the chatbot developed in this project.  The project also touches on recent advancements in language model research, such as DeepSeek's Multi-Head Latent Attention (MLA), which enhances memory efficiency.

## Project Overview

This project addresses the issue of unreliable nutritional information by developing a chatbot that provides dynamic and personalized nutritional guidance.  The chatbot is trained on a combination of the OpenWebText corpus and 57 scientific and governmental nutritional documents from sources like WHO, FAO, IOM, and USDA.  The user provides their age, height, weight, BMI, existing medical conditions, medications taken, and level of physical activity, and the chatbot provides tailored nutritional advice.

## Technical Details

### Bigram Language Model
* A character-level bigram language model is implemented to generate text based on nutritional data extracted from scientific PDF reports.
* Preprocessing steps include:
    * Text extraction from PDF files using the PyPDF2 library.
    * Cleaning text to keep only alphanumeric characters, spaces, and basic punctuation.
    * Mapping unique characters to integers for tensor conversion.
* The model architecture involves mapping each character index to a vector and calculating score distributions for the next character using SoftMax and cross-entropy loss.
* Training is carried out using the AdamW optimizer.

### Transformer-Based Language Model
* A transformer-based language model is implemented using PyTorch, trained on the OpenWebText corpus and extracted text from 57 nutrition-related PDF documents.
* Key components include:
    * Token and positional embedding.
    * Multi-head attention mechanism with Query, Key, and Value projections.
    * Causal masking.
    * Feedforward networks.
* The training process uses the AdamW optimizer, and the final model is saved using pickle.dump.

## Results

The report details the optimization process and the results obtained for both the bigram and transformer-based language models, including loss curves and text generation examples.

## Ethical Considerations

The report also discusses the ethical considerations associated with the development and deployment of such a nutrition chatbot.
