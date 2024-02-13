# Texttiling
Group project for the course Computational Discourse Modeling

Devika, Eliza, Xiulin, and Jingni

## Link to slides presentation 
- https://docs.google.com/presentation/d/18ouy0a7sPB47ktmkNpevgSBXbSa7JEwuBh2MA7QcwSQ/edit?usp=sharing
  

## How to run 
- Create a virtual environment with ```conda create -n texttiling python=3.8``` and install all the dependencies.
- To train the model, run ```python deberta.py``` (please change the corresponding input file directories if you want to fine-tune the model on different datasets)
- To get the predictions, run ```python predict.py``` (please change the corresponding input file directories)


## Data
General: From Brown Corpus

- 2500 sentences from a variety of published works (multi-genre), written in American English)
- 230 subsection boundaries 


Transcript: From ICSI Meeting Corpus 

- 2500 sentences from transcribed meetings, annotated for boundaries based on topic-change
- 8 subsection boundaries 

Literature:  From Project Gutenberg 

- first 2500 sentences of Mary Shelley's Frankenstein, with paragraph boundaries
- 573 subsection boundaries 

Wiki: From Wiki-727K Corpus 

- 2500 sentences from English Wikipedia documents, and their subsection boundaries
- 290 subsection boundaries


## Data pre-processing: 

Pre-processing the corpus data was done in four scripts in the preprocessing folder: 
- preprocess_literature.ipynb
- preprocess_wiki.ipynb
- preprocess_transcript.ipynb
-  preprocess.py (general) 
 
For each respective dataset (literature, wiki, transcript, and general), we looped through each document, tokenizing it into sentences, checking for empty sections. In wiki, section headings were removed. Each sentence was appended to a dictionary, with the structure: {‘genre’: ‘genre’, ‘document_id’:  ‘genre+document_id’, ‘sent’: ‘single sentence’, ‘boundary’: ‘yes or no’}. The boundary value was ‘yes’ if the sentence occurred at the start of a new document, or was the sentence directly after a subsection delineator. Dictionaries for each genre were appended to a list, and stored in a tsv file. 

### Creation of balanced dataset: 

- script: test_train_balanced_10000.ipynb

Balanced dataset was created by randomly shuffling documents within each of the four genre-specific lists of dictionaries, maintaining the internal order of dictionaries (i.e., sentences) with the same document ID. 

The four genre lists were then split into train and test sets, using a 85-15 train-test split. 

The four train sets were combined and randomly shuffled, maintaining the internal order within documents. The four tests sets were combined in the same fashion. This resulted in a balanced train set, containing 2125 dictionaries from each genre, and a balanced test set, containing 375 dictionaries from each genre. 

### Creation of by-genre datasets: 

