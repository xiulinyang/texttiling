# Texttiling
Group project for the course Computational Discourse Modeling

Devika, Eliza, Xiulin, and Jingni

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


## How to run 
- Create a virtual environment with ```conda create -n texttiling python=3.8``` and install all the dependencies.
- To train the model, run ```python deberta.py```
- To get the predictions, run ```python predict.py``` 
