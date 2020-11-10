# Chatbot

## Setup

From the project directory, run

        pip3 install -r requirements
        
## Running the server

Run

        cd Flaskapp
        python3 __init__.py

Afterwards, navigate to localhost:5001 in your browser.

## Interacting with a minimal-functionality chatbot

If no network has been trained, the bot is configured
to answer by repeating your question. This should work
out of the box.

## Training a model 

There will probably be multiple prototype networks that are developed and 
tested at the same time. Which model is used to generate an answer in the chat 
can be specified in the 

        ./FlaskApp/__init__.py

by importing the appropriate "get_answer" function

### first attempt: seq2seq RNN (see [this chatbot tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html))

from networks.greedy_attn_seq2seq_RNN import get_answer

To train the recurrent network, download the dataset from 
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
and move its contents into the project directory at

        ./training_data/cornell movie-dialogs corpus/
        
Run the network training & testing scripts via

        python3 ./train.py
        python3 ./test.py
        
We could use this network later to test against our own.

### second attempt: clustering & RNN

...

