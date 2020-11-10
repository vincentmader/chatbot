
# TODOs



network
-------

- [ ] data preparation
    - [ ] word tokenization (also sentence?)
    - [ ] make lowercase, remove stopwords
    - [ ] removing numbers, short words
    - [ ] encoder

    - [ ] semantic clustering
        - [ ] by conversation / by message ?
        - [ ] multiple layers
        - [ ] manual topic/intent tagging

- [ ] build model
    - [ ] input: current conversation history -> memory of context

    - [ ] intent tagging -> which cluster?
    - [ ] find action (find the right verb) -> how?
    - [ ] named-entity recognition (nltk?) -> list of saved entities
        - [ ] if logged-in: account data -> entities

    - [ ] recurrent network (?)
        - [ ] input: 
            - [ ] topic/intents
            - [ ] action 
            - [ ] entities 
        - [ ] output: 
            - [ ] answer (train on intents & answers from dataset)

    - [ ] output: answer

- [ ] answer modification
    - [ ] switch out words for synomyms
    - [ ] spell check


training & testing
------------------

- [ ] training
    - [ ] use remote gpu (cloud) -> choose provider

- [ ] testing
    - [ ] ...


deploy
------

- [o] frontend 
    - [X] basic structure
        - [X] start button
        - [X] chat history
            - [X] get message list and display
        - [X] input field & send button
            - [X] post message info to backend on send
    - [ ] beautify / css
        - [ ] profile info
        - [ ] typing animation -> lottie?

- [ ] backend
    - [ ] security
        - [ ] connect to account info

- [ ] move project on server -> online testing


approximate file structure
--------------------------

networks
    recurrent_seq2seq
        clustering
            create N-dim blobs
                in: conversations
                out: blobs 
            ...
        
        context recognition
            intent (point in blob -> nearest cluster center)
                in: message (message history)
                out: topic/grouping & subgrouping ?
            find entities
        
        recurrent_net
            ...
        
        data_preparation
            tokenize
            ...
        
