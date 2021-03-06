from datetime import datetime as dt

import flask
from flask import Flask, render_template, send_from_directory, request, jsonify

from networks.greedy_attn_seq2seq_RNN import get_answer as get_answer_from_RNN
from networks.greedy_attn_seq2seq_RNN.get_answer import *

from config import PATH_TO_PROJECT


app = Flask(__name__)

MESSAGES = {
    'messages': [
        dict(
            timestamp=dt.now().timestamp(),
            msg_content='hello . how may i help you ?',
            sender='bot'
        ),
    ]
}


def get_answer(question):
    if not os.path.exists(
        os.path.join(
            PATH_TO_PROJECT, 'saved_models',
            'test1_seq2seq_rnn', 'encoder'
        )
    ):
        return question
    else:
        return get_answer_from_RNN(question)


@ app.route('/')
def chatbot():
    return render_template('index.html')


@ app.route('/<customerID>/<chatID>', methods=['GET', 'POST'])
def chatbot_post(customerID, chatID):

    # print(request.form)

    timestamp = dt.now().timestamp()
    msg_content = request.form
    if msg_content:
        # TODO: this is shitty
        msg_content = [i for i in msg_content.keys()][0]
    else:
        msg_content = ''
    sender = 'customer'

    if msg_content:
        MESSAGES['messages'].append(dict(
            timestamp=timestamp, msg_content=msg_content, sender=sender
        ))

        MESSAGES['messages'].append(dict(
            timestamp=dt.now().timestamp(),
            msg_content=get_answer(msg_content),
            sender='bot'
        ))

    return render_template('index.html')


@ app.route('/messageList/', methods=['GET'])
def messageList():
    # return MESSAGES
    return jsonify(MESSAGES)


if __name__ == '__main__':
    app.run(port=5001)
