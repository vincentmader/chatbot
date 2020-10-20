from datetime import datetime as dt

import flask
from flask import Flask, render_template, send_from_directory, request, jsonify


app = Flask(__name__)

MESSAGES = {
    'messages': [
        dict(timestamp=109878978, msg_content='Test 1', sender='customer'),
        dict(timestamp=109923890, msg_content='Test 2', sender='bot'),
        dict(timestamp=109872390, msg_content='Test 3', sender='bot'),
    ]
}


@app.route('/')
def chatbot():
    return render_template('index.html')


@app.route('/<customerID>/<chatID>', methods=['POST'])
def chatbot_post(customerID, chatID):

    # print(request.form)

    timestamp = dt.now().timestamp()
    msg_content = request.form
    msg_content = [i for i in msg_content.keys()][0]  # TODO: this is shitty
    sender = 'customer'

    if msg_content:
        MESSAGES['messages'].append(dict(
            timestamp=timestamp, msg_content=msg_content, sender=sender
        ))

    return render_template('index.html')


@app.route('/messageList/', methods=['GET'])
def messageList():
    # return MESSAGES
    return jsonify(MESSAGES)


if __name__ == '__main__':
    app.run(port=5001)
