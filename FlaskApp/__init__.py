import flask
from flask import Flask, render_template, send_from_directory, request


app = Flask(__name__)


@app.route('/')
def chatbot():
    return render_template('index.html')


@app.route('/<customerID>/<chatID>', methods=['POST'])
def chatbot_post(customerID, chatID):

    msgs = {
        1: {
            1: [
                {'sender': 'A', 'content': 'Du Fischkopf', 'timestamp': 1603034040},
                {'sender': 'B', 'content': 'Du Kopffisch', 'timestamp': 1603034020},
                {'sender': 'B', 'content': 'Du Cockfisch', 'timestamp': 1603034340},
                {'sender': 'A', 'content': 'Du Fickkosch', 'timestamp': 1603022000},
            ]
        }
    }

    query = request.form
    print(query)

    return render_template('chatbot.html')


if __name__ == '__main__':
    app.run(port=5001)
