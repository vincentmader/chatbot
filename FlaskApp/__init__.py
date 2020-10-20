import flask
from flask import Flask, render_template, send_from_directory, request


app = Flask(__name__)


@app.route('/')
def chatbot():
    return render_template('index.html')


@app.route('/<customerID>/<chatID>', methods=['POST'])
def chatbot_post(customerID, chatID):

    query = request.form
    print(query)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5001)
