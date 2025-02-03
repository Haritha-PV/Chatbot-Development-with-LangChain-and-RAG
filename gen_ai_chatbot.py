
import os
from flask import Flask, request, jsonify, render_template
from gen_ai import get_llm

# Correct path to the data directory
path = 'data'

# Initialize the chatbot by passing the path
chatbot = get_llm(path)

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/get_answer', methods=["POST"])
def get_answer():
    question = request.form['question']  # Get the review from the form
    answer = chatbot.invoke({'question': question, 'chat_history': []})
    return jsonify(answer)

if __name__ == "__main__":
    app.run()
