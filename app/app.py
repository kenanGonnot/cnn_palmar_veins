from flask import Flask, jsonify

app = Flask(__name__)
resources = {r"/api/*": {"origins": "*"}}
app.config["CORS_HEADERS"] = "Content-Type"
app.config['JSON_SORT_KEYS'] = False


@app.route('/')
def home():
    return jsonify({"Message": "This is your flask app with docker"})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
