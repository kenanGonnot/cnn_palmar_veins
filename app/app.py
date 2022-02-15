import os

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

list_users = ["kenan", "faycal", "lorenzo"]


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/test/')
def test():
    return render_template('test.html', content=["hugo", "patrik", "dandy"], r=2)


@app.route('/identification')
def identification():
    return render_template('identification.html')


@app.route('/verification', methods=["POST", "GET"])
def verification():
    if request.method == "POST":
        user = request.form["username"]
        for usr in list_users:
            if user == usr:
                return redirect(url_for('connected', usr=user))
        return redirect(url_for('error', error="Username not found"))
    else:
        return render_template('verification.html')


@app.route('/connected/<usr>')
def connected(usr):
    return render_template('connected.html', usr=usr)


@app.route('/error/<error>')
def error(error):
    return render_template('error.html', error=error)


# @app.route('/admin')
# def admin():
#     return redirect(url_for('/home'))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
