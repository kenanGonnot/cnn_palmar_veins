import os

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/test/')
def test():
    return render_template('test.html', content=["hugo", "patrik", "dandy"], r=2)


@app.route('/verification')
def verification():
    return render_template('verification.html')


@app.route('/identification')
def identification():
    return render_template('identification.html')


# @app.route('/admin')
# def admin():
#     return redirect(url_for('/home'))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
