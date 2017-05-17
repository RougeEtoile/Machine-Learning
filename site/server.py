from flask import Flask, render_template
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('d3BasicComparison.html')


@app.route("/mpld3")
def mpld3():
    return render_template('example.html')

@app.route("/pured3")
def d3():
    return render_template('pured32.html')

if __name__ == "__main__":
    app.run()