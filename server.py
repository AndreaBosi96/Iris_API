from model import Model  # Import the python file containing the ML model
from flask import Flask, request, render_template  # Import flask libraries

class_model = Model()
# Initialize the flask class and specify the templates directory
app = Flask(__name__, template_folder="templates")

# Default route set as 'home'
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")  # Render home.html


# Route 'classify' accepts GET request
@app.route("/classify", methods=["GET"])
def classify_type():
    try:
        sepal_len = request.args.get("slen")  # Get parameters for sepal length
        sepal_wid = request.args.get("swid")  # Get parameters for sepal width
        petal_len = request.args.get("plen")  # Get parameters for petal length
        petal_wid = request.args.get("pwid")  # Get parameters for petal width

        # Get the output from the classification model
        variety = class_model.classify(sepal_len, sepal_wid, petal_len, petal_wid)

        # Render the output in new HTML page
        return render_template("output.html", variety=variety)
    except Exception as e:
        return "Error: " + str(e)


# Run the Flask server
if __name__ == "__main__":
    app.run(debug=False)
