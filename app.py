from flask import Flask, render_template, request
import numpy as np
import boto3
import json
import qrcode

app = Flask(__name__)

# Configure ton endpoint SageMaker
ENDPOINT_NAME = "pytorch-training-2025-05-05-16-41-52-802"

runtime = boto3.client('sagemaker-runtime', region_name='eu-west-1')

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        # Récupérer les champs du formulaire
        age = float(request.form["age"])
        sibsp = float(request.form["sibsp"])
        parch = float(request.form["parch"])
        fare = float(request.form["fare"])

        # Créer le payload
        payload = np.array([[age, sibsp, parch, fare]], dtype=np.float32).tolist()

        # Appeler l’endpoint SageMaker
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        result = json.loads(response["Body"].read().decode())
        prediction = result[0]

    return render_template("form.html", prediction=prediction)

## Générer un QR code au démarrage
#@app._before_first_request
#def generate_qr():
#    import os
#    if not os.path.exists("static/qr.png"):
#        url = "http://localhost:5000/"  # à remplacer par l'URL publique si déployé
#        img = qrcode.make(url)
#        img.save("static/qr.png")
#
if __name__ == "__main__":
    app.run(debug=True)
