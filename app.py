from flask import Flask, render_template, request
import numpy as np
import boto3
import json
import os 

app = Flask(__name__)

# Configure ton endpoint SageMaker
runtime = boto3.client('sagemaker-runtime',
    region_name=os.environ["AWS_REGION"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
)

endpoint_name = os.environ["SAGEMAKER_ENDPOINT"]

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
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        result = json.loads(response["Body"].read().decode())
        prediction = result[0]

    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))  # Render définit PORT automatiquement
    app.run(host="0.0.0.0", port=port)
