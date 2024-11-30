import sys
import os
import certifi
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

# Import custom utilities and pipeline
from networksecurity.utils.ml_util.model.estimator import NetworkModel
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.constants.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# Load environment variables
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
ca = certifi.where()

# Initialize MongoDB client
import pymongo
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Initialize FastAPI app
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")


# Define routes
@app.get("/", tags=["authentication"])
async def index():
    """Redirect to the documentation page."""
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    """Endpoint to trigger model training."""
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        logging.info("Model training completed successfully.")
        return Response("Training is successful")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise NetworkSecurityException(e, sys)


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    """Endpoint for batch predictions from uploaded CSV files."""
    try:
        # Read uploaded CSV file into DataFrame
        df = pd.read_csv(file.file)

        # Load preprocessor and model
        preprocessor = load_object("final_models/preprocessor.pkl")
        final_model = load_object("final_models/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        # Perform prediction
        y_pred = network_model.predict(df)
        df['predicted_column'] = y_pred

        # Print predictions in the terminal
        print("\n--- Predictions ---")
        print(y_pred)
        print("\n--- DataFrame with Predictions ---")
        print(df)

        # Save the predictions to a file
        output_path = "prediction_output/output.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        # Convert DataFrame to HTML table
        table_html = df.to_html(classes='table table-striped')

        # Return the table in the response
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise NetworkSecurityException(e, sys)


# Run app
if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8080)

    
