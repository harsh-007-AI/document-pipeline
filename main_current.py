import os
import json
import logging
import sys
import uuid
from io import BytesIO
from flask import Flask, request
from google.cloud import storage, firestore
from google.cloud import aiplatform
from PyPDF2 import PdfReader
from vertexai.preview.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold

# Configure thread-safe logger
class ThreadSafeLogger:
    def __init__(self):
        self.logger = logging.getLogger("doc-processor")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.propagate = False

        # Silence third-party logs
        logging.getLogger("google").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("vertexai").setLevel(logging.WARNING)

logger = ThreadSafeLogger().logger

app = Flask(__name__)

# Mandatory health check endpoint
@app.route('/')
def health_check():
    return "OK", 200

# Initialize cloud clients
try:
    firestore_client = firestore.Client()
    storage_client = storage.Client()
    aiplatform.init(project=os.getenv("PROJECTID"), location=os.getenv("LOCATION")) # Initialize Vertex AI
    model = GenerativeModel("gemini-2.0-flash-001") # Initialize Gemini Model
except Exception as e:
    logger.critical(f"Client initialization failed: {str(e)}", exc_info=True)
    sys.exit(1)

def process_file(file_data):
    try:
        logger.info(f"Processing {file_data['name']}")
        bucket = storage_client.bucket(file_data['bucket'])
        blob = bucket.blob(file_data['name'])
        
        # Retrieve file content from Cloud Storage
        if file_data['name'].lower().endswith('.pdf'):
            content = blob.download_as_bytes()
            reader = PdfReader(BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = blob.download_as_text()

        # Define the prompts for Vertex AI generation
        system_prompt = """
                    You are a medical document processing system. 
                    1. Validate if the provided text is a medical document or report.
                    2. If it is a medical document:
                        - Extract the following fields:
                            - Patient Name: (If available)
                            - Date of Birth: (If available)
                            - Medical Record Number: (If available)
                            - Date of Service: (If available)
                            - Diagnosis: (If available)
                            - Procedure: (If available)
                            - Medications: (If available)
                            - Lab Results: (If available)
                            - Notes: (If available)
                        - Format the extracted fields as a JSON object.
                    3. If the provided text is not a medical document, return:
                        "Not a medical document/report."
                    """
        user_prompt = f"validate and extract from :\n{text}"
        response = model.generate_content(
            [
                Part.from_text(system_prompt),
                Part.from_text(user_prompt)
            ],
            generation_config={
                "max_output_tokens": 8192,
                "temperature": 0.4,
                "top_p": 1,
                "top_k": 32,
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
        )
        content_result = response.candidates[0].content.text
        # Write to the parsed_documents collection
        firestore_client.collection("parsed_documents").document(file_data['name']).set({
            "content": content_result,
            "status": "processed",
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        # Write to the uploaded_files collection
        uploaded_file_doc = {
            "id": file_data.get("name"),  # Using file name as identifier
            "bucket": file_data.get("bucket"),
            "name": file_data.get("name"),
            "event_type": file_data.get("event_type"),
            "notification_config": file_data.get("notification_config"),
            "publish_time": file_data.get("publish_time"),
            "content": content_result,
            "status": "processed"
        }
        firestore_client.collection("uploaded_files").document(file_data['name']).set(uploaded_file_doc)

        # Write to the file_category collection with default values
        file_category_doc = {
            "id": str(uuid.uuid4()),
            "name": "Default Category",
            "loinc_code": None,
            "loinc_name": None,
            "parent_category_id": None,
            "source_type": None,
            "confidence_score": None,
            "model_info": None,
            "model_category": None
        }
        firestore_client.collection("file_category").document(file_data['name']).set(file_category_doc)

        # Write to the audit_logs collection to record the action
        audit_log_doc = {
            "id": str(uuid.uuid4()),
            "action": "process_file",
            "document_name": file_data.get("name"),
            "status": "processed",
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        firestore_client.collection("audit_logs").document(file_data['name']).set(audit_log_doc)

        logger.info(f"File processed and stored: {file_data['name']}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_data.get('name', 'unknown')}: {str(e)}", exc_info=True)
        return False

@app.route('/', methods=['POST'])
def handle_request():
    try:
        envelope = request.get_json()
        if not envelope or 'message' not in envelope:
            logger.warning("Invalid request format")
            return "Bad Request", 400
        
        # Extract Pub/Sub message attributes
        msg = envelope.get("message", {})
        attributes = msg.get("attributes", {})
        file_data = {
            "bucket": attributes.get("bucketId"),
            "name": attributes.get("objectId"),
            "event_type": attributes.get("eventType"),
            "notification_config": attributes.get("notificationConfig"),
            "publish_time": msg.get("publishTime")
        }

        if not file_data["bucket"] or not file_data["name"]:
            logger.error("Missing bucket or file name in message attributes")
            return "Bad Request", 400

        success = process_file(file_data)
        return ("OK", 200) if success else ("Processing Failed", 500)

    except Exception as e:
        logger.critical(f"Fatal error in request handler: {str(e)}", exc_info=True)
        return "Internal Server Error", 500

# For local development; in Cloud Run with Gunicorn, this block is ignored.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
