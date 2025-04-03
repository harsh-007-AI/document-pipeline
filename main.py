import os
import json
import logging
import sys
import base64
import uuid
from io import BytesIO
from flask import Flask, request
from google.cloud import storage, firestore, secretmanager
from PyPDF2 import PdfReader
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, Integer, Float, TIMESTAMP, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.orm import scoped_session
from google.cloud import aiplatform
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
        logging.getLogger("openai").setLevel(logging.WARNING)

logger = ThreadSafeLogger().logger

app = Flask(__name__)

# Mandatory health check endpoint
@app.route('/')
def health_check():
    return "OK", 200
@app.route('/debug-socket')
def debug_socket():
    try:
        files = os.listdir('/cloudsql')
        return f"Cloud SQL sockets: {files}", 200
    except Exception as e:
        return f"Error reading /cloudsql: {str(e)}", 500

def access_secret_version(secret_id, version_id="latest"):
    """Access the payload for the given secret version if one exists."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Initialize cloud clients
try:
    firestore_client = firestore.Client()
    storage_client = storage.Client()
    aiplatform.init(project=os.getenv("PROJECTID"), location=os.getenv("LOCATION")) # Initialize Vertex AI
    model = GenerativeModel("gemini-2.0-flash-001") # Initialize Gemini Model

    # Retrieve database credentials from Secret Manager
    db_user = access_secret_version("DB_USER")
    db_password = access_secret_version("DB_PASSWORD")
    db_name = access_secret_version("DB_NAME")
    db_ip=access_secret_version("DB_IP")
    cloud_sql_connection_name = access_secret_version("CONNECTION_NAME")
    logger.info(f"Cloud SQL Connection Name: {cloud_sql_connection_name}")
    logger.info(f"DB User: {db_user}")
    logger.info(f"DB Name: {db_name}")
    logger.info(f"IP : {db_ip}")
    # Update the database URI for PostgreSQL

    db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_ip}/{db_name}"     
    Base = declarative_base()

    engine = create_engine(db_uri)

    Session = scoped_session(sessionmaker(bind=engine))
    session = Session()

except Exception as e:
    logger.critical(f"Client initialization failed: {str(e)}", exc_info=True)
    sys.exit(1)

# Define the database schema
class UploadedFiles(Base):
    __tablename__ = 'UploadedFiles'
    id = Column(String(36), primary_key=True)
    gcp_file_name = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    uploaded_at = Column(TIMESTAMP, nullable=False)
    document_type = Column(String(36), ForeignKey('FileCategory.id'))
    patient_id = Column(String(36))
    uploaded_by = Column(String(36), nullable=False)
    deleted_by = Column(String(36))
    deleted_at = Column(TIMESTAMP)
    version_number = Column(Integer)
    last_accessed_at = Column(TIMESTAMP)
    last_accessed_by = Column(String(36))
    document_date = Column(TIMESTAMP)
    status = Column(String(50))
    hash_signature = Column(String(255))
    last_modified_date = Column(TIMESTAMP)
    last_modified_by = Column(String(36))
    mongo_history_id = Column(String(36))
    verified_by = Column(String(36))
    summary_verified_by = Column(String(36))
    content = Column(Text)

class FileCategory(Base):
    __tablename__ = 'FileCategory'
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    loinc_code = Column(String(50))
    loinc_name = Column(String(255))
    parent_category_id = Column(String(36))
    source_type = Column(String(50))
    confidence_score = Column(Float)
    model_info = Column(Text)
    model_category = Column(String(255))

class AuditLogs(Base):
    __tablename__ = 'AuditLogs'
    id = Column(String(36), primary_key=True)
    action = Column(String(50), nullable=False)
    endpoint = Column(String(255), nullable=False)
    status_code = Column(Integer, nullable=False)
    request_data = Column(JSON)
    response_data = Column(JSON)
    user_id = Column(String(36))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    sensitivity_level = Column(String(50))
    organization_id = Column(String(36))

# Create tables
Base.metadata.create_all(engine)

def ensure_collections_exist():
    collections = ["UploadedFiles", "FileCategory", "AuditLogs"]
    for collection_name in collections:
        collection_ref = firestore_client.collection(collection_name)
        if not collection_ref.get():
            logger.info(f"Creating collection: {collection_name}")
            # Firestore collections are created automatically when a document is added
            collection_ref.document().set({})

def log_audit(action, endpoint, status_code, request_data, response_data, user_id):
    audit_log = AuditLogs(
        id=str(uuid.uuid4()),
        action=action,
        endpoint=endpoint,
        status_code=status_code,
        request_data=request_data,
        response_data=response_data,
        user_id=user_id
    )
    session.add(audit_log)
    session.commit()

    firestore_client.collection("AuditLogs").add({
        "action": action,
        "endpoint": endpoint,
        "status_code": status_code,
        "request_data": request_data,
        "response_data": response_data,
        "user_id": user_id,
        "created_at": firestore.SERVER_TIMESTAMP
    })

def get_file_category(file_data):
    # Placeholder logic for determining file category
    category_name = "Default Category"
    loinc_code = None
    loinc_name = None
    parent_category_id = None
    source_type = "Human suggested"
    confidence_score = None
    model_info = None
    model_category = None

    file_category_id = str(uuid.uuid4())
    file_category = FileCategory(
        id=file_category_id,
        name=category_name,
        loinc_code=loinc_code,
        loinc_name=loinc_name,
        parent_category_id=parent_category_id,
        source_type=source_type,
        confidence_score=confidence_score,
        model_info=model_info,
        model_category=model_category
    )
    session.add(file_category)
    session.commit()

    firestore_client.collection("FileCategory").document(file_category_id).set({
        "id": file_category_id,
        "name": category_name,
        "loinc_code": loinc_code,
        "loinc_name": loinc_name,
        "parent_category_id": parent_category_id,
        "source_type": source_type,
        "confidence_score": confidence_score,
        "model_info": model_info,
        "model_category": model_category
    })

    return file_category_id


def process_file(file_data, user_id):
    try:
        # Deduplication check: Look for an existing record with the same md5Hash.
        duplicate = session.query(UploadedFiles).filter_by(hash_signature=file_data['md5Hash']).first()
        if duplicate:
            logger.info(f"Duplicate file detected: {file_data['name']} (hash: {file_data['md5Hash']}). Skipping processing.")
            return True  # Return True to indicate success and acknowledge the message

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

        # Vertex AI processing
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

        # Create a FileCategory record (and store it in SQL and Firestore)
        file_category_id = get_file_category(file_data)

        # Create SQL record for UploadedFiles
        uploaded_file = UploadedFiles(
            id=str(uuid.uuid4()),
            gcp_file_name=file_data['name'],
            original_filename=file_data['name'],
            uploaded_at=datetime.fromisoformat(file_data['timeCreated'][:-1]),
            document_type=file_category_id,
            patient_id=None,  # Replace with actual patient ID if available
            uploaded_by=user_id,
            deleted_by=None,
            deleted_at=None,
            version_number=1,
            last_accessed_at=None,
            last_accessed_by=None,
            document_date=None,
            status="processed",
            hash_signature=file_data['md5Hash'],
            last_modified_date=datetime.fromisoformat(file_data['updated'][:-1]),
            last_modified_by=user_id,
            mongo_history_id=None,
            verified_by=None,
            summary_verified_by=None,
            content=content_result
        )
        session.add(uploaded_file)
        session.commit()

        # Firestore write for UploadedFiles
        firestore_client.collection("UploadedFiles").document(uploaded_file.id).set({
            "id": uploaded_file.id,
            "gcp_file_name": uploaded_file.gcp_file_name,
            "original_filename": uploaded_file.original_filename,
            "uploaded_at": uploaded_file.uploaded_at,
            "document_type": uploaded_file.document_type,
            "patient_id": uploaded_file.patient_id,
            "uploaded_by": uploaded_file.uploaded_by,
            "deleted_by": uploaded_file.deleted_by,
            "deleted_at": uploaded_file.deleted_at,
            "version_number": uploaded_file.version_number,
            "last_accessed_at": uploaded_file.last_accessed_at,
            "last_accessed_by": uploaded_file.last_accessed_by,
            "document_date": uploaded_file.document_date,
            "status": uploaded_file.status,
            "hash_signature": uploaded_file.hash_signature,
            "last_modified_date": uploaded_file.last_modified_date,
            "last_modified_by": uploaded_file.last_modified_by,
            "mongo_history_id": uploaded_file.mongo_history_id,
            "verified_by": uploaded_file.verified_by,
            "summary_verified_by": uploaded_file.summary_verified_by,
            "content": uploaded_file.content
        })

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

        # Decode the base64-encoded data from the Pub/Sub envelope
        data_str = base64.b64decode(envelope['message']['data']).decode('utf-8')
        data = json.loads(data_str)
        file_data = {
            "bucket": data.get('bucket'),
            "name": data.get('name'),
            "timeCreated": data.get('timeCreated'),
            "updated": data.get('updated'),
            "md5Hash": data.get('md5Hash')
        }

        if not all(file_data.values()):
            logger.error("Missing bucket or name in message payload")
            return "Bad Request", 400

        user_id = "some-user-id"  # Replace with actual user ID extraction logic
        ensure_collections_exist()
        success = process_file(file_data, user_id)

        # Log audit
        log_audit(
            action="POST",
            endpoint="/",
            status_code=200 if success else 500,
            request_data=file_data,
            response_data={"success": success},
            user_id=user_id
        )

        return ("OK", 200) if success else ("Processing Failed", 500)

    except Exception as e:
        logger.critical(f"Fatal error in request handler: {str(e)}", exc_info=True)
        return "Internal Server Error", 500

# For local development; in Cloud Run with Gunicorn, this block is ignored.
if __name__ == "__main__":
    pass

