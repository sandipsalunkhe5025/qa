
#With PDF TXT and ZIP file
import os
import io
import fitz  # PyMuPDF
import logging
import boto3
import zipfile
from flask import Flask, request, render_template, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
from haystack.telemetry import tutorial_running
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http, print_answers
from haystack.pipelines.standard_pipelines import TextIndexingPipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# AWS S3 Configuration
S3_BUCKET = 'pythonqatest'
s3_client = boto3.client('s3', region_name='ap-south-1')

tutorial_running(1)
logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# Initialize Document Store
document_store = InMemoryDocumentStore(use_bm25=True)
# Download and prepare initial data
doc_dir = "data/build_your_first_question_answering_system"
fetch_archive_from_http(
    url="https://pythonqatest.s3.ap-south-1.amazonaws.com/wiki_gameofthrones",
    output_dir=doc_dir)

files_to_index = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_content):
    pdf_document = fitz.open(stream=pdf_content, filetype='pdf')
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            prediction = pipe.run(
                query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}}
            )
            # Return JSON response for AJAX requests
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'prediction': prediction,
                    'file_content': session.get('file_content', '')
                })
    return render_template('index2.html', prediction=None, file_content=session.get('file_content', ''))


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            s3_client.upload_file(file_path, S3_BUCKET, filename)
            app.logger.info(f"File '{filename}' uploaded successfully to S3 bucket '{S3_BUCKET}'.")
            indexing_pipeline.run_batch(file_paths=[file_path])
            app.logger.info(f"File '{filename}' indexed successfully.")
            if filename.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    content = extract_text_from_pdf(f.read())
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            session['file_content'] = content
            os.remove(file_path)  # Remove local file after successful upload and indexing
            return jsonify(success=True, file_content=content)
        except boto3.exceptions.S3UploadFailedError as e:
            app.logger.error(f"S3 upload failed for file '{filename}': {str(e)}")
            return jsonify(success=False, message=f"S3 upload failed for file '{filename}': {str(e)}"), 500
        except Exception as e:
            app.logger.error(f"Error uploading and indexing file '{filename}': {str(e)}")
            return jsonify(success=False, message=f"Error uploading and indexing file '{filename}': {str(e)}"), 500
    return jsonify(success=False, message='File not allowed'), 400

@app.route('/list-files', methods=['GET'])
def list_files():
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET)
        files = [{'key': obj['Key']} for obj in response.get('Contents', [])]
        return jsonify(files)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500
@app.route('/file-content', methods=['GET'])
def file_content():
    key = request.args.get('key')
    app.logger.info(f"Fetching content for file: {key}")
    if not key:
        app.logger.error('File key is required')
        return jsonify(success=False, message='File key is required'), 400
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        file_content = obj['Body'].read()
        if key.endswith('.pdf'):
            content = extract_text_from_pdf(file_content)
            pages = [content]  # Treat the entire PDF content as one page for simplicity
        elif key.endswith('.zip'):
            app.logger.info('File is a ZIP archive, extracting content')
            with zipfile.ZipFile(io.BytesIO(file_content)) as the_zip:
                file_names = the_zip.namelist()
                app.logger.info(f"Files in the ZIP archive: {file_names}")
                if not file_names:
                    app.logger.error('ZIP file is empty')
                    return jsonify(success=False, message='ZIP file is empty'), 400
                pages = []
                for file_name in file_names:
                    with the_zip.open(file_name) as extracted_file:
                        content = extracted_file.read().decode('utf-8')
                        pages.append(content)
        else:
            app.logger.info('File is a plain text file')
            pages = [file_content.decode('utf-8')]
        return jsonify(success=True, content=pages)

    except Exception as e:
        app.logger.error(f"Error reading file '{key}': {str(e)}")
        return jsonify(success=False, message=str(e)), 500

@app.route('/uploads/<path:filename>')
def uploads(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='127.0.0.1', port=8000, debug=True)







# import os
# import io
# import fitz
# import logging
# import boto3
# import zipfile
# from flask import Flask, request, render_template, jsonify, session, send_from_directory
# from werkzeug.utils import secure_filename
# from haystack.telemetry import tutorial_running
# from haystack.nodes import BM25Retriever, FARMReader
# from haystack.pipelines import ExtractiveQAPipeline
# from haystack.document_stores import InMemoryDocumentStore
# from haystack.utils import fetch_archive_from_http, print_answers
# from haystack.pipelines.standard_pipelines import TextIndexingPipeline
#
# app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # Replace with your actual secret key
#
# # Configuration for file uploads
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'zip'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# # AWS S3 Configuration
# S3_BUCKET = 'pythonqatest'
# s3_client = boto3.client('s3', region_name='ap-south-1')
#
# tutorial_running(1)
# logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
# logging.getLogger("haystack").setLevel(logging.INFO)
#
# # Initialize Document Store
# document_store = InMemoryDocumentStore(use_bm25=True)
# # Download and prepare initial data
# doc_dir = "data/build_your_first_question_answering_system"
# fetch_archive_from_http(
#     url="https://pythonqatest.s3.ap-south-1.amazonaws.com/wiki_gameofthrones",
#     output_dir=doc_dir)
#
# files_to_index = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir)]
# indexing_pipeline = TextIndexingPipeline(document_store)
# indexing_pipeline.run_batch(file_paths=files_to_index)
# retriever = BM25Retriever(document_store=document_store)
# reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
# pipe = ExtractiveQAPipeline(reader, retriever)
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
# def extract_text_from_pdf(pdf_content):
#     pdf_document = fitz.open(stream=pdf_content, filetype='pdf')
#     text = ""
#     for page_num in range(pdf_document.page_count):
#         page = pdf_document.load_page(page_num)
#         text += page.get_text("text")
#     return text
#
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         query = request.form.get('query')
#         if query:
#             prediction = pipe.run(
#                 query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}}
#             )
#             # Return JSON response for AJAX requests
#             if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#                 return jsonify({
#                     'prediction': prediction,
#                     'file_content': session.get('file_content', '')
#                 })
#     return render_template('index2.html', prediction=None, file_content=session.get('file_content', ''))
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     file = request.files.get('file')
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#         try:
#             s3_client.upload_file(file_path, S3_BUCKET, filename)
#             app.logger.info(f"File '{filename}' uploaded successfully to S3 bucket '{S3_BUCKET}'.")
#             # Index the uploaded file
#             indexing_pipeline.run_batch(file_paths=[file_path])
#             app.logger.info(f"File '{filename}' indexed successfully.")
#             if filename.endswith('.pdf'):
#                 with open(file_path, 'rb') as f:
#                     content = extract_text_from_pdf(f.read())
#             else:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     content = f.read()
#             session['file_content'] = content
#             os.remove(file_path)  # Remove local file after successful upload and indexing
#             return jsonify(success=True, file_content=content)
#         except boto3.exceptions.S3UploadFailedError as e:
#             app.logger.error(f"S3 upload failed for file '{filename}': {str(e)}")
#             return jsonify(success=False, message=f"S3 upload failed for file '{filename}': {str(e)}"), 500
#         except Exception as e:
#             app.logger.error(f"Error uploading and indexing file '{filename}': {str(e)}")
#             return jsonify(success=False, message=f"Error uploading and indexing file '{filename}': {str(e)}"), 500
#     return jsonify(success=False, message='File not allowed'), 400
#
# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         response = s3_client.list_objects_v2(Bucket=S3_BUCKET)
#         files = [{'key': obj['Key']} for obj in response.get('Contents', [])]
#         return jsonify(files)
#     except Exception as e:
#         return jsonify(success=False, message=str(e)), 500
# @app.route('/file-content', methods=['GET'])
# def file_content():
#
#     key = request.args.get('key')
#     app.logger.info(f"Fetching content for file: {key}")
#     if not key:
#         app.logger.error('File key is required')
#         return jsonify(success=False, message='File key is required'), 400
#     try:
#         obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
#         file_content = obj['Body'].read()
#         if key.endswith('.pdf'):
#             content = extract_text_from_pdf(file_content)
#             pages = [content]  # Treat the entire PDF content as one page for simplicity
#         elif key.endswith('.zip'):
#
#             app.logger.info('File is a ZIP archive, extracting content')
#             with zipfile.ZipFile(io.BytesIO(file_content)) as the_zip:
#                 file_names = the_zip.namelist()
#                 app.logger.info(f"Files in the ZIP archive: {file_names}")
#                 if not file_names:
#                     app.logger.error('ZIP file is empty')
#                     return jsonify(success=False, message='ZIP file is empty'), 400
#
#                 pages = []
#                 for file_name in file_names:
#                     with the_zip.open(file_name) as extracted_file:
#                         content = extracted_file.read().decode('utf-8')
#                         pages.append(content)
#
#         else:
#             app.logger.info('File is a plain text file')
#             pages = [file_content.decode('utf-8')]
#         return jsonify(success=True, content=pages)
#
#     except Exception as e:
#         app.logger.error(f"Error reading file '{key}': {str(e)}")
#         return jsonify(success=False, message=str(e)), 500
#
# @app.route('/uploads/<path:filename>')
# def uploads(filename):
#     return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)
#
# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run(host='127.0.0.1', port=8000, debug=True)