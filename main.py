from flask import Flask, request, render_template, redirect, url_for, flash, send_file, Response
from google.cloud import storage
import google.generativeai as genai
import os
import json
import io
from PIL import Image


app = Flask(__name__)
app.secret_key = 'your-secret-key'

# Configuration

BUCKET_NAME = "photos-app"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize clients
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# Configure Gemini AI
try:
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
except KeyError:
    print("Please set the GEMINI_API_KEY environment variable")
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.4,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }
)

PROMPT = "Give me a simple title and description in json format for this image."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_with_gemini(file):
    """Process image using Gemini AI and return title and description"""
    
    try:
        response = model.generate_content(
            [Image.open(file), PROMPT]
        )
        
        if response and response.text:  # Check if a valid response was received
            # Try to parse the JSON response
            try:
                # The response should be JSON format with title and description
                json_data = json.loads(response.text)
                return json_data
            except json.JSONDecodeError:
                # If not valid JSON, return as is
                return {"title": "Untitled", "description": response.text}
        else:  # Gemini returned an empty response
            return {"title": "Untitled", "description": "No description available"}
    except Exception as e:
        print(f"Error processing image with Gemini: {str(e)}")
        return {"title": "Error", "description": f"Error processing image: {str(e)}"}

@app.route('/')
def index():
    image_data = []  # Will hold image filename, title and description data
    
    blobs = bucket.list_blobs()
    for blob in blobs:
        if blob.name.endswith(('jpeg', 'jpg', 'png', 'gif')):
            # Get image description
            description_filename = os.path.splitext(blob.name)[0] + '_description.json'
            try:
                image_data.append({
                    'filename': blob.name,
                    'title': description_data.get('title', 'Untitled'),
                    'description': description_data.get('description', 'No description available')
                })
            except Exception as e:
                print(f"Error fetching description for {blob.name}: {str(e)}")
                image_data.append({
                    'filename': blob.name,
                    'title': 'Untitled',
                    'description': 'No description available'
                })

    return render_template('index.html', image_data=image_data)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            filename = file.filename

            # Save file locally temporarily
            local_path = os.path.join('/tmp', filename)
            file.save(local_path)

            # Upload to Google Cloud Storage
            blob = bucket.blob(filename)
            blob.upload_from_filename(local_path)

            # Process with Gemini AI
            description_data = process_image_with_gemini(file)

            if description_data:
                print(f"Gemini Description Data: {description_data}")
                
                # Correctly get the base name *without* the extension
                base_name = os.path.splitext(filename)[0]
                # Construct the JSON filename with the base_name
                json_filename = f"{base_name}_description.json"
                print(f"Uploading Description JSON: {json_filename}")  # Debug print

                # Upload JSON to the same bucket
                description_blob = bucket.blob(json_filename)
                description_blob.upload_from_string(
                    json.dumps(description_data, indent=2),
                    content_type='application/json'
                )

            # Clean up local file
            os.remove(local_path)

            flash('File successfully uploaded and processed')
            return redirect(url_for('index'))

        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))

    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/file/<filename>')
def get_fil(filename):
    f_name = filename.split('.')[0] + '_description.json'
    description_blob = bucket.blob(f_name)
    description_data = None
    with description_blob.open("r") as file_object:
        description_data = file_object.read()
    html = f"""

<body>
<img src = "/image/{filename}">
<p>
{description_data}
</p>
</body>
    """
    return html
    

@app.route('/image/<filename>')
def get_file(filename):
    try:
        blob = bucket.blob(filename)
        file_data = blob.download_as_bytes()

        response = send_file(
            io.BytesIO(file_data),
            mimetype=blob.content_type,
            as_attachment=True,
            download_name=filename
        )
        return response
    except Exception as e:
        flash(f'Error downloading file: {str(e)}')
        return redirect(url_for('index'))


@app.route('/description/<filename>')
def get_description(filename):
    try:
        # Correctly get the base name *without* the extension
        base_name = os.path.splitext(filename)[0]
        # Construct the JSON filename with the base_name
        json_filename = f"{base_name}_description.json"

        blob = bucket.blob(json_filename)
        description_data = json.loads(blob.download_as_string())
        return description_data
    except Exception as e:
        return f"Error retrieving description: {str(e)}"


if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
