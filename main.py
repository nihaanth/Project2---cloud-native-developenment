from flask import Flask, request, render_template, redirect, url_for, flash, send_file, Response
from google.cloud import storage
import google.generativeai as genai
import os
import json
import io
from PIL import Image
import logging # Import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


app = Flask(__name__)
# It's recommended to load the secret key from environment variables or a config file
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a-default-fallback-secret-key')

# --- Configuration ---
# It's better to get the bucket name from an environment variable
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "photos-app") # Use env var or default
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Initialize Clients ---
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    logging.info(f"Successfully connected to GCS bucket: {BUCKET_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize Google Cloud Storage client: {e}")
    # Depending on the application, you might want to exit or handle this differently
    storage_client = None
    bucket = None

# --- Configure Gemini AI ---
try:
    # It's crucial to get the API key from environment variables for security
    gemini_api_key = os.environ['GEMINI_API_KEY']
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 4096, # Reduced for title/desc
            "response_mime_type": "application/json", # Request JSON directly
        }
    )
    logging.info("Successfully configured Gemini AI model.")
except KeyError:
    logging.error("GEMINI_API_KEY environment variable not set.")
    model = None # Indicate that the model is not available
except Exception as e:
    logging.error(f"Failed to configure Gemini AI: {e}")
    model = None

# Prompt for Gemini, asking specifically for JSON output
PROMPT = """
Analyze this image and provide a concise title and a short description.
Return the output strictly in the following JSON format:
{
  "title": "Your Image Title",
  "description": "Your image description."
}
Do not include any other text or formatting outside the JSON structure.
"""

def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_with_gemini(image_bytes):
    """
    Process image bytes using Gemini AI and return title and description.
    Expects Gemini to return JSON directly based on the prompt and config.
    """
    if not model:
        logging.warning("Gemini model not initialized. Cannot process image.")
        return {"title": "Error", "description": "AI model not available."}

    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Send image and prompt to Gemini
        response = model.generate_content([img, PROMPT])

        logging.info(f"Gemini Raw Response: {response.text}") # Log the raw response

        # Gemini is configured to return JSON, so parse it directly
        if response and response.text:
            try:
                # Clean potential markdown backticks if they still appear
                cleaned_text = response.text.strip().strip('```json').strip('```').strip()
                json_data = json.loads(cleaned_text)
                # Basic validation
                if isinstance(json_data, dict) and 'title' in json_data and 'description' in json_data:
                     logging.info(f"Successfully parsed Gemini JSON: {json_data}")
                     return json_data
                else:
                    logging.warning(f"Gemini response was not the expected JSON format: {cleaned_text}")
                    # Fallback if JSON structure is wrong
                    return {"title": "Processing Error", "description": "Received invalid format from AI."}
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode Gemini JSON response: {e}. Response text: {response.text}")
                # Fallback if JSON parsing fails
                return {"title": "Processing Error", "description": f"Could not parse AI response: {response.text[:100]}..."} # Show partial response
            except Exception as e_inner: # Catch other potential errors during parsing/validation
                 logging.error(f"Error processing Gemini response content: {e_inner}. Response text: {response.text}")
                 return {"title": "Processing Error", "description": f"Error processing AI response: {str(e_inner)}"}
        else:
            logging.warning("Gemini returned an empty or invalid response.")
            return {"title": "Untitled", "description": "No description generated."}

    except Exception as e:
        logging.error(f"Error processing image with Gemini: {e}", exc_info=True) # Log stack trace
        return {"title": "Error", "description": f"Error during AI processing: {str(e)}"}


@app.route('/')
def index():
    """Displays the main gallery page with image links."""
    if not bucket:
        flash("Storage bucket not configured. Cannot display images.", "error")
        return render_template('index.html', image_data=[])

    image_data = []
    try:
        blobs = list(bucket.list_blobs()) # Get all blobs once
        logging.info(f"Found {len(blobs)} blobs in the bucket.")

        image_files = [b for b in blobs if b.name.lower().endswith(tuple(ALLOWED_EXTENSIONS))]
        logging.info(f"Found {len(image_files)} potential image files.")

        # Create a dictionary of description blobs for quick lookup
        description_blobs = {b.name: b for b in blobs if b.name.lower().endswith('_description.json')}
        logging.info(f"Found {len(description_blobs)} potential description files.")


        for img_blob in image_files:
            base_name = os.path.splitext(img_blob.name)[0]
            desc_filename = f"{base_name}_description.json"
            title = 'Untitled'
            description = 'No description available'

            # Check if the corresponding description JSON exists
            if desc_filename in description_blobs:
                desc_blob = description_blobs[desc_filename]
                try:
                    # Download and parse the description JSON
                    desc_content = desc_blob.download_as_string()
                    desc_json = json.loads(desc_content)
                    title = desc_json.get('title', 'Untitled') # Use .get for safety
                    description = desc_json.get('description', 'No description available')
                    logging.debug(f"Successfully loaded description for {img_blob.name}")
                except json.JSONDecodeError:
                    logging.warning(f"Could not decode JSON for {desc_filename}. Content: {desc_content[:100]}...")
                    description = 'Error loading description (invalid format).'
                except Exception as e:
                    logging.error(f"Error fetching/parsing description for {img_blob.name}: {e}")
                    description = 'Error loading description.'
            else:
                 logging.warning(f"Description file {desc_filename} not found for image {img_blob.name}")


            image_data.append({
                'filename': img_blob.name,
                'title': title,
                # 'description': description # No longer needed for the index page link
            })

    except Exception as e:
        logging.error(f"Error listing blobs or processing index data: {e}", exc_info=True)
        flash(f"Error retrieving image list: {str(e)}", "error")

    return render_template('index.html', image_data=image_data)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, processing with Gemini, and storing results."""
    if not bucket or not storage_client:
        flash("Storage or AI service not configured. Upload disabled.", "error")
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file part in the request.', 'warning')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected for upload.', 'warning')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = file.filename # Keep original filename
        # It's generally safer to generate a unique filename to avoid collisions
        # unique_filename = f"{uuid.uuid4()}{os.path.splitext(filename)[1]}"
        # For simplicity, we'll stick with the original name here.

        try:
            # Read file content into memory for processing and upload
            file_bytes = file.read()
            file.seek(0) # Reset stream position if needed elsewhere, though not strictly necessary here

            # --- Upload Image to GCS ---
            blob = bucket.blob(filename)
            # Use upload_from_string with content type for better handling
            blob.upload_from_string(file_bytes, content_type=file.mimetype)
            logging.info(f"Successfully uploaded image '{filename}' to GCS.")

            # --- Process with Gemini AI ---
            # Pass the image bytes directly
            description_data = process_image_with_gemini(file_bytes)
            logging.info(f"Received description data for '{filename}': {description_data}")


            # --- Upload Description JSON to GCS ---
            if description_data and isinstance(description_data, dict):
                base_name = os.path.splitext(filename)[0]
                json_filename = f"{base_name}_description.json"
                description_blob = bucket.blob(json_filename)
                description_blob.upload_from_string(
                    json.dumps(description_data, indent=2),
                    content_type='application/json'
                )
                logging.info(f"Successfully uploaded description JSON '{json_filename}' to GCS.")
                flash(f"File '{filename}' uploaded and processed successfully!", 'success')
            else:
                 # Handle cases where Gemini processing failed but image was uploaded
                 flash(f"File '{filename}' uploaded, but AI processing failed. Description might be missing or incorrect.", 'warning')


            return redirect(url_for('index'))

        except Exception as e:
            logging.error(f"Error during upload/processing for file '{filename}': {e}", exc_info=True)
            flash(f'Error processing file {filename}: {str(e)}', 'error')
            return redirect(url_for('index'))

    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg, gif', 'warning')
        return redirect(url_for('index'))

# Renamed route for clarity
@app.route('/view/<filename>')
def view_image_details(filename):
    """Displays a single image with its title and description."""
    if not bucket:
        return "Storage bucket not configured.", 500

    title = "Error"
    description = "Could not load description."
    image_url = url_for('get_image_file', filename=filename) # Use a dedicated route for image serving

    # Construct the expected description filename
    base_name = os.path.splitext(filename)[0]
    json_filename = f"{base_name}_description.json"
    description_blob = bucket.blob(json_filename)

    try:
        if description_blob.exists():
            # Download and parse the description JSON
            description_data_bytes = description_blob.download_as_string()
            description_data = json.loads(description_data_bytes)
            title = description_data.get('title', 'Untitled')
            description = description_data.get('description', 'No description available.')
            logging.info(f"Loaded title/description for {filename}")
        else:
            logging.warning(f"Description file {json_filename} not found for image {filename}")
            title = "Untitled"
            description = "No description file found."

    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON for {json_filename}")
        description = "Error: Description file has invalid format."
    except Exception as e:
        logging.error(f"Error retrieving description for {filename}: {e}", exc_info=True)
        description = f"Error retrieving description: {str(e)}"


    # Render using a template for better structure and maintainability
    return render_template('image_detail.html',
                           title=title,
                           description=description,
                           image_url=image_url,
                           filename=filename)


# Renamed route for clarity and purpose
@app.route('/image/<filename>')
def get_image_file(filename):
    """Serves the image file directly from GCS."""
    if not bucket:
        return "Storage bucket not configured.", 500

    try:
        blob = bucket.blob(filename)
        if not blob.exists():
             logging.warning(f"Image file not found in GCS: {filename}")
             return "Image not found", 404

        file_data = blob.download_as_bytes()

        # Use Response for more control over headers if needed, or send_file
        return send_file(
            io.BytesIO(file_data),
            mimetype=blob.content_type or 'application/octet-stream', # Provide default mimetype
            # as_attachment=False, # Display inline by default
            download_name=filename # Suggest filename if user saves
        )
    except Exception as e:
        logging.error(f"Error serving image file {filename}: {e}", exc_info=True)
        # Don't flash here as it might redirect unexpectedly
        return f"Error downloading file: {str(e)}", 500


# This route is likely redundant now if get_fil is replaced by view_image_details
# and the description is loaded within that route.
# If you still need a dedicated API endpoint for JUST the description JSON:
@app.route('/api/description/<filename>')
def get_description_json(filename):
     """Returns the raw description JSON data."""
     if not bucket:
        return {"error": "Storage bucket not configured."}, 500

     try:
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}_description.json"
        blob = bucket.blob(json_filename)

        if not blob.exists():
             return {"error": "Description not found."}, 404

        description_data = json.loads(blob.download_as_string())
        return description_data # Flask automatically jsonify's dicts

     except json.JSONDecodeError:
         return {"error": "Invalid description format."}, 500
     except Exception as e:
        logging.error(f"Error retrieving description JSON for {filename}: {e}", exc_info=True)
        return {"error": f"Error retrieving description: {str(e)}"}, 500


if __name__ == '__main__':
    # Use environment variable for port, default to 8080
    port = int(os.environ.get('PORT', 8080))
    # Debug should ideally be False in production, controlled by an env var
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode) # Listen on all interfaces
