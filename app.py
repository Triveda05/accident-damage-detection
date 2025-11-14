from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import Counter
import os
import json
import shutil # Added for safer file operations

# --- CONFIGURATION ---
# 1. Path to your model weights (You MUST update this path on your local system)
# NOTE: This is a placeholder. Update to your actual path like "C:/path/to/best.pt"
MODEL_PATH = r"C:\Users\rayal\OneDrive\Desktop\accident-damage-detection\models\model weights\best.pt" 

# 2. Path to your pricing JSON file
PRICE_JSON_PATH = "car_parts_prices.json" 

# 3. Define UPLOAD folder relative to the app.py location
UPLOAD_FOLDER = 'static/uploads'

# Corrected Flask app initialization with double underscores
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB file size

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- GLOBAL DATA AND MODEL LOADING ---
CAR_PRICES = {}
AVAILABLE_BRANDS = []
MODEL = None

def load_data():
    """Loads car parts prices from the JSON file."""
    global CAR_PRICES, AVAILABLE_BRANDS
    try:
        # NOTE: This assumes car_parts_prices.json is in the same directory as app.py
        with open(PRICE_JSON_PATH, 'r') as f:
            CAR_PRICES = json.load(f)
            AVAILABLE_BRANDS = sorted(list(CAR_PRICES.keys()))
        print(f"✓ Pricing data loaded successfully from: {PRICE_JSON_PATH}")
    except FileNotFoundError:
        print(f"❌ ERROR: Price file not found at '{PRICE_JSON_PATH}'. Pricing will fail.")
    except json.JSONDecodeError:
        print(f"❌ ERROR: Could not parse JSON data in '{PRICE_JSON_PATH}'.")

load_data()

# Load YOLO model
try:
    # Use global variable for the model
    MODEL = YOLO(MODEL_PATH) 
    print(f"✓ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR: Could not load YOLO model from {MODEL_PATH}. Using 'yolov8n.pt' as fallback. Error: {e}")
    # Fallback model is used if the specified model path fails
    MODEL = YOLO('yolov8n.pt') 

# --- UTILITIES ---

# Mapping of class IDs to part names (MUST match your training data)
def get_part_name_from_id(class_id):
    # Adjust this list if your model uses different class names or order
    class_names = ['Bonnet', 'Bumper', 'Dickey', 'Door', 'Fender', 'Light', 'Windshield']
    try:
        # Convert class_id to integer for list indexing
        return class_names[int(class_id)]
    except (IndexError, TypeError):
        return None

def calculate_prices(car_brand, car_model, class_counts):
    """Calculates the estimated cost based on detected parts and user-selected model/brand."""
    prices = {}
    
    # Check if brand and model exist in the loaded data
    if car_brand not in CAR_PRICES or car_model not in CAR_PRICES[car_brand]:
        print(f"WARNING: Price data not found for {car_brand} - {car_model}. Cannot calculate prices.")
        return {}

    model_prices = CAR_PRICES[car_brand][car_model]
    
    for class_id, count in class_counts.items():
        part_name = get_part_name_from_id(class_id)
        
        # Check if the detected part has a price defined for the selected model
        if part_name and part_name in model_prices:
            price_per_part = model_prices[part_name]
            total_price = price_per_part * count
            prices[part_name] = {
                'count': count, 
                'price': price_per_part, 
                'total': total_price
            }
        elif part_name:
            print(f"WARNING: Part '{part_name}' detected but no price found for {car_model}.")
            
    return prices

# --- ROUTES ---

@app.route('/')
def home():
    # Direct users to the main prediction page immediately
    return redirect(url_for('predict_damage'))

@app.route('/predict', methods=['GET', 'POST'])
def predict_damage():
    # Pass this data for both GET and POST requests that render predict.html
    template_data = {'brands': AVAILABLE_BRANDS, 'prices': CAR_PRICES}
    original_image_path = None
    detected_image_path = None
    
    if request.method == 'POST':
        # Check if the model was loaded successfully before proceeding with prediction
        if not MODEL:
            template_data['error'] = "The YOLO model failed to load. Cannot perform damage analysis."
            return render_template('predict.html', **template_data)

        file = request.files.get('image')
        car_brand = request.form.get('car_brand')
        car_model = request.form.get('car_model')

        if not all([file, car_brand, car_model]):
            template_data['error'] = 'Please select a car and upload an image.'
            return render_template('predict.html', **template_data)

        if file.filename == '':
            template_data['error'] = 'Please upload an image.'
            return render_template('predict.html', **template_data)

        filename = secure_filename(file.filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            template_data['error'] = 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'
            return render_template('predict.html', **template_data)
        
        # Prepare file paths
        unique_id = os.urandom(8).hex()
        original_filename = f"original_{unique_id}_{filename}"
        detected_filename = f"detected_{unique_id}_{filename}"
        
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        detected_image_path = os.path.join(app.config['UPLOAD_FOLDER'], detected_filename)
        
        try:
            # 1. Save the uploaded image
            file.save(original_image_path)
            
            # 2. Make predictions using YOLO
            # Setting 'save=True' here will save the detected image to the default YOLO run folder, 
            # which isn't the UPLOAD_FOLDER. We'll manually save the result below.
            results = MODEL(original_image_path) 
            
            result = results[0] if results else None
            
            # Check for results before accessing boxes
            if result and result.boxes:
                detected_objects = result.boxes
                class_ids = [box.cls.item() for box in detected_objects]
                class_counts = Counter(class_ids)

                # 3. Save the image with detections to the correct path
                result.save(filename=detected_image_path)
                
                # 4. Calculate estimation using JSON data
                part_prices = calculate_prices(car_brand, car_model, class_counts)
            else:
                # Handle case where no detections are made
                class_counts = Counter()
                part_prices = {}
                # If no damage is detected, just use the original image for both views
                # We copy the original file to the 'detected' filename path 
                # to ensure both URLs are valid and point to an image.
                shutil.copyfile(original_image_path, detected_image_path)
                
            return render_template(
                'estimate.html', 
                original_image=url_for('static', filename=f'uploads/{original_filename}'), 
                detected_image=url_for('static', filename=f'uploads/{detected_filename}'), 
                part_prices=part_prices,
                car_info={'brand': car_brand, 'model': car_model}
            )
            
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            template_data['error'] = f"An unexpected error occurred during analysis: {e}"
            return render_template('predict.html', **template_data)
        
        finally:
            # Use finally to ensure cleanup happens whether try block succeeded or failed
            # We only remove the original file as the 'detected' file is needed for the estimate page.
            # However, for a complete cleanup approach (if files aren't meant to persist), 
            # you'd need a separate mechanism or remove both here. 
            # For this MVP, we leave the files in `static/uploads` to be served.
            # If you want to remove all files after estimate, you'd move this cleanup to a post-view action.
            # Keeping cleanup as per original intent, but making it more robust:
            pass # Files remain in /static/uploads for display
            
    # GET request handler (Renders the upload form)
    return render_template('predict.html', **template_data)

# Corrected main execution block with double underscores
if __name__ == '__main__':
    # Set debug=False for production
    app.run(debug=True, port=8000) # Changed port from default 5000 to 8000