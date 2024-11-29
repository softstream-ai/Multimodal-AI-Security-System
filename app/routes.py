########### ----------------- Imports --------------------------------------------- #########
from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
from cryptography.fernet import Fernet, InvalidToken  
from werkzeug.utils import secure_filename
from flask import jsonify, json
from ultralytics import YOLO
from app.models import User
from app import db
import numpy as np
import smtplib
import shutil
import base64
import torch
import cv2
import os 


# -------------------------------------------------------------------------------------------- #

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

main = Blueprint('main', __name__)
detector = MTCNN()
face_model = InceptionResnetV1(pretrained='vggface2').eval()
yolo_model = YOLO('best.pt')
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
encryption_key = "m4ghY5lXfpBC8EhdFkS4Snv-27v-5JmTHKrKZQXMJJY="
cipher_suite = Fernet(encryption_key)

# -------------------------------------- <<<<<<<<<< HELPER FUNCTIONS >>>>>>> ----------------------------------------- #

# Base directory structure for logged-in user
def get_base_folder(username):
    return os.path.join("image_data", f"{username}_family_img_data")

# Helper function to preprocess images for embeddings
def preprocess_image(face_img):
    face_img_resized = cv2.resize(face_img, (160, 160)) / 255.0
    face_img_tensor = torch.tensor(face_img_resized).permute(2, 0, 1).float().unsqueeze(0)
    for t, m, s in zip(face_img_tensor, mean, std):
        t.sub_(m).div_(s)
    return face_img_tensor

# Generate embedding for a face image
def generate_embedding(face_img):
    face_img_tensor = preprocess_image(face_img)
    with torch.no_grad():
        embedding = face_model(face_img_tensor)
    return embedding

# Create annotations for each family member's images
def annotate_faces_for_user(username):
    base_folder = get_base_folder(username)
    
    for person_name in os.listdir(base_folder):
        person_folder = os.path.join(base_folder, person_name)
        
        if not os.path.isdir(person_folder):
            continue

        annotations_file = os.path.join(person_folder, "annotations.json")
        
        # Skip if annotations already exist
        if os.path.exists(annotations_file):
            print(f"Annotations already exist for {person_name}, skipping.")
            continue

        annotations = []
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error: Failed to load image {img_path}. Skipping.")
                continue

            # Detect faces and create bounding boxes
            boxes, _ = detector.detect(img)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    annotations.append({
                        "image": img_file,
                        "bounding_box": [x1, y1, x2, y2],
                        "label": person_name
                    })
            else:
                print(f"No faces detected in {img_file} for {person_name}.")

        # Save annotations if detected
        if annotations:
            with open(annotations_file, "w") as f:
                json.dump(annotations, f)
            print(f"Annotations for {person_name} saved.")
        else:
            print(f"No annotations created for {person_name}. No faces detected in any images.")

# Create embeddings for each annotated face in family members' images
def save_embeddings_for_user(username):
    base_folder = get_base_folder(username)

    for person_name in os.listdir(base_folder):
        person_folder = os.path.join(base_folder, person_name)
        
        if not os.path.isdir(person_folder):
            continue

        annotations_file = os.path.join(person_folder, "annotations.json")
        embeddings_file = os.path.join(person_folder, "embeddings.json")

        # Skip if embeddings already exist or annotations file is missing
        if os.path.exists(embeddings_file):
            print(f"Embeddings already exist for {person_name}, skipping.")
            continue
        if not os.path.exists(annotations_file):
            print(f"Annotations for {person_name} not found, skipping.")
            continue

        # Load annotations for current family member
        with open(annotations_file, "r") as f:
            annotations = json.load(f)

        embeddings = {}

        for annotation in annotations:
            img_file = annotation["image"]
            bounding_box = annotation["bounding_box"]
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error: Failed to load image '{img_path}'. Skipping.")
                continue

            # Extract the face from the bounding box
            x1, y1, x2, y2 = bounding_box
            face_img = img[y1:y2, x1:x2]

            # Generate embedding and store it
            embedding = generate_embedding(face_img)
            embeddings[img_file] = embedding.detach().numpy().flatten().tolist()

        # Save embeddings to file
        with open(embeddings_file, "w") as f:
            json.dump(embeddings, f)

        print(f"Embeddings for {person_name} saved.")

# Call helper functions for the logged-in user's family data
def process_family_images(username):
    annotate_faces_for_user(username)
    save_embeddings_for_user(username)

# Check each family member's folder for existing annotations and embeddings will Generate them if missing.
def check_and_generate_annot_embedd_data(username):
    base_folder = get_base_folder(username)
    
    for person_name in os.listdir(base_folder):
        person_folder = os.path.join(base_folder, person_name)
        
        if not os.path.isdir(person_folder):
            continue

        annotations_file = os.path.join(person_folder, "annotations.json")
        embeddings_file = os.path.join(person_folder, "embeddings.json")

        # Only generate annotation or embedding if missing
        if not os.path.exists(annotations_file):
            print(f"Annotations not found for {person_name}. Generating...")
            annotate_faces_for_user(username)
            print(f"Annotation generated for : {person_name}")
        if not os.path.exists(embeddings_file):
            print(f"Embeddings not found for {person_name}. Generating...")
            save_embeddings_for_user(username)
            print(f"Embedding generated for : {person_name}")
            
        print(f"Checked and updated data for {person_name} in {username}'s family folder.")

# Funciton to load user specific family member embeddings 
def load_user_family_embeddings(username):
    base_folder = get_base_folder(username)
    family_embeddings = {}

    for person_name in os.listdir(base_folder):
        person_folder = os.path.join(base_folder, person_name)
        embeddings_file = os.path.join(person_folder, 'embeddings.json')
        
        if os.path.exists(embeddings_file):
            with open(embeddings_file, "r") as f:
                embeddings = json.load(f)
                family_embeddings[person_name] = {img_name: np.array(embedding) for img_name, embedding in embeddings.items()}
    
    return family_embeddings

# Email sending function
def send_email_alert(sender_email, password, receiver_email, subject, body):
    message = f"Subject: {subject}\n\n{body}"
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    print(f"Alert email sent to {receiver_email}")

# Align face
def align_face(img, box):
    x1, y1, x2, y2 = [int(b) for b in box]
    return img[y1:y2, x1:x2]

# draw label on image
def draw_label(img, label, box):
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Label above the box
    else:
        print("Invalid box coordinates:", box) 

# -------------------------------------- <<<<<<<<<< ROUTES >>>>>>> ----------------------------------------- #

# Home route
@main.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"An error occurred: {str(e)}"

# -------------------------------------------------------------------------------------------- #


# Signup route
@main.route('/signup', methods=['POST'])
def signup():
    try:
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            return jsonify({"success": False, "message": "Passwords do not match!"})
        
        # Assuming you have logic to check if the email is already taken
        if User.query.filter_by(email=email).first():
            return jsonify({"success": False, "message": "Email already in use!"})

        # Save new user to the database
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"success": True, "message": "Account created successfully!"})
    except Exception as e:
        print("Error during signup:", str(e))
        return jsonify({"success": False, "message": "An error occurred. Please try again."})


# -------------------------------------------------------------------------------------------- #

# Login route
@main.route('/login', methods=['POST'])
def login():
    try:
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            print("User logged in:", session['username'])
            return jsonify({
                "success": True,
                "username": user.username,  # Include the username in the response
                "message": "Login successful!"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Invalid email or password!"
            })
    except Exception as e:
        print("Error during login:", str(e))
        return jsonify({
            "success": False,
            "message": "An error occurred. Please try again."
        })



# -------------------------------------------------------------------------------------------- #

# Logout route
@main.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out!", "success")
    return redirect(url_for('main.home'))


# -------------------------------------------------------------------------------------------- #

## alert system route
@main.route('/configure-alert-system', methods=['POST'])
def configure_alert_system():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"success": False, "message": "User not logged in"})

        # Get email details from the request
        sender_email = request.json.get('sender_email')
        sender_password = request.json.get('sender_password')  # App-specific password
        receiver_email = request.json.get('receiver_email')

        if not sender_email or not receiver_email or not sender_password:
            return jsonify({"success": False, "message": "Missing required fields"})

        # Encrypt the app-specific password before storing it
        encrypted_password = base64.b64encode(cipher_suite.encrypt(sender_password.encode())).decode('utf-8')

        # Update the user's alert configuration
        user = User.query.get(user_id)
        user.alert_sender_email = sender_email
        user.alert_receiver_email = receiver_email
        user.encrypted_password = encrypted_password

        db.session.commit()

        return jsonify({"success": True, "message": f"Alert system configured successfully for user: {user.username}"})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    
    
# -------------------------------------------------------------------------------------------- #

## Data gathering UPload images route
@main.route('/upload-images', methods=['POST'])
def upload_images():
    try:
        user_id = session.get('user_id')
        username = session.get('username')

        if not user_id or not username:
            return jsonify({"success": False, 
                            "message": "User not logged in. Please log in to upload images."})

        person_name = request.form.get('person_name')
        uploaded_files = request.files.getlist('images')

        if not person_name:
            return jsonify({"success": False, 
                            "message": "Please provide the family member's name."})
        if len(uploaded_files) < 40:
            return jsonify({"success": False, 
                            "message": f"A minimum of 40 images is required. You have uploaded {len(uploaded_files)} image(s)."})

        base_dir = os.path.join('image_data', f"{username}_family_img_data")
        person_dir = os.path.join(base_dir, person_name)

        # Check if person_name directory already exists
        if os.path.exists(person_dir):
            return jsonify({"success": False, 
                            "message": f"Family member with name '{person_name}' already exists. Please use a different name."})

        os.makedirs(person_dir, exist_ok=True)

        for file in uploaded_files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(person_dir, filename))

        return jsonify({"success": True, 
                        "message": f"Successfully uploaded {len(uploaded_files)} images for family member: {person_name}."})

    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})


# -------------------------------------------------------------------------------------------- #

## Data gatehring camera feed route
@main.route('/capture-image', methods=['POST'])
def capture_image():
    try:
        user_id = session.get('user_id')
        username = session.get('username')

        if not user_id or not username:
            return jsonify({"success": False, 
                            "message": "User not logged in. Please log in to upload images."})
            
        person_name = request.form.get('person_name')
        if not person_name:
            return jsonify({"success": False, "message": "Please provide the family member's name."})

        # Check if the person's directory exists; if not, create it once
        base_dir = os.path.join('image_data', f"{username}_family_img_data")
        person_dir = os.path.join(base_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        # Process the uploaded image
        file = request.files.get('image')
        if not file:
            return jsonify({"success": False, "message": "No image provided."})

        # Convert image to OpenCV format
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Detect faces
        from mtcnn import MTCNN
        mtcnn = MTCNN()
        faces = mtcnn.detect_faces(img)
        if not faces:
            return jsonify({"success": False, "message": "No face detected. Ensure your face is visible to the camera."})

        # Save the image without creating directories repeatedly
        count = len(os.listdir(person_dir))
        img_path = os.path.join(person_dir, f"{person_name}_{count + 1}.jpg")
        cv2.imwrite(img_path, img)

        return jsonify({"success": True, "message": f"Captured image {count + 1} for {person_name}."})
    
    except Exception as e:
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})
    
    
# -------------------------------------------------------------------------------------------- #

embeddings_cache = {}

# Real-time drone and face detection API
@main.route('/detect-drones-and-faces', methods=['POST'])
def detect_drones_and_faces():
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    # Verify user and email configuration
    if not user or not user.alert_receiver_email:
        return jsonify({"success": False, "message": "User not logged in or alert configuration missing."})

    # Load embeddings once per session and cache them
    username = user.username
    if username not in embeddings_cache:
        print(f"Loading annotation and embedding for: {username} ...")
        check_and_generate_annot_embedd_data(username)
        embeddings_cache[username] = load_user_family_embeddings(username)
        print(f"Annotation and Embedding for user: {username} loaded successfully")
    family_embeddings = embeddings_cache[username]

    # Decrypt email password
    sender_email = user.alert_sender_email
    receiver_email = user.alert_receiver_email
    try:
        encrypted_password_bytes = base64.b64decode(user.encrypted_password)
        decrypted_password = cipher_suite.decrypt(encrypted_password_bytes).decode('utf-8')
    except (InvalidToken, Exception) as e:
        print(f"An error occurred during decryption: {e}")
        return jsonify({"success": False, "message": "Decryption failed. Please check encryption setup."})

    # Process incoming frame
    file = request.files.get('frame')
    if not file:
        return jsonify({"success": False, "message": "No frame provided."})

    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Resize frame for faster processing
    img_resized = cv2.resize(img, (320, 240))  # Reducing resolution for faster processing

    # Perform drone detection
    results = yolo_model(img_resized)
    drone_detected = False
    intruder_detected = False

    # Drone detection logic
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls in zip(boxes, confs, classes):
            if cls == 0 and conf > 0.4:  # Assuming class '0' is 'drone'
                drone_detected = True
                x1, y1, x2, y2 = [int(coord * 2) for coord in box]  # Scale coordinates back up to original image size
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"Drone: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Face detection and recognition
    boxes, _ = detector.detect(img)
    if boxes is not None:
        for box in boxes:
            if len(box) == 4:  # Ensure the box is in the expected format
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                face_img = align_face(img, (x1, y1, x2, y2))
                embedding = generate_embedding(face_img).detach().numpy().flatten()

                best_match = None
                best_similarity = 0.0
                threshold = 0.9

                for person_name, emb_dict in family_embeddings.items():
                    for img_name, person_emb in emb_dict.items():
                        similarity = cosine_similarity([embedding], [person_emb])[0][0]
                        if similarity > best_similarity and similarity > threshold:
                            best_match = person_name
                            best_similarity = similarity

                # Use `draw_label` to show the result on the image
                if best_match:
                    draw_label(img, f"Family: {best_match}", (x1, y1, x2, y2))  # Label family member's name
                else:
                    draw_label(img, "Intruder", (x1, y1, x2, y2))  # Label as "Intruder"
                    intruder_detected = True
            else:
                print("Skipping invalid bounding box:", box)

    # Send email alerts if threats are detected
    if drone_detected or intruder_detected:
        subject = "Security Alert: "
        body = ""
        if drone_detected and intruder_detected:
            subject += "Drone and Intruder Detected"
            body = "Both a drone and an intruder have been detected on your property."
        elif drone_detected:
            subject += "Drone Detected"
            body = "A drone has been detected on your property."
        elif intruder_detected:
            subject += "Intruder Detected"
            body = "An intruder has been detected on your property."

        send_email_alert(sender_email, decrypted_password, receiver_email, subject, body)

    # Encode the processed image as base64 to return it
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Message and image data for client display
    message = "Suspicious activity detected!" if drone_detected or intruder_detected else "No threats detected."
    return jsonify({"success": True, "message": message, "image": img_base64})

# -------------------------------------------------------------------------------------------- #

@main.route('/family_members_details', methods=['GET'])
def get_family_members():
    # Get the current logged-in username from session
    username = session.get('username')
    if not username:
        return jsonify({'error': 'User not logged in.'}), 401

    # Path to the current user's family image data directory
    family_dir = get_base_folder(username)

    # Check if the family data directory exists for the user
    if not os.path.exists(family_dir):
        return jsonify({'error': 'Family data not found.'}), 404

    family_members = []
    image_urls = []  # List to hold image URLs for saving

    # Static directory where images will be copied
    static_image_dir = os.path.join('static', 'family_images')

    # Ensure static image directory exists
    os.makedirs(static_image_dir, exist_ok=True)

    # Iterate through each family member's folder
    for member_name in os.listdir(family_dir):
        member_folder = os.path.join(family_dir, member_name)
        if os.path.isdir(member_folder):
            # Get the first image in the member's folder
            images = [img for img in os.listdir(member_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                first_image = images[0]

                # Define the source and destination paths
                source_image_path = os.path.join(member_folder, first_image)
                destination_image_path = os.path.join(static_image_dir, f"{username}_{member_name}_{first_image}")

                # Copy the image to the static folder
                shutil.copy(source_image_path, destination_image_path)

                # Append family member details with new image URL
                family_members.append({
                    'name': member_name,
                    'image_url': f"/static/family_images/{username}_{member_name}_{first_image}"  # URL for the frontend
                })

                # Collect image URL for saving
                image_urls.append(destination_image_path)  # You can save the file paths if needed
    print(family_members)
    return jsonify({
        'total_family_members': len(family_members),
        'family_members': family_members
    })

# -------------------------------------------------------------------------------------------- #

# Update Password Route
@main.route('/update_password', methods=['POST'])
def update_password():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"success": False, "message": "User not logged in!"})
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({"success": False, "message": "User not found!"})
        
        current_password = request.json.get('current_password')
        new_password = request.json.get('new_password')
        
        # Verify the current password
        if not user.check_password(current_password):
            return jsonify({"success": False, "message": "Incorrect current password!"})
        
        # Update the password
        user.set_password(new_password)
        db.session.commit()
        
        return jsonify({"success": True, "message": "Password updated successfully!"})
    except Exception as e:
        print("Error updating password:", str(e))
        return jsonify({"success": False, "message": "An error occurred. Please try again."})
    
# -------------------------------------------------------------------------------------------- #

# Update Username Route with Folder Renaming
@main.route('/update_username', methods=['POST'])
def update_username():
    try:
        # Assuming the user is logged in and their user ID is stored in the session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"success": False, "message": "User not logged in!"})
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({"success": False, "message": "User not found!"})
        
        new_username = request.json.get('new_username')
        login_password = request.json.get('login_password')
        
        # Check if password is correct
        if not user.check_password(login_password):
            return jsonify({"success": False, "message": "Incorrect password!"})
        
        # Check if the new username is already taken
        if User.query.filter_by(username=new_username).first():
            return jsonify({"success": False, "message": "Username already taken!"})
        
        # Define the old and new folder paths
        old_folder_path = get_base_folder(user.username)
        new_folder_path = get_base_folder(new_username)
        
        # Update username in the database
        user.username = new_username
        db.session.commit()
        
        # Rename the user's folder if it exists
        if os.path.exists(old_folder_path):
            os.rename(old_folder_path, new_folder_path)
        
        return jsonify({"success": True, "message": "Username and folder updated successfully!, login again"})
    
    except Exception as e:
        print("Error updating username:", str(e))
        return jsonify({"success": False, "message": "An error occurred. Please try again."})
