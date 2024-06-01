from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
import os
import face_recognition
import cv2
import logging
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')  # Use environment variable for security
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_faces_in_image(image_path):
    logger.info(f"Detecting faces in image: {image_path}")
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    return image, face_locations

def align_and_blend_faces(image, face_location, new_face_image):
    top, right, bottom, left = face_location

    # Extract the face region from the original image
    face_image = image[top:bottom, left:right]

    # Resize the new face to match the size of the original face
    new_face_resized = cv2.resize(new_face_image, (right - left, bottom - top))

    # Ensure the new face has the same shape as the original face
    new_face_resized = new_face_resized[:face_image.shape[0], :face_image.shape[1]]

    # Detect face landmarks in both images
    face_landmarks_original = face_recognition.face_landmarks(face_image)
    face_landmarks_new = face_recognition.face_landmarks(new_face_resized)

    if not face_landmarks_original or not face_landmarks_new:
        raise ValueError("Failed to detect face landmarks in one of the images.")

    # Get the facial points for warping
    points_original = np.array(face_landmarks_original[0]['chin'], np.float32)
    points_new = np.array(face_landmarks_new[0]['chin'], np.float32)

    # Compute the affine transformation
    matrix = cv2.estimateAffinePartial2D(points_new, points_original)[0]

    # Warp the new face to match the position and orientation of the original face
    new_face_warped = cv2.warpAffine(new_face_resized, matrix, (face_image.shape[1], face_image.shape[0]))

    # Create a mask for blending
    mask = np.zeros_like(face_image)
    cv2.fillConvexPoly(mask, np.int32(face_landmarks_original[0]["chin"]), (255, 255, 255))

    # Use seamless cloning to blend the warped face with the original image
    blended_face = cv2.seamlessClone(new_face_warped, face_image, mask, (face_image.shape[1] // 2, face_image.shape[0] // 2), cv2.NORMAL_CLONE)

    # Replace the face in the original image with the blended face
    image[top:bottom, left:right] = blended_face

    return image

def swap_faces(image, face_locations, new_face_path):
    logger.info(f"Swapping faces using new face image: {new_face_path}")
    new_face_image = face_recognition.load_image_file(new_face_path)
    
    if not face_locations:
        raise ValueError("No faces detected in the original image.")

    for face_location in face_locations:
        try:
            image = align_and_blend_faces(image, face_location, new_face_image)
        except Exception as e:
            logger.error(f"Error swapping faces: {e}")
            continue
        
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        new_face_file = request.files['new_face']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        new_face_path = os.path.join(app.config['UPLOAD_FOLDER'], new_face_file.filename)
        
        logger.info(f"Received file: {file.filename}")
        logger.info(f"Received new face file: {new_face_file.filename}")
        
        file.save(file_path)
        new_face_file.save(new_face_path)

        try:
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image, face_locations = detect_faces_in_image(file_path)
                swapped_image = swap_faces(image, face_locations, new_face_path)
                
                output_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_image.jpg')
                cv2.imwrite(output_image_path, cv2.cvtColor(swapped_image, cv2.COLOR_RGB2BGR))
                flash('Face swap successful!', 'success')
                return redirect(url_for('download_file', filename='output_image.jpg'))

            else:
                raise ValueError("Unsupported file type.")
        except ValueError as e:
            logger.error(f"Error: {e}")
            flash(str(e), 'error')
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            flash('An unexpected error occurred. Please try again.', 'error')

    return render_template('index.html')

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
