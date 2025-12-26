from flask import Flask, render_template, request, jsonify, send_from_directory
import face_recognition
import numpy as np
import os
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# ===============================
# PHOTO STORAGE (ONE SOURCE ONLY)
# ===============================
PHOTO_FOLDER = "static/photos"
os.makedirs(PHOTO_FOLDER, exist_ok=True)

# ===============================
# FACE ENCODING
# ===============================
def get_face_encoding(image_np):
    enc = face_recognition.face_encodings(image_np)
    return enc[0] if enc else None

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/instruction")
def instruction():
    return render_template("instruction.html")

# ===============================
# LOAD PHOTOS
# ===============================
@app.route("/load_photos")
def load_photos():
    photos = []

    for f in os.listdir(PHOTO_FOLDER):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            photos.append({
                "id": f,
                "name": f,
                "view_url": f"/static/photos/{f}"
            })

    return jsonify({
        "photos": photos,
        "count": len(photos)
    })

# ===============================
# DOWNLOAD
# ===============================
@app.route("/download/<filename>")
def download_photo(filename):
    return send_from_directory(PHOTO_FOLDER, filename, as_attachment=True)

# ===============================
# FACE SEARCH
# ===============================
@app.route("/face_search", methods=["GET", "POST"])
def face_search():
    if request.method == "GET":
        return render_template("face_search.html")

    file = request.files.get("face_image")
    img = Image.open(file).convert("RGB")
    target = get_face_encoding(np.array(img))

    if target is None:
        return render_template("face_search.html", error="No face detected")

    matches = []
    for f in os.listdir(PHOTO_FOLDER):
        path = os.path.join(PHOTO_FOLDER, f)
        try:
            img = Image.open(path).convert("RGB")
            enc = get_face_encoding(np.array(img))
            if enc is None:
                continue
            dist = np.linalg.norm(target - enc)
            if dist <= 0.5:
                matches.append({
                    "id": f,
                    "name": f,
                    "distance": round(float(dist), 3),
                    "view_url": f"/static/photos/{f}"
                })
        except:
            continue

    return render_template("face_search.html", results=matches)

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
