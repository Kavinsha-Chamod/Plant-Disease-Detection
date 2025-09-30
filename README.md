Plant Disease Detection Using Leaf Images

## Dependencies

Python 3.13
Libraries (install via pip install -r requirements.txt):

torch==2.3.0 (or latest compatible version)
torchvision==0.18.0
flask==3.0.0
scikit-learn==1.5.0
seaborn==0.13.2
numpy==1.26.0
pillow==10.3.0

## Setup Instructions

### Local Development

1.**Clone the Repository:**
git clone https://github.com/Kavinsha-Chamod/Plant-Disease-Detection.git
cd Plant-Disease-Detection

2.**Create and Activate Virtual Environment:**
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

3.**Install Dependencies:**
pip install -r requirements.txt

4.**Prepare Dataset:**
Download the "New Plant Diseases Dataset" from Kaggle.
Extract and place the train and valid folders in the data/ directory.

5.**Run the Application:**
Ensure resnet18.pth and simple_cnn.pth are in the root directory (generated from training).

Start the Flask app:
python app.py

Open http://127.0.0.1:5000 in your browser (for local testing).

### Deployment on Render

1.**Push to GitHub:**
Commit and push your code to this repository.

2.**Create Render Service:**
Sign up at Render.
Create a new Web Service, connecting to this GitHub repository.
Configure:

Build Command: pip install -r requirements.txt
Start Command: python app.py
Instance Type: Free (shared CPU)
Upload resnet18.pth as a static file via Render’s file system.

3.**Access the App:**
After deployment, Render provides a URL (e.g., https://plant-disease-scanner.onrender.com). Test it to ensure functionality.
Note: Free tier may sleep after inactivity; wake it by visiting the URL.

