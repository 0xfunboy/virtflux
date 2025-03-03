# **Virtflux (Alpha 0.01t)**
**Owner:** [0xfunboy](https://github.com/0xfunboy)

## **Overview**
Virtflux is an alpha-stage AI project that provides the following functionalities:

- **Image Analysis:** Analyze images using BLIP and OmniParser pipelines.
- **Image Generation:** Generate images using Stable Diffusion 2.1, FLUX Schnell, or Dreamlike Photoreal 2.0.
- **Text-to-Speech (TTS):** Convert text to speech using either the system's voice (pyttsx3) or Transformers TTS pipelines.

The FastAPI server runs on **port 2727**, while a separate static frontend is served on **port 2780**.

---
## **Project Structure**
```
virtflux/
├── .env                 # Environment variables (do not commit)
├── .gitignore           # Files and folders to ignore in Git
├── README.md            # This file
│
├── flux_projects/
│   ├── frontend/        # Static frontend files
│   │   ├── index.html
│   │   ├── documentation.html
│   │   └── docs/
│   │       └── index.html  # Swagger docs proxy page (iframe)
│   │
│   ├── inputs/          # Input test files (e.g., test images)
│   ├── outputs/         # Generated output files (images, audio, etc.)
│   │   └── tts/         # Text-to-Speech output directory
│   │
│   ├── scripts/         # Backend and test scripts
│   │   ├── api_server.py  # FastAPI server
│   │   └── test/
│   │       ├── stabled21_test.py
│   │       ├── tts_test.py
│   │       └── vision_ocr_caption.py
```

---
## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/0xfunboy/virtflux.git
cd virtflux
```

### **2. Create and Activate a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### **3. Install Dependencies**
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate xformers fastapi uvicorn pillow pyttsx3 soundfile
```

### **4. Create a `.env` File**
Copy the provided `.env` file and replace the placeholder values with your actual secrets.

---
## **Running the Application**
### **Start the FastAPI Server**
```bash
cd flux_projects/scripts
python3 api_server.py
```
The API server will run on `http://0.0.0.0:2727` and expose the Swagger docs at `http://x.x.x.x:2727/docs`.

### **Start the Frontend Static Server (Optional)**
```bash
cd ../frontend
python3 -m http.server 2780
```
Access the frontend at `http://x.x.x.x:2780/index.html`.

---
## **License**
This project is licensed under the **GPL v3.0**.

---
## **Contributing**
1. Fork this repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add new feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/my-feature
   ```
5. Open a Pull Request.

---
## **Notes**
- **This is an alpha release.** Use with caution in production environments.
- Report issues and contribute via [GitHub Issues](https://github.com/0xfunboy/virtflux/issues).

