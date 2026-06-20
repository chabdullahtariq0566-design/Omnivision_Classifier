# ============================================================
# mlflow_ui.py — Start MLflow UI and expose via ngrok
# ============================================================

import subprocess, time, os
from pyngrok import ngrok

MLRUNS_DIR = "/content/Omnivision_Classifier/mlruns"

# ⚠️  APNA NGROK TOKEN YAHAN PASTE KARO
# Free account: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_TOKEN = "3FNDHsJ3z11CKEAIX6DuSTQoCvX_3EPWCA623tNRfb7NHbTof"

# Token set karo PEHLE connect se — yahi missing tha
ngrok.set_auth_token(NGROK_TOKEN)

if not os.path.exists(MLRUNS_DIR):
    print("ERROR: mlruns/ folder not found!")
    print(f"Expected at: {MLRUNS_DIR}")
    print("Copy mlruns/ from Kaggle output to this folder first.")
else:
    print("Starting MLflow server...")
    process = subprocess.Popen(
        [
            "mlflow", "ui",                        # <-- 'server' ki jagah 'ui' use karo
            "--host", "127.0.0.1",
            "--port", "5001",
            "--backend-store-uri", f"file://{MLRUNS_DIR}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(4)

    print("Creating ngrok tunnel...")
    tunnel     = ngrok.connect(5001)
    public_url = tunnel.public_url

    print("\n" + "="*60)
    print("  MLFLOW UI IS LIVE!")
    print("="*60)
    print(f"  URL: {public_url}")
    print("="*60)
    print("  Keep this cell running to keep the link active.")
    print("  Press Stop (square button) to close.")