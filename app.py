import sys, os
sys.path.insert(0, "/content/Omnivision_Classifier")

import gradio as gr
from src.model_utils import load_model, load_class_names, predict

model       = load_model()
class_names = load_class_names()

def gradio_predict(img):
    if img is None:
        return {cls: 0.0 for cls in class_names}, "Upload an image."
    return predict(model, class_names, img)

def get_img(name):
    path = f"/content/Omnivision_Classifier/results/{name}"
    return path if os.path.exists(path) else None

def get_accuracy_table():
    csv_path = "/content/Omnivision_Classifier/results/model_comparison.csv"
    if not os.path.exists(csv_path):
        return "<p>CSV not found.</p>"
    with open(csv_path) as f:
        lines = f.readlines()
    html = '<table style="width:100%;border-collapse:collapse;font-family:sans-serif;">'
    for i, line in enumerate(lines):
        cols = line.strip().split(",")
        if i == 0:
            html += "<tr style='background:#1e293b;color:white;'>"
            for col in cols:
                html += f"<th style='padding:10px;text-align:left;'>{col}</th>"
            html += "</tr>"
        else:
            bg = "#f8fafc" if i % 2 == 0 else "white"
            html += f"<tr style='background:{bg};border-bottom:1px solid #eee;'>"
            for j, c in enumerate(cols):
                if j == 0:
                    html += f"<td style='padding:10px;font-weight:bold;'>{c}</td>"
                else:
                    html += f"<td style='padding:10px;text-align:center;'>{c}</td>"
            html += "</tr>"
    html += "</table>"
    return html

with gr.Blocks(theme=gr.themes.Base(primary_hue="slate", secondary_hue="cyan")) as demo:
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#0f172a,#1e293b);padding:40px 20px;
                border-radius:15px;margin-bottom:25px;border:1px solid #334155;">
      <h1 style="color:#38bdf8;text-align:center;font-size:2.2rem;margin:0 0 10px 0;">
        OMNIVISION CLASSIFIER</h1>
      <h3 style="color:#94a3b8;text-align:center;font-weight:400;margin:0 0 20px 0;
                 text-transform:uppercase;letter-spacing:2px;">
        ML &amp; Programming for AI Project</h3>
      <div style="width:60px;height:2px;background:#38bdf8;margin:0 auto 20px;"></div>
      <h4 style="color:#e2e8f0;text-align:center;margin:0 0 25px 0;">
        Riphah International University</h4>
      <div style="display:flex;justify-content:center;gap:30px;">
        <div style="background:#334155;padding:10px 25px;border-radius:50px;">
          <span style="color:#38bdf8;font-weight:600;">Abdullah</span></div>
        <div style="background:#334155;padding:10px 25px;border-radius:50px;">
          <span style="color:#a78bfa;font-weight:600;">Mustafa</span></div>
      </div>
    </div>
    """)
    with gr.Tabs():
        with gr.Tab("⚡ Live Classifier"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img   = gr.Image(type="pil", label="Upload Image")
                    predict_btn = gr.Button("Run Prediction", variant="primary", size="lg")
                with gr.Column(scale=1):
                    status_text   = gr.Markdown("Waiting for image upload...")
                    output_labels = gr.Label(num_top_classes=5, label="Top-5 Predictions")
            predict_btn.click(fn=gradio_predict, inputs=input_img,
                              outputs=[output_labels, status_text])
        with gr.Tab("📊 Accuracy & Graphs"):
            gr.Markdown("### 📈 Model Comparison Metrics")
            gr.HTML(get_accuracy_table())
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📉 Training Curves")
                    gr.Image(value=get_img("training_curves.png"), interactive=False, show_label=False)
                with gr.Column():
                    gr.Markdown("### 🎯 Confusion Matrix")
                    gr.Image(value=get_img("confusion_matrix.png"), interactive=False, show_label=False)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📈 ROC Curves")
                    gr.Image(value=get_img("roc_curves.png"), interactive=False, show_label=False)
                with gr.Column():
                    gr.Markdown("### 🖼️ Sample Predictions")
                    gr.Image(value=get_img("sample_predictions.png"), interactive=False, show_label=False)

demo.launch(share=True)
