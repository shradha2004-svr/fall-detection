
import os
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime


# ============================================================
# LOAD DEPLOYMENT DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "lightweight_model.h5"
)

demo_X = np.load(
    os.path.join(BASE_DIR, "demo_X.npy")
)

demo_y = np.load(
    os.path.join(BASE_DIR, "demo_y.npy")
)

demo_indices = np.load(
    os.path.join(BASE_DIR, "demo_indices.npy")
)

demo_meta = np.load(
    os.path.join(BASE_DIR, "demo_meta.npy"),
    allow_pickle=True
)


# ============================================================
# LOAD ACTUAL TRAINED MODEL
# ============================================================

lite_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ============================================================
# SAMPLE NAMES
# ============================================================

example_names = [
    "Normal ADL — Verified",
    "Real Fall — Detected 1",
    "Real Fall — Detected 2",
    "Hard Case — D08",
    "Hard Case — D10",
    "Hard Case — D11",
    "Hard Case — D13",
    "Hard Case — D18",
    "Hard Case — D19"
]

# Keep only names for available samples
example_names = example_names[:len(demo_X)]


# ============================================================
# CSS
# ============================================================

custom_css = """

body {
    background: #eef2f1 !important;
    font-family: Arial, Helvetica, sans-serif !important;
}

.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
    padding: 0 18px 40px 18px !important;
}

.dashboard-header {
    background: #071f2d !important;
    color: white !important;
    padding: 30px 36px;
    border-radius: 0 0 16px 16px;
    border-bottom: 4px solid #12a889;
    margin-bottom: 25px;
}

.dashboard-title {
    color: white !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    margin-bottom: 8px;
}

.dashboard-subtitle {
    color: #b8d6d3 !important;
    font-size: 14px !important;
}

.header-tag {
    display: inline-block;
    margin-top: 15px;
    padding: 6px 10px;
    border-radius: 5px;
    background: #0d3445;
    border: 1px solid #1c556b;
    color: #78d8c3;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

.panel {
    background: white;
    border: 1px solid #d8e2df;
    border-radius: 13px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(15,45,43,0.07);
}

.section-title {
    color: #102f3d !important;
    font-size: 18px !important;
    font-weight: 750 !important;
    margin: 8px 0 12px 0;
}

.prediction-card {
    background: white;
    border: 1px solid #d8e2df;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 4px 16px rgba(15,45,43,0.08);
}

.metric {
    background: #f6f9f8;
    border: 1px solid #e0e9e6;
    border-radius: 9px;
    padding: 14px;
}

.metric-label {
    color: #71807d;
    font-size: 10px;
    letter-spacing: .7px;
    font-weight: 600;
    margin-bottom: 7px;
}

.metric-value {
    color: #102f3d;
    font-size: 20px;
    font-weight: 750;
}

.metric-teal {
    color: #087d68;
    font-size: 20px;
    font-weight: 750;
}

.metric-red {
    color: #b52e27;
    font-size: 20px;
    font-weight: 750;
}

.status-ok {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 6px;
    background: #e7f6f0;
    border: 1px solid #a9ddcd;
    color: #08755f;
    font-size: 11px;
    font-weight: 800;
}

.status-fall {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 6px;
    background: #fff0ef;
    border: 1px solid #e8b6b2;
    color: #b52e27;
    font-size: 11px;
    font-weight: 800;
}

.info-strip {
    background: #eef7f5;
    border-left: 3px solid #0b8d75;
    padding: 10px 12px;
    border-radius: 4px;
    color: #41615c;
    font-size: 12px;
    margin-top: 14px;
}

.architecture {
    background: #071f2d;
    color: #b9e4db;
    border-radius: 9px;
    padding: 15px;
    font-family: Consolas, monospace;
    font-size: 11px;
    line-height: 1.7;
}

.footer {
    text-align: center;
    color: #7b8986;
    font-size: 11px;
    padding: 25px;
}

"""


# ============================================================
# SESSION
# ============================================================

session_log = []


# ============================================================
# PREDICTION
# ============================================================

def predict_sample(example_name, current_log):

    idx = example_names.index(example_name)

    sample = demo_X[idx]

    probability = float(
        lite_model.predict(
            sample[np.newaxis, :, :],
            verbose=0
        )[0][0]
    )

    prediction = (
        "FALL"
        if probability >= 0.50
        else "NORMAL / ADL"
    )

    confidence = (
        probability
        if prediction == "FALL"
        else 1 - probability
    )

    actual = (
        "FALL"
        if demo_y[idx] == 1
        else "NORMAL / ADL"
    )

    correct = prediction == actual


    if prediction == "FALL":

        status = '<span class="status-fall">FALL DETECTED</span>'

        prediction_color = "#b52e27"

    else:

        status = '<span class="status-ok">NORMAL ACTIVITY</span>'

        prediction_color = "#087d68"


    result_html = f"""

    <div class="prediction-card">

        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom:20px;
        ">

            <div style="
                color:#71807d;
                font-size:10px;
                letter-spacing:1.3px;
                font-weight:700;
            ">
                MODEL OUTPUT
            </div>

            {status}

        </div>

        <div style="
            color:#71807d;
            font-size:10px;
            letter-spacing:1px;
        ">
            CLASSIFICATION
        </div>

        <div style="
            color:{prediction_color};
            font-size:29px;
            font-weight:800;
            margin-top:7px;
        ">
            {prediction}
        </div>

        <div style="
            display:grid;
            grid-template-columns:repeat(3,1fr);
            gap:10px;
            margin-top:22px;
        ">

            <div class="metric">
                <div class="metric-label">
                    FALL PROBABILITY
                </div>
                <div class="metric-red">
                    {probability * 100:.2f}%
                </div>
            </div>

            <div class="metric">
                <div class="metric-label">
                    CONFIDENCE
                </div>
                <div class="metric-teal">
                    {confidence * 100:.2f}%
                </div>
            </div>

            <div class="metric">
                <div class="metric-label">
                    ACTUAL LABEL
                </div>
                <div class="metric-value">
                    {actual}
                </div>
            </div>

        </div>

        <div class="info-strip">

            {"Model output agrees with the labelled test sample."
             if correct
             else
             "Model output differs from the labelled test sample."}

        </div>

        <div style="
            margin-top:12px;
            color:#7a8986;
            font-size:10px;
        ">

            Test window #{int(demo_indices[idx])}
            &nbsp; • &nbsp;
            Input: 200 × 9

        </div>

    </div>

    """


    # ========================================================
    # SIGNAL PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 4.5),
        dpi=110
    )

    for ch in range(9):

        ax.plot(
            sample[:, ch],
            linewidth=1.15,
            label=f"Ch {ch+1}"
        )

    ax.set_title(
        f"{example_name} | Test Window #{int(demo_indices[idx])}",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Sensor Value")

    ax.grid(
        True,
        alpha=0.18
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        ncol=5,
        fontsize=8,
        frameon=False
    )

    plt.tight_layout()


    # ========================================================
    # LOG
    # ========================================================

    new_entry = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Sample": example_name,
        "Prediction": prediction,
        "Fall Prob.": f"{probability * 100:.2f}%",
        "Confidence": f"{confidence * 100:.2f}%",
        "Actual": actual,
        "Result": "Correct" if correct else "Mismatch"
    }

    if current_log is None:
        current_log = []

    current_log = [new_entry] + current_log

    log_df = pd.DataFrame(current_log)

    return result_html, fig, log_df, current_log


# ============================================================
# INITIAL OUTPUT
# ============================================================

initial_name = example_names[0]

initial_result, initial_plot, initial_log, initial_state = predict_sample(
    initial_name,
    []
)


# ============================================================
# DASHBOARD
# ============================================================

with gr.Blocks(
    css=custom_css,
    title="Fall Detection Dashboard"
) as demo:

    gr.HTML(
        """
        <div class="dashboard-header">

            <div class="dashboard-title">
                FALL DETECTION DASHBOARD
            </div>

            <div class="dashboard-subtitle">
                Lightweight CNN-LSTM model for sensor-based
                fall classification using the SisFall test set
            </div>

            <div class="header-tag">
                SOFTWARE MODEL DEMONSTRATION
            </div>

        </div>
        """
    )

    session_state = gr.State(initial_state)

    with gr.Row():

        with gr.Column(scale=3):

            gr.HTML(
                '<div class="section-title">TEST WINDOW</div>'
            )

            example_dropdown = gr.Dropdown(
                choices=example_names,
                value=initial_name,
                label="Select test sample"
            )

            analyze_button = gr.Button(
                "ANALYZE WINDOW",
                variant="primary"
            )

            gr.HTML(
                """
                <div class="panel" style="margin-top:15px">

                    <div class="section-title">
                        INPUT
                    </div>

                    <div style="
                        color:#687976;
                        font-size:12px;
                        line-height:1.8;
                    ">

                        Sensor channels: <b>9</b><br>
                        Time steps: <b>200</b><br>
                        Classification: <b>Binary</b><br>
                        Decision threshold: <b>0.50</b>

                    </div>

                </div>
                """
            )


        with gr.Column(scale=7):

            result_box = gr.HTML(
                value=initial_result
            )


        with gr.Column(scale=3):

            gr.HTML(
                """
                <div class="panel">

                    <div class="section-title">
                        MODEL
                    </div>

                    <div style="line-height:2;font-size:12px">

                        Dataset:
                        <b>SisFall</b><br>

                        Architecture:
                        <b>Lightweight CNN-LSTM</b><br>

                        Parameters:
                        <b>9,318</b><br>

                        Input:
                        <b>200 × 9</b><br>

                        Output:
                        <b>Fall / ADL</b><br>

                        Threshold:
                        <b>0.50</b>

                    </div>

                </div>
                """
            )


    gr.HTML(
        '<div class="section-title" style="margin-top:25px">'
        'SENSOR SIGNAL ANALYSIS'
        '</div>'
    )

    signal_plot = gr.Plot(
        value=initial_plot,
        show_label=False
    )


    with gr.Row():

        with gr.Column(scale=8):

            gr.HTML(
                '<div class="section-title">'
                'PREDICTION SESSION'
                '</div>'
            )

            log_table = gr.Dataframe(
                value=initial_log,
                interactive=False,
                wrap=True,
                max_height=260
            )


        with gr.Column(scale=4):

            gr.HTML(
                """
                <div class="panel">

                    <div class="section-title">
                        REPORTED PERFORMANCE
                    </div>

                    <b>90.51%</b> Accuracy<br>
                    <b>87.96%</b> Precision<br>
                    <b>83.36%</b> Recall<br>
                    <b>85.60%</b> F1 Score

                </div>
                """
            )


    gr.HTML(
        """
        <div style="margin-top:25px"
             class="section-title">

            MODEL ARCHITECTURE

        </div>

        <div class="architecture">

Input (200 × 9)
↓
SeparableConv1D — 24
↓
BatchNorm → MaxPool
↓
SeparableConv1D — 48
↓
BatchNorm → MaxPool
↓
LSTM — 24
↓
Dropout — 0.30
↓
Dense — 16
↓
Sigmoid
↓
FALL / NORMAL

        </div>
        """
    )


    gr.HTML(
        """
        <div class="footer">

            Lightweight CNN-LSTM
            • SisFall
            • MCA Project Demonstration

        </div>
        """
    )


    analyze_button.click(
        fn=predict_sample,
        inputs=[
            example_dropdown,
            session_state
        ],
        outputs=[
            result_box,
            signal_plot,
            log_table,
            session_state
        ]
    )


if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
