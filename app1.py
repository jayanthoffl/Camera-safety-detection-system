import streamlit as st
import cv2
from ultralytics import YOLO
from datetime import datetime
import os
from deepface import DeepFace
import streamlink

# --- Page Config and Styling ---
st.set_page_config(page_title="OASIS Edge AI Unit", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3 { color: #FFFFFF; }
    .stExpander { border-radius: 10px; border: 2px solid #4A4A4A; }
    .stExpander[aria-expanded="true"] { border-color: #4CAF50; }
    .threat-alert[aria-expanded="true"] { border-color: #e63946; }
    .stButton>button { width: 100%; border: 2px solid #4A4A4A; background-color: #262730; color: #FAFAFA; }
    .stButton>button:hover { border-color: #e63946; color: #e63946; }
    .stFileUploader { background-color: #262730; border-radius: 10px; padding: 15px; }
    </style>
""", unsafe_allow_html=True)

st.title(" Real-time Security & Surveillance")

# --- Registered Faces DB ---
DB_PATH = "registered_faces"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# --- Session State ---
if "run_stream" not in st.session_state:
    st.session_state.run_stream = False
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# --- Load YOLO Models ---
@st.cache_resource
def load_models():
    general_model = YOLO("yolov8n.pt")   # General object detection
    gun_model = YOLO("bestgun.pt")       # Custom trained gun model
    knife_model = YOLO("bestknife.pt")   # Custom trained knife model
    return general_model, gun_model, knife_model

general_model, gun_model, knife_model = load_models()

# --- Load face detector (for drawing boxes) ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --- Helper: Resolve YouTube Live to direct stream URL ---
def get_stream_url(youtube_url):
    if not youtube_url:
        st.error("Please paste a YouTube Live URL first.")
        return None
    try:
        streams = streamlink.streams(youtube_url)
        if "best" in streams:
            url = streams["best"].url
            # Debug: see what URL we got (optional)
            st.write("Resolved stream URL:", url)
            return url
        else:
            st.error("No suitable stream quality found from this URL.")
            return None
    except Exception as e:
        st.error(f"Failed to fetch stream with streamlink: {e}")
        return None

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔴 Live Detection", "👤 Face Registration", "🚨 Alerts Log"])

#  Live Detection ---
with tab1:
    st.header("Live Monitoring (YOLOv8 + Weapon + Face Recognition)")

    # --- Stream Source Selector ---
    source_type = st.radio("Choose Video Source:", ["YouTube Live Stream", "Webcam"])

    youtube_url = None
    stream_url = None

    if source_type == "YouTube Live Stream":    
        youtube_url = st.text_input(
            "📺 Paste YouTube Live URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url_input",
        )
    else:
        stream_url = 0  # Webcam index

    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("▶️ Start Stream"):
        st.session_state.run_stream = True

    if btn_col2.button("⏹️ Stop Stream"):
        st.session_state.run_stream = False

    frame_placeholder = st.empty()

    if st.session_state.run_stream:
        # Resolve YouTube URL only when needed
        if source_type == "YouTube Live Stream":
            stream_url = get_stream_url(youtube_url)

        if stream_url is None:
            st.error("⚠️ No valid video source. Fix the URL or select Webcam.")
        else:
            cap = cv2.VideoCapture(stream_url)

            if not cap.isOpened():
                st.error("⚠️ Unable to open video source. Check URL/webcam/OpenCV-FFmpeg support.")
            else:
                last_threat_alert_time = {}

                while st.session_state.run_stream:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Stream ended or cannot read frame.")
                        break

                    annotated_frame = frame.copy()

                    # General Object Detection (YOLO) 
                    results_general = general_model(frame, verbose=False, conf=0.3)
                    for r in results_general:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = r.names[int(box.cls[0])]
                            conf = float(box.conf[0])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                annotated_frame,
                                f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )

                    # Gun Detection (custom YOLO)
                    results_gun = gun_model(frame, verbose=False, conf=0.5)
                    for r in results_gun:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(
                                annotated_frame,
                                "!! GUN !!",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )

                            now = datetime.now()
                            last_time = last_threat_alert_time.get("Gun", datetime.min)
                            if (now - last_time).total_seconds() > 10:
                                last_threat_alert_time["Gun"] = now
                                _, buffer = cv2.imencode(".jpg", frame)
                                st.session_state.alerts.append(
                                    {
                                        "time": now.strftime("%H:%M:%S"),
                                        "event": "Gun Detected",
                                        "screenshot": buffer.tobytes(),
                                        "type": "threat",
                                    }
                                )
                                st.toast("🚨 Gun Detected", icon="🚨")

                    # Knife Detection (custom YOLO)
                    results_knife = knife_model(frame, verbose=False, conf=0.5)
                    for r in results_knife:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(
                                annotated_frame,
                                "!! KNIFE !!",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )

                            now = datetime.now()
                            last_time = last_threat_alert_time.get("Knife", datetime.min)
                            if (now - last_time).total_seconds() > 10:
                                last_threat_alert_time["Knife"] = now
                                _, buffer = cv2.imencode(".jpg", frame)
                                st.session_state.alerts.append(
                                    {
                                        "time": now.strftime("%H:%M:%S"),
                                        "event": "Knife Detected",
                                        "screenshot": buffer.tobytes(),
                                        "type": "threat",
                                    }
                                )
                                st.toast("🚨 Knife Detected", icon="🚨")

                    # --- 4️⃣ Face Detection + Recognition (DeepFace) ---
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                        for (x, y, w, h) in faces:
                            face_img = frame[y : y + h, x : x + w]

                            # Find closest match in DB for this face
                            df_list = DeepFace.find(
                                img_path=face_img,
                                db_path=DB_PATH,
                                model_name="SFace",
                                enforce_detection=False,
                                detector_backend="opencv",
                                silent=True,
                            )

                            identity_name = "Unknown"

                            if len(df_list) > 0 and not df_list[0].empty:
                                best_match = df_list[0].iloc[0]
                                identity_path = best_match["identity"]
                                identity_name = os.path.basename(identity_path).rsplit(".", 1)[0]

                            # Draw bounding box + name
                            cv2.rectangle(
                                annotated_frame,
                                (x, y),
                                (x + w, y + h),
                                (255, 255, 0),
                                2,
                            )
                            cv2.putText(
                                annotated_frame,
                                identity_name,
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 0),
                                2,
                            )
                    except Exception:
                        # If DeepFace or cascade fails, just skip recognition for this frame
                        pass

                    # --- Show frame in UI ---
                    frame_placeholder.image(annotated_frame, channels="BGR")

                cap.release()
    else:
        frame_placeholder.markdown("### Stream is stopped.")

# --- 2️⃣ Face Registration ---
with tab2:
    st.header("Register Known Individuals")
    st.info("Upload a clear photo. Name should have no spaces (e.g., 'jayanth_ramakrishnan').")
    col1, col2 = st.columns([1, 2])

    with col1:
        name_input = st.text_input("Enter person's name:", key="reg_name")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "png", "jpeg"], key="reg_file"
        )

        if st.button("Register Face", disabled=(not name_input or not uploaded_file)):
            if " " in name_input:
                st.error("❌ Please remove spaces from the name.")
            else:
                try:
                    file_ext = os.path.splitext(uploaded_file.name)[1]
                    file_path = os.path.join(DB_PATH, f"{name_input}{file_ext}")
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    with st.spinner("Updating face database..."):
                        pkl_file = os.path.join(DB_PATH, "representations_sface.pkl")
                        if os.path.exists(pkl_file):
                            os.remove(pkl_file)
                        DeepFace.find(
                            img_path=file_path,
                            db_path=DB_PATH,
                            enforce_detection=False,
                            silent=True,
                            model_name="SFace",
                            detector_backend="opencv",
                        )
                    st.success(f"✅ Registered {name_input}!")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("Registered Individuals")
        image_files = [
            f for f in os.listdir(DB_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            st.warning("No faces registered yet.")
        else:
            for file in image_files:
                file_path = os.path.join(DB_PATH, file)
                col_image, col_btn = st.columns([2, 1])
                with col_image:
                    st.image(file_path, caption=os.path.splitext(file)[0], width=150)
                with col_btn:
                    if st.button(f"❌ Remove {file}", key=f"remove_{file}"):
                        os.remove(file_path)
                        pkl_file = os.path.join(DB_PATH, "representations_sface.pkl")
                        if os.path.exists(pkl_file):
                            os.remove(pkl_file)
                        with st.spinner("Updating face database..."):
                            remaining_files = [
                                f
                                for f in os.listdir(DB_PATH)
                                if f.lower().endswith((".png", ".jpg", ".jpeg"))
                            ]
                            if remaining_files:
                                DeepFace.find(
                                    img_path=os.path.join(DB_PATH, remaining_files[0]),
                                    db_path=DB_PATH,
                                    enforce_detection=False,
                                    silent=True,
                                    model_name="SFace",
                                    detector_backend="opencv",
                                )
                        st.success(f"✅ Removed {file} and updated database.")
                        st.experimental_rerun()

# --- 3️⃣ Alerts Log ---
with tab3:
    st.header("Security & Activity Alerts")
    if not st.session_state.alerts:
        st.info("No alerts have been recorded yet.")
    else:
        for alert in reversed(st.session_state.alerts):
            expander_title = f"🚨 {alert['event']} at {alert['time']}"
            with st.expander(expander_title):
                st.image(alert["screenshot"], caption=f"Detection at {alert['time']}")
