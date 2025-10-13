import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
import requests
from streamlit_lottie import st_lottie
import time
from pathlib import Path

# --- Custom Modules (from your original script) ---
from image_ops import loader, preprocess, label
from models import predict_age, predict_feature

# --- Project Structure and Configuration ---
ROOT_DIR = Path(__file__).parent.absolute()
MODELS_DIR = ROOT_DIR / "models"
CASCADE_DIR = ROOT_DIR / "image_ops"
DATA_DIR = ROOT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
LOGS_DIR = DATA_DIR / "logs"

# Ensure all necessary directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Correct, portable paths to model and cascade files
FEATURES_MODEL_PATH = MODELS_DIR / "efficientnet_b0_face_classifier_finetuned.h5"
AGE_MODEL_PATH = MODELS_DIR / "age_mobilenet_regression_stratified_old.h5"
CASCADE_FILENAME = CASCADE_DIR / "haarcascade_frontalface_default.xml"

# --- UI Enhancements & Animations ---

@st.cache_data(ttl=3600)
def load_lottieurl(url: str):
    """Fetches a Lottie animation from a URL with caching."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None

def apply_custom_css():
    """Applies custom CSS for styling and animations."""
    css = """
    <style>
        /* Hide Streamlit's default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* --- Animated Gradient Background --- */
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .stApp {
            background: linear-gradient(-45deg, #ffffff, #12a7b9, #48574f, #192650);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }

        /* --- Card Styling with Rise Effect --- */
        .interactive-card {
            padding: 2rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.95);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            margin-bottom: 1.5rem;
        }
        
        .interactive-card:hover {
            transform: translateY(-15px) scale(1.02);
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.4);
        }

        /* --- Smooth Fade-In Animations --- */
        @keyframes fadeInUp {
            from { 
                opacity: 0; 
                transform: translateY(40px); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(50px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes scaleIn {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        .fade-in {
            animation: fadeInUp 0.6s ease-out forwards;
        }
        
        .slide-in {
            animation: slideInRight 0.6s ease-out forwards;
        }
        
        .scale-in {
            animation: scaleIn 0.5s ease-out forwards;
        }
        
        .delay-1 { animation-delay: 0.1s; }
        .delay-2 { animation-delay: 0.2s; }
        .delay-3 { animation-delay: 0.3s; }
        .delay-4 { animation-delay: 0.4s; }

        /* --- Button Styling --- */
        .stButton>button {
            border-radius: 25px;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transition: all 0.3s ease;
            padding: 14px 32px;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        .stButton>button:active {
            transform: translateY(-1px);
        }

        /* --- Progress Bar Styling --- */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        /* --- Metric Styling --- */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* --- Text Input Styling --- */
        .stTextInput>div>div>input {
            border-radius: 15px;
            border: 2px solid rgba(102, 126, 234, 0.3);
            padding: 12px 16px;
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #667eea;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.3);
        }
        
        /* --- File Uploader Styling --- */
        [data-testid="stFileUploader"] {
            border-radius: 20px;
            border: 3px dashed rgba(102, 126, 234, 0.4);
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }
        
        /* --- Info Box Styling --- */
        .stAlert {
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        
        /* --- Expander Styling --- */
        .streamlit-expanderHeader {
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(102, 126, 234, 0.2);
            transform: translateX(5px);
        }
        
        /* --- Scrollbar Styling --- */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- User Authentication & Data Management ---
USERS_FILE = DATA_DIR / 'users.csv'

def get_user_log_path(username):
    """Get the log file path for a specific user."""
    return LOGS_DIR / f"{username}_log.csv"

def initialize_user_db():
    """Initialize the users database CSV file."""
    if not USERS_FILE.exists():
        pd.DataFrame(columns=['username', 'email', 'password_hash']).to_csv(
            USERS_FILE, index=False
        )

def initialize_user_log_file(username):
    """Initialize a user's log file if it doesn't exist."""
    user_log_file = get_user_log_path(username)
    if not user_log_file.exists():
        pd.DataFrame(columns=[
            'image_filename', 'timestamp', 'numpy_array_path', 'annotated_image_path',
            'clear_skin_%', 'dark_spots_%', 'puffy_eyes_%', 'wrinkles_%', 'skin_age'
        ]).to_csv(user_log_file, index=False)

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Basic email validation."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username):
    """Validate username (alphanumeric and underscore only, 3-20 chars)."""
    import re
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None

def validate_password(password):
    """Validate password strength (min 8 chars, at least one number)."""
    return len(password) >= 8 and any(c.isdigit() for c in password)

def create_user(username, email, password):
    """Create a new user account with validation."""
    if not validate_username(username):
        return False, "Username must be 3-20 characters (letters, numbers, underscore only)."
    
    if not validate_email(email):
        return False, "Please enter a valid email address."
    
    if not validate_password(password):
        return False, "Password must be at least 8 characters with at least one number."
    
    users_df = pd.read_csv(USERS_FILE)
    
    if username in users_df['username'].values:
        return False, "Username already exists."
    
    if email in users_df['email'].values:
        return False, "Email already registered."
    
    password_hash = hash_password(password)
    new_user = pd.DataFrame(
        [[username, email, password_hash]], 
        columns=['username', 'email', 'password_hash']
    )
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)
    
    return True, "Account created successfully! Please sign in."

def verify_user(username, password):
    """Verify user credentials."""
    try:
        users_df = pd.read_csv(USERS_FILE)
        user_record = users_df[users_df['username'] == username]
        
        if not user_record.empty:
            stored_hash = user_record.iloc[0]['password_hash']
            return hash_password(password) == stored_hash
    except Exception as e:
        st.error(f"Authentication error: {e}")
    
    return False

def append_log(username, log_data):
    """Append analysis results to user's log file."""
    user_log_file = get_user_log_path(username)
    initialize_user_log_file(username)
    
    try:
        log_df = pd.read_csv(user_log_file)
        new_log_entry = pd.DataFrame([log_data])
        log_df = pd.concat([log_df, new_log_entry], ignore_index=True)
        log_df.to_csv(user_log_file, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to save log: {e}")
        return False

# --- Model Loading ---
@st.cache_resource
def load_models_and_cascade():
    """Load ML models and cascade classifier with caching."""
    try:
        face_cascade = loader.load_cascade(str(CASCADE_FILENAME))
        age_model = predict_age.load_model(str(AGE_MODEL_PATH))
        feature_model = predict_feature.load_model(str(FEATURES_MODEL_PATH))
        return face_cascade, age_model, feature_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# --- UI Pages ---

def show_auth_page():
    """Display the authentication page (login/register)."""
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_x6s26o.json"
    lottie_json = load_lottieurl(lottie_url)

    if lottie_json:
        st_lottie(lottie_json, speed=1, height=200, key="auth_lottie")
    
    st.title("🌟 Welcome to DermalScan")
    st.write("Unlock AI-powered insights into your skin's health and age.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🔐 Sign In")
        
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please fill in all fields.")
                elif verify_user(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success("Login successful!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    
    with col2:
        st.header("✨ Create Account")
        
        with st.form("register_form"):
            new_username = st.text_input("Choose a Username", key="reg_username")
            new_email = st.text_input("Email Address", key="reg_email")
            new_password = st.text_input("Create a Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            submit = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("Please fill in all fields.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, message = create_user(new_username, new_email, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

def show_dashboard():
    """Display the main dashboard for logged-in users."""
    username = st.session_state['username']
    
    # Header with logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 DermalScan Dashboard")
        st.markdown(f"Hello, **{username}**! Ready for your analysis?")
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.rerun()
    
    st.markdown("---")
    
    # Load models
    face_cascade, age_model, feature_model = load_models_and_cascade()
    
    # File uploader
    st.subheader("📸 Upload Your Image")
    st.info("💡 **Tips**: Use a well-lit, front-facing photo. Remove glasses and accessories for best results.")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file:
        # Read and decode image
        img_bytes = uploaded_file.read()
        full_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if full_image is None:
            st.error("❌ Could not decode image. Please try a different file.")
            return

        # Show loading animation
        analysis_placeholder = st.empty()
        lottie_analysis_url = "https://assets1.lottiefiles.com/packages/lf20_gijbedz7.json"
        lottie_analysis = load_lottieurl(lottie_analysis_url)
        
        with analysis_placeholder.container():
            if lottie_analysis:
                st_lottie(lottie_analysis, speed=1, height=200, key="analysis_lottie")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("🔍 Detecting face...")
                elif i < 60:
                    status_text.text("🧠 Analyzing features...")
                else:
                    status_text.text("📊 Calculating results...")
                time.sleep(0.02)
        
        # Perform analysis
        try:
            face_image = preprocess.bytes_to_image(img_bytes)
            age = predict_age.predict_age(age_model, face_image)
            features = predict_feature.predict_features(feature_model, face_image)
            annotated_image = label.draw_labels_on_image(
                full_image.copy(), age, features, face_cascade
            )
        except Exception as e:
            analysis_placeholder.empty()
            st.error(f"❌ No face detected or prediction failed: {e}")
            st.info("💡 Try a well-lit image with a clearly visible face.")
            return

        # Clear loading animation
        analysis_placeholder.empty()
        
        # Display results
        st.header("✨ Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Processed Image")
            # Show annotated image (convert BGR to RGB for display)
            st.image(
                cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                caption="Your analyzed image with annotations",
                use_container_width=True
            )
        
        with col2:
            st.subheader("📈 Metrics")
            st.metric(
                "Estimated Skin Age", 
                f"{age:.1f} years",
                help="Estimated biological age of your skin"
            )
            
            st.markdown("---")
            st.subheader("🔬 Feature Analysis")
            
            # Display features with progress bars
            for feat, prob in features.items():
                feature_name = feat.replace('_', ' ').title()
                st.write(f"**{feature_name}**")
                # Convert numpy float32 to Python float and ensure value is between 0 and 1
                progress_value = float(prob.item() if hasattr(prob, 'item') else prob)
                if progress_value > 1.0:
                    progress_value = progress_value / 100.0
                progress_value = max(0.0, min(1.0, progress_value))  # Clamp between 0 and 1
                st.progress(progress_value)
                st.caption(f"{progress_value*100:.1f}%")
            
            st.markdown("---")
            st.subheader("💾 Download")
            
            success, img_encoded = cv2.imencode(".png", annotated_image)
            if success:
                st.download_button(
                    "⬇️ Download Annotated Image",
                    img_encoded.tobytes(),
                    f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    "image/png",
                    use_container_width=True
                )
                
            # Download predictions CSV
            csv_header = "timestamp,age," + ",".join(features.keys())
            csv_values = [datetime.now().isoformat(), f"{age:.1f}"] + [f"{progress_value*100:.1f}%" for progress_value in [float(p.item() if hasattr(p, 'item') else p) for p in features.values()]]
            csv_row = ",".join(csv_values)
            csv_text = csv_header + "\n" + csv_row
            
            st.download_button(
                "📊 Download Predictions CSV",
                csv_text,
                f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        # Save results to log
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{username}_{timestamp_str}_{Path(uploaded_file.name).stem}"
        annotated_image_path = UPLOADS_DIR / f"{base_filename}_annotated.png"
        numpy_array_path = UPLOADS_DIR / f"{base_filename}_face.npy"
        
        # Save annotated image (no double conversion)
        cv2.imwrite(str(annotated_image_path), annotated_image)
        np.save(str(numpy_array_path), face_image)

        # Save to user log
        log_data = {
            'image_filename': uploaded_file.name,
            'timestamp': datetime.now().isoformat(),
            'numpy_array_path': str(numpy_array_path),
            'annotated_image_path': str(annotated_image_path),
            'clear_skin_%': float(features.get('clear skin', 0).item() if hasattr(features.get('clear skin', 0), 'item') else features.get('clear skin', 0)) * 100,
            'dark_spots_%': float(features.get('dark spots', 0).item() if hasattr(features.get('dark spots', 0), 'item') else features.get('dark spots', 0)) * 100,
            'puffy_eyes_%': float(features.get('puffy eyes', 0).item() if hasattr(features.get('puffy eyes', 0), 'item') else features.get('puffy eyes', 0)) * 100,
            'wrinkles_%': float(features.get('wrinkles', 0).item() if hasattr(features.get('wrinkles', 0), 'item') else features.get('wrinkles', 0)) * 100,
            'skin_age': float(age.item() if hasattr(age, 'item') else age)
        }
        
        # Also save to daily log
        log_filename = LOGS_DIR / f"predictions_{datetime.now().date()}.csv"
        csv_header = "timestamp,username,image_filename,age," + ",".join(features.keys())
        csv_values = [
            datetime.now().isoformat(),
            username,
            uploaded_file.name,
            f"{log_data['skin_age']:.1f}"
        ] + [f"{float(p.item() if hasattr(p, 'item') else p)*100:.1f}%" for p in features.values()]
        csv_row = ",".join(csv_values)
        
        if not log_filename.exists():
            with open(log_filename, "w") as f:
                f.write(csv_header + "\n")
        with open(log_filename, "a") as log_file:
            log_file.write(csv_row + "\n")
        
        if append_log(username, log_data):
            st.success("✅ Analysis complete and results saved!")
            st.balloons()

    # Analysis History
    st.markdown("---")
    st.header("📜 Your Analysis History")
    
    user_log_file = get_user_log_path(username)
    
    try:
        log_df = pd.read_csv(user_log_file)
        
        if not log_df.empty:
            st.info(f"📊 Showing your last {min(5, len(log_df))} analyses")
            
            for index, row in log_df.tail(5).iloc[::-1].iterrows():
                timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                
                with st.expander(
                    f"🕐 {timestamp} - `{row['image_filename']}`",
                    expanded=False
                ):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        img_path = Path(row['annotated_image_path'])
                        if img_path.exists():
                            image = cv2.imread(str(img_path))
                            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        else:
                            st.warning("⚠️ Image file not found")
                    
                    with col2:
                        st.metric("Skin Age", f"{row['skin_age']:.1f} years")
                        st.write(f"**Clear Skin:** {row['clear_skin_%']:.1f}%")
                        st.write(f"**Dark Spots:** {row['dark_spots_%']:.1f}%")
                        st.write(f"**Puffy Eyes:** {row['puffy_eyes_%']:.1f}%")
                        st.write(f"**Wrinkles:** {row['wrinkles_%']:.1f}%")
        else:
            st.info("📭 No analysis history yet. Upload an image to get started!")
            
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.info("📭 No analysis history yet. Upload an image to get started!")

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="DermalScan - AI Skin Analysis",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    apply_custom_css()
    initialize_user_db()

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
    
    # Route to appropriate page
    if st.session_state.logged_in:
        initialize_user_log_file(st.session_state.username)
        show_dashboard()
    else:
        show_auth_page()

if __name__ == "__main__":
    main()