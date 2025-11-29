import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Potato Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Earthy color palette CSS
st.markdown("""
    <style>
    /* Main background - warm earth tone */
    .main {
        background: linear-gradient(135deg, #f5f3e8 0%, #e8e5d6 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4a5f3a 0%, #3d4f2f 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #f5f3e8 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #3d4f2f !important;
        font-family: 'Georgia', serif;
    }
    
    /* Cards/containers */
    .stContainer, .element-container {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6b8e5f 0%, #4a5f3a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #7da472 0%, #5a7048 100%);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px dashed #6b8e5f;
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6b8e5f 0%, #4a5f3a 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #3d4f2f;
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.9);
        border-left: 5px solid #6b8e5f;
        border-radius: 5px;
    }
    
    /* Custom container */
    .earthy-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(245,243,232,0.9) 100%);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Leaf decoration */
    .leaf-decoration {
        font-size: 3rem;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = 'models/potato_model.h5'
    if not os.path.exists(model_path):
        st.error(f"🚨 Model not found! Please download it to {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Class names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Disease information
DISEASE_INFO = {
    "Early Blight": {
        "emoji": "⚠️",
        "description": "Early blight is caused by the fungus Alternaria solani. It typically appears as dark spots with concentric rings.",
        "symptoms": [
            "Dark brown spots with concentric rings (bull's-eye pattern)",
            "Yellowing around spots",
            "Leaf drop in severe cases",
            "Affects lower, older leaves first"
        ],
        "treatment": [
            "Remove and destroy infected leaves",
            "Apply fungicide containing chlorothalonil",
            "Improve air circulation between plants",
            "Avoid overhead watering",
            "Rotate crops annually"
        ],
        "color": "#d4a574"
    },
    "Late Blight": {
        "emoji": "🚨",
        "description": "Late blight is caused by Phytophthora infestans. This is a serious disease that can destroy entire crops rapidly.",
        "symptoms": [
            "Water-soaked spots on leaves",
            "Gray-white fuzzy growth on undersides",
            "Brown-black lesions spreading quickly",
            "Affects stems and tubers",
            "Rapid plant death in humid conditions"
        ],
        "treatment": [
            "Remove infected plants immediately",
            "Apply copper-based fungicides preventatively",
            "Destroy infected plant material (don't compost)",
            "Monitor weather - spreads in cool, wet conditions",
            "Use resistant varieties"
        ],
        "color": "#c74440"
    },
    "Healthy": {
        "emoji": "✅",
        "description": "Your potato plant looks healthy! Continue good practices to keep it that way.",
        "symptoms": [
            "Vibrant green leaves",
            "No spots or discoloration",
            "Strong, upright growth",
            "Normal leaf texture"
        ],
        "treatment": [
            "Maintain consistent watering",
            "Ensure good drainage",
            "Provide adequate sunlight (6-8 hours)",
            "Regular monitoring for early signs of disease",
            "Proper spacing for air circulation"
        ],
        "color": "#6b8e5f"
    }
}

model = load_model()

# Header
st.markdown('<div class="leaf-decoration">🌿🥔🌿</div>', unsafe_allow_html=True)
st.title("🌱 Potato Leaf Disease Detector")
st.markdown("### *Nurturing healthy crops with AI*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🌾 About")
    st.markdown("""
    This application uses deep learning to identify diseases in potato leaves.
    
    **Detectable Conditions:**
    - 🟤 Early Blight
    - 🔴 Late Blight  
    - 🟢 Healthy
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 How It Works")
    st.markdown("""
    1. **Upload** a clear photo of a potato leaf
    2. **Analyze** using our trained AI model
    3. **Review** the diagnosis and recommendations
    """)
    
    st.markdown("---")
    
    if model:
        st.success("✓ Model Ready")
    else:
        st.error("✗ Model Not Loaded")
    
    st.markdown("---")
    st.markdown("### 🌍 Best Practices")
    st.info("""
    - Take photos in natural light
    - Focus on affected areas
    - Include full leaf when possible
    - Avoid blurry images
    """)

# Main content
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown('<div class="earthy-container">', unsafe_allow_html=True)
    st.markdown("## 📸 Upload Leaf Image")
    
    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a potato leaf for analysis"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    else:
        st.info("👆 Please upload an image to begin analysis")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="earthy-container">', unsafe_allow_html=True)
    st.markdown("## 🔍 Analysis Results")
    
    if uploaded_file is not None and model is not None:
        with st.spinner('🌿 Analyzing leaf health...'):
            # Preprocess
            img = image.resize((256, 256))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx] * 100
            
            predicted_disease = CLASS_NAMES[predicted_class_idx]
            disease_data = DISEASE_INFO[predicted_disease]
            
            # Display result
            st.markdown(f"### {disease_data['emoji']} **{predicted_disease}**")
            st.metric("Confidence Level", f"{confidence:.1f}%")
            
            # Confidence bar
            if confidence > 85:
                st.success(f"High confidence detection")
            elif confidence > 70:
                st.warning(f"Moderate confidence - consider consulting an expert")
            else:
                st.error(f"Low confidence - image may be unclear")
            
            st.markdown("---")
            
            # Probability bars
            st.markdown("#### 📊 Detection Probabilities")
            for i, class_name in enumerate(CLASS_NAMES):
                prob = predictions[0][i] * 100
                st.progress(prob / 100, text=f"{class_name}: {prob:.1f}%")
    
    elif uploaded_file is None:
        st.info("Upload an image to see results")
    else:
        st.error("Model not available")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Detailed information section
if uploaded_file is not None and model is not None:
    st.markdown("---")
    st.markdown('<div class="leaf-decoration">🍃</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 About This Condition", "🩺 Symptoms", "💊 Treatment"])
    
    with tab1:
        st.markdown(f"### {disease_data['emoji']} {predicted_disease}")
        st.write(disease_data['description'])
    
    with tab2:
        st.markdown("### Observable Symptoms")
        for symptom in disease_data['symptoms']:
            st.markdown(f"• {symptom}")
    
    with tab3:
        st.markdown("### Recommended Actions")
        for treatment in disease_data['treatment']:
            st.markdown(f"✓ {treatment}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b8e5f; padding: 2rem;'>
    <p style='font-size: 1.2rem;'>🌱 <strong>Cultivating Healthier Crops Through Technology</strong> 🌱</p>
    <p style='font-size: 0.9rem;'>Built with TensorFlow • Streamlit • Love for Agriculture</p>
</div>
""", unsafe_allow_html=True)
