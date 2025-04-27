import os
os.environ["PYTORCH_NO_DISTRIBUTED"] = "1"
import torch
torch.classes.__path__ = []
import streamlit as st

import wfdb

import tempfile
import numpy as np
from Preprocess import ECGPreprocess
from model_handler import load_model, predict

# Set page configuration
st.set_page_config(
    page_title="Chagas Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling for white background
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    .main-header {
        font-size: 1.8rem;
        color: #4A4A4A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 2rem;
        border-top: 1px solid #ddd;
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Password protection
def check_password():
    """Returns `True` if the user had the correct password and agreed to disclaimer."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "demo123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.markdown("<h1 class='main-header'>Using electrocardiogram data with deep neural networks to predict a neglected tropical disease: Chagas Disease</h1>", unsafe_allow_html=True)
        
        st.text_input(
            "Please enter the password", 
            type="password", 
            key="password"
        )
        
        disclaimer = """
        **Disclaimer & Acknowledgment**
        
        This application is a demonstration tool developed solely for educational and portfolio purposes. It is intended to showcase the developer's technical skills in machine learning, deep learning, and software development. **It is not a medical device**, has **not been reviewed or approved by any medical authority**, and **must not be used for diagnosis, treatment, or any medical decision-making**.
        
        By using this tool, you acknowledge and agree that:
        * You understand that the application is not intended for clinical or diagnostic use.
        * You will not interpret the output of this tool as medical advice or a substitute for professional healthcare.
        * You assume full responsibility for any decisions or actions taken based on the output of the application.
        * The developer of this application is not liable for any damages, losses, or harm resulting from its use.
        """
        
        st.markdown(disclaimer)
        agree = st.checkbox("**I acknowledge and agree to the terms above.**")
        
        # Add submit button
        if st.button("Submit"):
            password_entered()
            
            if not st.session_state["password_correct"]:
                st.error("😕 Password incorrect")
            elif not agree:
                st.error("Please acknowledge the disclaimer to continue")
            else:
                st.session_state["disclaimer_agreed"] = True
                st.rerun()
                
        return False
    elif not st.session_state.get("password_correct", False) or not st.session_state.get("disclaimer_agreed", False):
        # Password not correct or disclaimer not agreed, show input + error.
        st.markdown("<h1 class='main-header'>Using electrocardiogram data with deep neural networks to predict a neglected tropical disease: Chagas Disease</h1>", unsafe_allow_html=True)
        
        st.text_input(
            "Please enter the password", 
            type="password", 
            key="password"
        )
        
        disclaimer = """
        **Disclaimer & Acknowledgment**
        
        This application is a demonstration tool developed solely for educational and portfolio purposes. It is intended to showcase the developer's technical skills in machine learning, deep learning, and software development. **It is not a medical device**, has **not been reviewed or approved by any medical authority**, and **must not be used for diagnosis, treatment, or any medical decision-making**.
        
        By using this tool, you acknowledge and agree that:
        * You understand that the application is not intended for clinical or diagnostic use.
        * You will not interpret the output of this tool as medical advice or a substitute for professional healthcare.
        * You assume full responsibility for any decisions or actions taken based on the output of the application.
        * The developer of this application is not liable for any damages, losses, or harm resulting from its use.
        """
        
        st.markdown(disclaimer)
        agree = st.checkbox("**I acknowledge and agree to the terms above.**")
        
        # Add submit button
        if st.button("Submit"):
            password_entered()
            
            if not st.session_state["password_correct"]:
                st.error("😕 Password incorrect")
            elif not agree:
                st.error("Please acknowledge the disclaimer to continue")
            else:
                st.session_state["disclaimer_agreed"] = True
                st.rerun()
                
        return False
    else:
        # Password correct and disclaimer agreed.
        return True

if check_password():
    st.markdown("<h1 class='main-header'>Using electrocardiogram data with deep neural networks to predict a neglected tropical disease: Chagas Disease</h1>", unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Patient Information")
        
        # Sex selection
        sex_options = {
            "Female": 0,
            "Male": 1,
            "Unknown": 2
        }
        sex = st.selectbox("Sex", options=list(sex_options.keys()))
        sex_value = sex_options[sex]
        
        # Age group selection
        age_options = {
            "0-9": 0,
            "10-19": 1,
            "20-29": 2,
            "30-39": 3,
            "40-49": 4,
            "50-59": 5,
            "60-69": 6,
            "70-79": 7,
            "80-89": 8,
            "90+": 9,
            "Unknown": 9
        }
        age_group = st.selectbox("Age Group", options=list(age_options.keys()))
        age_value = age_options[age_group]
        
        # Sampling rate input
        sampling_rate = st.number_input("Sampling Rate (Hz)", min_value=1, value=400, step=1)
        
        # File uploads
        st.header("ECG Data Files")
        dat_file = st.file_uploader("Upload .dat file", type=["dat"])
        hea_file = st.file_uploader("Upload .hea file", type=["hea"])
        
        # Extract record name from uploaded file name
        record_name = None
        if dat_file:
            # Get filename without extension
            record_name = os.path.splitext(dat_file.name)[0]
            st.write(f"Record name detected: {record_name}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("ECG Analysis")
        st.write("Upload the required files and enter patient information to get a prediction.")
        
        if dat_file and hea_file and record_name:
            st.success("Files uploaded successfully!")
            
            # Process the data when button is clicked
            if st.button("Run Analysis"):
                with st.spinner("Processing ECG data..."):
                    try:
                        # Create temp directory to save files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save uploaded files using the original filename
                            dat_path = os.path.join(temp_dir, f"{record_name}.dat")
                            hea_path = os.path.join(temp_dir, f"{record_name}.hea")
                            
                            with open(dat_path, "wb") as f:
                                f.write(dat_file.getvalue())
                            with open(hea_path, "wb") as f:
                                f.write(hea_file.getvalue())
                            
                            # Debug info
                            #st.write(f"Files saved to: {temp_dir}")
                            #st.write(f"Record path: {os.path.join(temp_dir, record_name)}")
                            
                            # List files in directory to debug
                            #st.write("Files in temporary directory:")
                            #for file in os.listdir(temp_dir):
                            #    st.write(f"- {file}")
                            
                            # Load the record using the original record name
                            record_path = os.path.join(temp_dir, record_name)
                            
                            try:
                                record = wfdb.rdrecord(record_path)
                                st.success("Record loaded successfully!")
                                
                                # Display record information
                                #st.write("Record information:")
                                #st.write(f"- Signal shape: {record.p_signal.shape}")
                                #st.write(f"- Number of leads: {record.n_sig}")
                                #st.write(f"- Sample frequency: {record.fs}")
                                #st.write(f"- Number of samples: {record.sig_len}")
                                
                                # Convert inputs to tensors
                                sex_tensor = torch.tensor(sex_value, dtype=torch.long)
                                age_tensor = torch.tensor(age_value, dtype=torch.long)
                                
                                # Get device
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                #st.write(f"Using device: {device}")
                                
                                # Manual preprocessing to debug
                                #st.write("Beginning preprocessing...")
                                try:
                                    ecg_filter = ECGPreprocess(sampling_rate=sampling_rate)
                                    
                                    # Step by step processing
                                    #st.write("Starting ECG preprocessing...")
                                    data = ecg_filter.process_wfdb_files(
                                        record=record,
                                        apply_resample=True,
                                        apply_highpass=True,
                                        apply_lowpass=True,
                                        pad_to_length=4096,
                                        apply_wavelet=False,
                                        device=device
                                    )
                                    #st.write(f"ECG data processed, shape: {data.shape}")
                                    
                                    # Add batch dimension
                                    data = torch.unsqueeze(data, 0)
                                    #st.write(f"After adding batch dimension, shape: {data.shape}")
                                    
                                    # Prepare other inputs
                                    sex_tensor = torch.unsqueeze(sex_tensor, 0).to(device)
                                    age_tensor = torch.unsqueeze(age_tensor, 0).to(device)
                                    
                                    #st.write("Loading model...")
                                    # Load model
                                    model = load_model(device)
                                    #st.write("Model loaded successfully!")
                                    
                                    #st.write("Running inference...")
                                    # Run inference
                                    model.eval()
                                    with torch.no_grad():
                                        outputs = model(ecg=data, sex=sex_tensor, age_group=age_tensor)
                                        probabilities = torch.sigmoid(outputs).cpu().numpy()
                                        #st.write(f"Raw model output: {outputs}")
                                        #st.write(f"Sigmoid probabilities: {probabilities}")
                                    
                                    # Get probability value
                                    probability = float(probabilities)
                                    
                                    # Store results in session state
                                    st.session_state["prediction_result"] = probability
                                    
                                    # Display result
                                    st.markdown(f"""
                                    <div class="result-box">
                                        <h2>Prediction Result</h2>
                                        <h3>Probability of Chagas Disease: {probability:.2%}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                except Exception as e:
                                    st.error(f"Error in preprocessing or model inference: {str(e)}")
                                    import traceback
                                    st.error(traceback.format_exc())
                                
                            except Exception as e:
                                st.error(f"Error reading record: {str(e)}")
                                
                                # Try to read the header file content to debug
                                try:
                                    with open(hea_path, 'r') as f:
                                        st.write("Header file content:")
                                        st.code(f.read())
                                except Exception as e2:
                                    st.error(f"Could not read header file: {str(e2)}")
                    
                    except Exception as e:
                        st.error(f"Error processing ECG data: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
    
    with col2:
        st.header("About Chagas Disease")
        st.write("""
        Chagas disease is caused by the parasite *Trypanosoma cruzi* and is primarily transmitted by 
        infected triatomine bugs. It's endemic in Latin America but has spread to other regions due to migration.

        Early detection is crucial for effective treatment. This tool demonstrates how deep learning 
        can potentially assist in screening for this neglected tropical disease using ECG data.
        """)

        st.subheader("Acknowledgments")
        st.markdown("""
        The following model was developed using 
        [CODE-15](https://zenodo.org/records/4916206) and 
        [SaMi-Trop](https://zenodo.org/records/4905618) data as part of the 
        [George B. Moody PhysioNet Challenge 2025](https://moody-challenge.physionet.org/2025/).

        This application was inspired by a collaborative project originally developed by a team of three: myself, 
        [Marc Lafargue](https://www.linkedin.com/in/marclafargue/), and 
        [Matheus Rama Amorim](https://www.linkedin.com/in/matamorim/), 
        as part of our coursework at the Georgia Institute of Technology.
        """)

        st.markdown("""For a write-up of this project, you can download the full report below. The model used here is a 
                    streamlined version of that presented in the paper. It swaps full convolutions for depthwise-seperable and
                    each convolution block is only two deep. 
        """)

        # Add download button for the paper
        with open("Chagas_Prediction_Report.pdf", "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="📄 Download paper",
            data=pdf_bytes,
            file_name="Chagas_Prediction_Report.pdf",
            mime="application/pdf"
        )
        
    # Footer
    st.markdown("""
    <div class="disclaimer">
    <p><strong>Note:</strong> This application is for demonstration purposes only and is not intended for clinical use.</p>
    </div>
    """, unsafe_allow_html=True)