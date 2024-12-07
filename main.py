import streamlit as st
from ollama import chat
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any, List
from PIL import Image
import io
import psutil
import subprocess
import time
import sys
import os
import json
import platform
from datetime import datetime
import logging
from statistics import mean
from collections import deque

# Set up logging
log_file = "app.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# System metrics collection
METRICS_WINDOW = 100  # Keep last 100 readings
cpu_readings = deque(maxlen=METRICS_WINDOW)
memory_readings = deque(maxlen=METRICS_WINDOW)

def collect_system_metrics():
    """Collect and calculate system metrics."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    cpu_readings.append(cpu_percent)
    memory_readings.append(memory.percent)
    
    return {
        "current_cpu": cpu_percent,
        "current_memory": memory.percent,
        "avg_cpu": mean(cpu_readings) if cpu_readings else 0,
        "avg_memory": mean(memory_readings) if memory_readings else 0,
        "peak_cpu": max(cpu_readings) if cpu_readings else 0,
        "peak_memory": max(memory_readings) if memory_readings else 0
    }

def get_system_info():
    """Get system information including performance metrics."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "cpu_cores": psutil.cpu_count(),
        "memory_total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "memory_available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        "performance_metrics": collect_system_metrics()
    }

def kill_ollama_processes():
    """Kill all running Ollama processes."""
    logger.info("Attempting to kill Ollama processes")
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'ollama' in proc.info['name'].lower():
                psutil.Process(proc.info['pid']).terminate()
                killed_count += 1
                logger.info(f"Killed Ollama process with PID: {proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Error killing process: {e}")
    logger.info(f"Killed {killed_count} Ollama processes")
    time.sleep(2)

def start_ollama_server():
    """Start the Ollama server."""
    logger.info("Starting Ollama server")
    try:
        if sys.platform == 'win32':
            subprocess.Popen(['ollama', 'serve'], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(['ollama', 'serve'])
        logger.info("Ollama server start command issued")
        time.sleep(5)
        logger.info("Waited 5 seconds for server startup")
    except Exception as e:
        logger.error(f"Failed to start Ollama server: {e}", exc_info=True)
        raise

@dataclass
class Object:
    name: str
    confidence: float
    attributes: str

@dataclass
class ImageDescription:
    summary: str
    objects: List[Object]
    scene: str
    colors: List[str]
    time_of_day: str
    setting: str
    text_content: Optional[str] = None

@dataclass
class AnalysisResult:
    image_name: str
    analysis_time: float
    timestamp: str
    model: str
    system_info: Dict[str, Any]
    description: ImageDescription

# Initialize session state
if 'monitor_cpu' not in st.session_state:
    st.session_state.monitor_cpu = True
    logger.info("Initialized CPU monitoring session state")

# Load previous results if they exist
RESULTS_FILE = "analysis_results.json"
if 'analysis_results' not in st.session_state:
    try:
        with open(RESULTS_FILE, 'r') as f:
            st.session_state.analysis_results = json.load(f)
            logger.info("Loaded previous analysis results")
    except:
        st.session_state.analysis_results = None
        logger.info("No previous analysis results found")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Image Analysis", "README", "Source Code", "Logs", "JSON Output"])

with tab1:
    # CPU Monitor
    if st.session_state.monitor_cpu:
        metrics = collect_system_metrics()
        st.metric(
            "System Metrics",
            f"CPU: {metrics['current_cpu']}% (avg: {metrics['avg_cpu']:.1f}%)",
            f"Memory: {metrics['current_memory']}% (avg: {metrics['avg_memory']:.1f}%)"
        )

    st.title("Image Analysis with Llama Vision")
    st.write("This application uses Llama Vision to analyze images and provide structured descriptions.")

    # Ollama process management
    if st.button("Restart Ollama Server"):
        logger.info("Restart Ollama Server button clicked")
        with st.spinner("Restarting Ollama server..."):
            try:
                kill_ollama_processes()
                start_ollama_server()
                st.success("Ollama server restarted!")
                logger.info("Ollama server restart completed")
            except Exception as e:
                error_msg = f"Failed to restart Ollama server: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)

    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        logger.info(f"File uploaded: {uploaded_file.name}")
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            logger.info("Image loaded and displayed successfully")
            
            # Convert the uploaded file to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format)
            img_byte_arr = img_byte_arr.getvalue()
            logger.info("Image converted to bytes successfully")
            
            analyze_button = st.button("Analyze Image")
            
            if analyze_button:
                progress_text = st.empty()
                progress_text.text("Starting analysis...")
                
                try:
                    logger.info("Starting image analysis")
                    start_time = time.time()
                    
                    # Create schema for response format
                    schema = {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "objects": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "confidence": {"type": "number"},
                                        "attributes": {"type": "string"}
                                    }
                                }
                            },
                            "scene": {"type": "string"},
                            "colors": {"type": "array", "items": {"type": "string"}},
                            "time_of_day": {"type": "string", "enum": ["Morning", "Afternoon", "Evening", "Night"]},
                            "setting": {"type": "string", "enum": ["Indoor", "Outdoor", "Unknown"]},
                            "text_content": {"type": "string", "nullable": true}
                        }
                    }
                    
                    response = chat(
                        model='llama3.2-vision',
                        format=schema,
                        messages=[
                            {
                                'role': 'user',
                                'content': 'Analyze this image and describe what you see, including any objects, the scene, colors and any text you can detect.',
                                'images': [img_byte_arr],
                            },
                        ],
                        options={'temperature': 0},
                    )
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    logger.info("Received response from Ollama")
                    
                    # Stop Ollama server after analysis
                    kill_ollama_processes()
                    logger.info("Stopped Ollama server after analysis")
                    
                    # Parse response into ImageDescription
                    response_data = json.loads(response.message.content)
                    objects = [Object(**obj) for obj in response_data['objects']]
                    image_description = ImageDescription(
                        summary=response_data['summary'],
                        objects=objects,
                        scene=response_data['scene'],
                        colors=response_data['colors'],
                        time_of_day=response_data['time_of_day'],
                        setting=response_data['setting'],
                        text_content=response_data.get('text_content')
                    )
                    logger.info("Successfully parsed response data")
                    
                    # Create full analysis result
                    analysis_result = AnalysisResult(
                        image_name=uploaded_file.name,
                        analysis_time=processing_time,
                        timestamp=datetime.now().isoformat(),
                        model='llama3.2-vision',
                        system_info=get_system_info(),
                        description=image_description
                    )
                    
                    # Store results
                    st.session_state.analysis_results = asdict(analysis_result)
                    with open(RESULTS_FILE, 'w') as f:
                        json.dump(asdict(analysis_result), f, indent=2)
                    logger.info("Saved analysis results")
                    
                    progress_text.empty()
                    
                    # Display results
                    st.header("Analysis Results")
                    
                    st.subheader("Summary")
                    st.write(image_description.summary)
                    
                    st.subheader("Scene")
                    st.write(image_description.scene)
                    
                    st.subheader("Time of Day")
                    st.write(image_description.time_of_day)
                    
                    st.subheader("Setting")
                    st.write(image_description.setting)
                    
                    st.subheader("Colors")
                    st.write(", ".join(image_description.colors))
                    
                    st.subheader("Detected Objects")
                    for obj in image_description.objects:
                        st.write(f"- {obj.name} (Confidence: {obj.confidence:.2f})")
                        st.write(f"  Attributes: {obj.attributes}")
                    
                    if image_description.text_content:
                        st.subheader("Detected Text")
                        st.write(image_description.text_content)
                    
                    st.subheader("Processing Information")
                    st.write(f"Processing Time: {processing_time:.2f} seconds")
                    
                    logger.info("Successfully displayed all analysis results")
                        
                except Exception as e:
                    progress_text.empty()
                    error_msg = f"Analysis error: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)
                    st.error("Please ensure the Ollama server is running and the llama3.2-vision model is installed.")
        except Exception as e:
            error_msg = f"Image loading error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            st.error("Please ensure you've uploaded a valid image file.")

# Load and display content for other tabs only when they're active
if tab2.selectbox("", ["README"], key="readme_select"):
    st.title("README")
    try:
        with open("README.md", "r", encoding='utf-8') as f:
            readme_content = f.read()
        st.markdown(readme_content)
    except Exception as e:
        logger.error(f"Error loading README.md: {e}", exc_info=True)
        st.error("Error loading README.md file")

if tab3.selectbox("", ["Source Code"], key="source_select"):
    st.title("Source Code")
    try:
        with open(__file__, "r", encoding='utf-8') as f:
            source_code = f.read()
        st.code(source_code, language="python")
    except Exception as e:
        logger.error(f"Error loading source code: {e}", exc_info=True)
        st.error("Error loading source code file")

if tab4.selectbox("", ["Logs"], key="logs_select"):
    st.title("Application Logs")
    try:
        with open(log_file, "r", encoding='utf-8') as f:
            logs = f.read()
        st.text_area("Log Output", logs, height=400)
    except Exception as e:
        logger.error(f"Error loading log file: {e}", exc_info=True)
        st.error("Error loading log file")

if tab5.selectbox("", ["JSON Output"], key="json_select"):
    st.title("JSON Output")
    if st.session_state.analysis_results:
        # Display JSON
        st.json(st.session_state.analysis_results)
        
        # Create filename with image name and timestamp
        image_name = st.session_state.analysis_results['image_name'].rsplit('.', 1)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_name}_analysis_{timestamp}.json"
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(st.session_state.analysis_results, indent=2),
            file_name=filename,
            mime="application/json"
        )
    else:
        st.info("No analysis results available. Please analyze an image first.")

# Update CPU metrics every 5 seconds
if st.session_state.monitor_cpu:
    time.sleep(5)
    st.rerun()
