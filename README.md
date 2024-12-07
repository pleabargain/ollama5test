# Image Analysis with Llama Vision

This is a Streamlit application that demonstrates the use of Llama Vision model for structured image analysis. It's designed to test the new JSON output capabilities of the Llama Vision model by extracting structured information from images.




## ⚠️ Important Notes

- This application must be run **locally** due to its dependencies on the Ollama server
- Requires the Llama Vision model to be installed locally
- **Very CPU intensive** - expect significant processing time for each image
- This is a demonstration of the new JSON output tool functionality

## Requirements

- Python 3.7+
- Ollama server running locally with the llama3.2-vision model installed
- Significant CPU resources for image processing
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Ensure Ollama is running with the llama3.2-vision model installed:
```bash
ollama pull llama3.2-vision
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```
2. Open your web browser to the displayed local URL (typically http://localhost:8501)
3. Upload an image using the file uploader
4. Wait for the analysis to complete (this may take several minutes due to CPU processing)

## Data Model

The application extracts structured information using the following model:

### Object
- name (str): Name of the detected object
- confidence (float): Confidence score of the detection
- attributes (str): Additional attributes of the object

### ImageDescription
- summary (str): Overall description of the image
- objects (list[Object]): List of detected objects
- scene (str): Description of the scene
- colors (list[str]): Dominant colors in the image
- time_of_day: One of ['Morning', 'Afternoon', 'Evening', 'Night']
- setting: One of ['Indoor', 'Outdoor', 'Unknown']
- text_content (optional): Any text detected in the image

## Performance Considerations

- The analysis process is computationally intensive and may take several minutes per image
- Performance depends heavily on your CPU capabilities
- Consider closing other CPU-intensive applications while using this tool
- The first analysis may take longer as the model loads into memory

## Technical Details

This application uses:
- Streamlit for the web interface
- Ollama's Llama Vision model for image analysis
- Pydantic for data validation and JSON schema generation
- PIL (Python Imaging Library) for image processing

## Application Interface

The application features several tabs for easy navigation and transparency:

### Main Tab
The primary interface where you can upload and analyze images.

### Help Tab
Contains this documentation for quick reference while using the application.

### Source Code Tab
Displays the application's source code (main.py) for transparency and educational purposes.

### Requirements Tab
Shows the complete list of Python package dependencies required to run this application.

### Logging Tab
Provides real-time visibility into:
- Function calls and their execution
- Processing status and outcomes
- System performance metrics
- Error tracking and debugging information

### Test Log Tab
Displays the results of unit tests, ensuring code reliability and proper functionality.
