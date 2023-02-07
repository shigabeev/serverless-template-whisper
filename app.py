import torch
import whisperx
import os
import uuid
import base64
import urllib.request
from io import BytesIO

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    model = whisperx.load_model("large-v2")

def download_from_link(url):
    filename = str(uuid.uuid4()) + ".mp3"
    response = urllib.request.urlopen(url)
    urllib.request.urlretrieve(url, filename)
    return filename

def save_to_file(mp3BytesString):
    filename = str(uuid.uuid4()) + ".mp3"
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open(filename,'wb') as file:
        file.write(mp3Bytes.getbuffer())
    return filename

def whisperx_align(audio_path, text_segments, language_code, device='cuda'):
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    result_aligned = whisperx.align(text_segments, model_a, metadata, audio_path, device)
    return result_aligned

    
# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    """
    inputs: json with the following structure.
    {
        "mp3BytesString":"1231abc12345f...fac3158"
        "return_alignments":true
    }
    
    or
    
    {
        "url":"https://example.com/audio.mp3"
    }
    
    input format only mp3
    """
    global model
    
    if 'mp3BytesString' in model_inputs.keys():
        mp3BytesString = model_inputs['mp3BytesString']
        filename = save_to_file(mp3BytesString)
    elif 'url' in model_inputs.keys():
        url = model_inputs['url']
        filename = download_from_link(url)
    else:
        return {'message': "No input provided"}
    
    # Run the model
    transcription_results = model.transcribe(filename)
    if model_inputs.get('return_alignments', False):
        transcription_results = whisperx_align(audio_path=filename,
                       text_segments=transcription_results["segments"],
                       language_code=transcription_results["language"]) 
    
    os.remove(filename)
    # Return the results as a dictionary
    return transcription_results
