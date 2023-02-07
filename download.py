# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import whisperx

def download_model():
    model = whisperx.load_model("large-v2")
    for language_code in ['fr', 'de', 'ru']:
        model_a, _ = whisperx.load_align_model(language_code=language_code, device='cpu')

if __name__ == "__main__":
    download_model()