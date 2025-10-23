import speech_recognition as sr
import io

def transcribe_audio(audio_bytes: bytes):
    """
    Transcribes audio bytes into text using Google's Web Speech API.

    Args:
        audio_bytes: The raw audio data in WAV format as bytes.

    Returns:
        A tuple of (str, bool) where the first element is the transcribed text
        or an error message, and the second is a success flag.
    """
    recognizer = sr.Recognizer()
    if not audio_bytes:
        return "No audio received. Please try recording again.", False
    try:
        # Use an in-memory buffer to treat the bytes as a file
        with io.BytesIO(audio_bytes) as wav_file:
            with sr.AudioFile(wav_file) as source:
                audio_data = recognizer.record(source)
        
        # Recognize speech using Google's free web API
        text = recognizer.recognize_google(audio_data)
        return text, True
        
    except sr.UnknownValueError:
        return "Could not understand the audio. Please try speaking clearly.", False
    except sr.RequestError as e:
        return f"API unavailable or unresponsive: {e}", False
    except Exception as e:
        return f"An unexpected error occurred during transcription: {e}", False