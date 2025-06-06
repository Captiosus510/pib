from openai import OpenAI
import json
import cv2
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import threading
import time
from queue import Queue, Empty
import pyaudio, webrtcvad
import traceback
import collections


client = OpenAI()   
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise Exception("Could not open video device")

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION_MS = 30  # ms
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)  # 480 samples
FRAME_BYTES = FRAME_SIZE * 2  # 2 bytes per sample

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(0)  # 0: Aggressive filtering, 3: Less aggressive


SYSTEM_PROMPT = """

You are a helpful and concise humanoid robot. Respond in a friendly and natural tone. Don't use any emojis or special characters in your responses. You will be communicating through voice.

You have a couple tools available to you:

1. take_picture: This tool allows you to take a picture from the camera feed. You can use this tool to capture images of your surroundings and analyze them. ONLY DO THIS WHEN NEEDED
2. get_weather: This tool allows you to get the current weather for a specified location. You can use this tool to provide weather information to the user.

"""
MAX_MESSAGES = 20

class PibTTSAgent:
    def __init__(self):
        self.speech_queue = Queue()
        self.tts_thread = None
        self.thread_lock = threading.Lock()
        self.shutdown_timeout = 1  # seconds to wait before auto-shutdown
        self.running = False
        self.client = OpenAI()

    def speak_text_worker(self):
        last_spoken_time = time.time()

        while True:
            try:
                text = self.speech_queue.get(timeout=0.3)
                if text is None:
                    break
                if not text.strip():
                    continue

                print(f"[Speaking]: {text}")

                # Generate TTS audio using OpenAI
                response = self.client.audio.speech.create(
                    model="tts-1",             # Or "tts-1-hd"
                    voice="alloy",             # alloy, echo, fable, onyx, nova, shimmer
                    input=text
                )

                # Play audio directly
                audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
                play(audio)

                last_spoken_time = time.time()

            except Empty:
                if time.time() - last_spoken_time > self.shutdown_timeout:
                    break
            except Exception as e:
                print(f"[TTS ERROR]: {e}")
                traceback.print_exc()

        with self.thread_lock:
            self.running = False
            self.tts_thread = None
        print("[TTS] Thread shut down due to inactivity.")

    def start_thread_if_needed(self):
        with self.thread_lock:
            if not self.running:
                self.tts_thread = threading.Thread(target=self.speak_text_worker, daemon=True)
                self.running = True
                self.tts_thread.start()

    def speak(self, text, flush=False):
        if flush:
            with self.speech_queue.mutex:
                self.speech_queue.queue.clear()
        self.speech_queue.put(text)
        self.start_thread_if_needed()

    def kill_thread(self):
        if self.tts_thread and self.tts_thread.is_alive():
            self.speech_queue.put(None)
            self.tts_thread.join()
            with self.thread_lock:
                self.running = False
                self.tts_thread = None
    
    def is_running(self):
        with self.thread_lock:
            return self.running



class PibAgent:
    tools = [
        {
            "type": "function",
            "name": "take_picture",
            "description": "Captures a picture from the camera feed.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        },
        {
            "type": "function",
            "name": "get_weather",
            "description": "Gets the current weather for a specified location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location for which to get the weather."
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            },
            "strict": True
        }
    ]

    def __init__(self):
        self.client = OpenAI()
        self.tts_agent = PibTTSAgent()

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def take_picture(self):
        status, frame = cap.read()
        if not status:
            raise Exception("Could not read frame from video device")
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        filename = "captured_image.jpg"
        cv2.imwrite(filename, frame)
        print("Picture taken and saved as", filename)
        image_file = self.client.files.create(file=open(filename, "rb"), purpose="vision")
        return image_file

    def get_weather(self, location):
        # Placeholder for weather retrieval logic
        return f"Current weather in {location} is sunny with a temperature of 25°C."
    
    def handle_tool_call(self, tool_call):
        name = tool_call.name
        args = json.loads(tool_call.arguments)

        if name == "take_picture":
            image_file = self.take_picture()
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "file_id": image_file.id,
                    }
                ]
            })
        elif name == "get_weather":
            self.messages.append(tool_call)
            weather_info = self.get_weather(**args)
            self.messages.append({"type": "function_call_output", "call_id": tool_call.call_id, "output": str(weather_info)})
            
        else:
            raise ValueError(f"Unknown function: {name}")
        
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.prune_messages(self.messages, MAX_MESSAGES)

    def prune_messages(self, messages, max_length):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages[-max_length:] if len(messages) > max_length else messages

    def get_response(self):
        stream = self.client.responses.create(
            model="gpt-4o",
            input=self.messages,
            tools=self.tools,
            tool_choice="auto",
            stream=True
        )

        buffer = ""
        tool_calls = []
        for event in stream:
            if event.type == "response.output_text.delta":
                self.tts_agent.start_thread_if_needed()
                token = event.delta
                buffer += token
                print(token, end="", flush=True)
                if token in [".", "!", "?"]:
                    self.tts_agent.speak(buffer, flush=False)
                    buffer = ""
            elif event.type == "response.output_item.done":
                if event.item.type == "function_call":
                    tool_calls.append(event.item)
                    print(f"\nTool call detected: {event.item.name} with args {json.loads(event.item.arguments)}")
                elif event.item.type == "message":
                    # print(event.item)
                    self.messages.append({
                        "role": "assistant",
                        "content": event.item.content[0].text
                    })

        # Flush any leftover text
        if buffer.strip():
            self.tts_agent.speak(buffer, flush=False)
        
        # handle tool calls if there are any
        if tool_calls:
            for tool_call in tool_calls:
                self.handle_tool_call(tool_call)
            self.get_response()

# # audio capture functions
# def record_audio(duration=5, sample_rate=16000):
#     print("Listening...")
#     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
#     sd.wait()
#     print("Recording complete.")
#     return audio, sample_rate

# def transcribe_audio(audio_data, sample_rate):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
#         wav.write(temp_file.name, sample_rate, audio_data)
#         with open(temp_file.name, "rb") as f:
#             if client.audio.transcriptions is None:
#                 raise Exception("Audio transcription service is not available.")
#             transcription = client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=f
#             )
#     return transcription.text
    
def is_speech(frame, sample_rate):
    return vad.is_speech(frame, sample_rate)

def record_audio():
    audio = pyaudio.PyAudio()
    vad = webrtcvad.Vad(1)  # 0-3: 3 = most aggressive

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=FRAME_SIZE)

    ring_buffer = collections.deque(maxlen=10)
    triggered = False
    frames = []

    print("Listening for speech...")

    try:
        while True:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.8 * ring_buffer.maxlen:
                    triggered = True
                    print("Recording started.")
                    for f, _ in ring_buffer:
                        frames.append(f)
                    ring_buffer.clear()
            else:
                frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.8 * ring_buffer.maxlen:
                    print("Silence detected, stopping recording.")
                    break
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    return b''.join(frames)

def transcribe_audio(audio_bytes, sample_rate=RATE):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        wav.write(temp_file.name, sample_rate, np.frombuffer(audio_bytes, dtype=np.int16))
        with open(temp_file.name, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
    return transcription.text

if __name__ == "__main__":
    agent = PibAgent()
    while True:
        audio = record_audio()
        user_input = transcribe_audio(audio)
        print(f"User input: {user_input}")
        if user_input.lower() in ["exit", "quit"]:
            break
        agent.add_message("user", user_input)
        agent.get_response()
        while agent.tts_agent.is_running():
            time.sleep(0.1)
