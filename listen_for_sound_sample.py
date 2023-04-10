import pyaudio
import numpy as np
import time

def listen_for_sound_samples(callback, duration=5, sample_rate=44100, chunk_size=1024):
    p = pyaudio.PyAudio()
    
    # Callback function to process audio stream
    def process_audio_stream(in_data, frame_count, time_info, status):
        callback(np.frombuffer(in_data, dtype=np.int16))
        return (in_data, pyaudio.paContinue)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size,
                    stream_callback=process_audio_stream)

    stream.start_stream()

    # Listen for the specified duration
    time.sleep(duration)

    stream.stop_stream()
    stream.close()

    p.terminate()
