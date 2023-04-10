import numpy as np
from scipy.signal import find_peaks
import wavio


START_CODE = "111"
END_CODE = "999"
SAMPLE_RATE = 44100

decimal_btoc = {chr(i): i for i in range(0,10)}


def define_symbols():
    symbols = {
        '0': (1000, 0.1),
        '1': (1200, 0.1),
        '2': (1400, 0.1),
        '3': (1600, 0.1),
        '4': (1800, 0.1),
        '5': (2000, 0.1),
        '6': (2200, 0.1),
        '7': (2400, 0.1),
        '8': (2600, 0.1),
        '9': (2800, 0.1),
    }
    return symbols


def encode_message(message):
    if isinstance(message, str):
        if not message or not set(message).issubset(set(define_symbols().keys())):
            raise ValueError(f"Invalid message: only {set(define_symbols().keys())} are allowed.")
        
    symbols = define_symbols()
    encoded_message = [symbols[str(int(symbol))] for symbol in message]
    print('encoded_message',encoded_message)
    return encoded_message

def read_sound_file(filename):
    sound_samples = wavio.read(filename).data
    sound_samples = sound_samples.astype(np.float32) / np.iinfo(np.int16).max
    return sound_samples

# Helper function to generate a time-domain signal from frequency-amplitude pairs
def generate_sound_samples(encoded_message, sample_rate=SAMPLE_RATE, duration=0.1):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sound_samples = []
    for freq, amp in encoded_message:
        sample = amp * np.sin(2 * np.pi * freq * t)
        sound_samples.append(sample)
    # add padding at the end
    sound_samples.append(np.zeros_like(sample))
    return sound_samples

# Helper function to process the recorded sound samples and identify peaks in the frequency domain
def process_sound_samples(sound_samples, sample_rate=SAMPLE_RATE, threshold=0.1):
    processed_samples = []
    for sample in sound_samples:
        # Compute the Fourier Transform of the sound sample
        ft = np.fft.rfft(sample)

        # Find the peaks in the magnitude spectrum
        peaks, _ = find_peaks(np.abs(ft), height=threshold)

        # Find the frequency and amplitude corresponding to the largest peak
        peak_idx = np.argmax(np.abs(ft[peaks]))
        frequency = peaks[peak_idx] * sample_rate / len(sample)
        amplitude = np.abs(ft[peaks[peak_idx]]) / len(sample) * 2

        processed_samples.append((frequency, amplitude))
    return processed_samples

# Update the decode_message function to work with processed sound samples
def decode_message(processed_samples):
    symbols = define_symbols()
    # Invert the symbols dictionary for decoding
    # symbols_inverse = {v: k for k, v in symbols.items()}
    
    decoded_message = []
    for sound_sample in processed_samples:
        frequency, amplitude = sound_sample
        print('frequency, amplitude',frequency, amplitude)
        # Decode the frequency-amplitude pair into the corresponding symbol
        if frequency < 100:
            continue
        symbol = None
        for key, value in symbols.items():
            freq_diff = abs(value[0] - frequency)
            amp_diff = abs(value[1] - amplitude)
            # print('key, value, freq_diff, amp_diff',key, value, freq_diff, amp_diff)
            # Set a threshold for frequency and amplitude differences
            if freq_diff < 50 and amp_diff < 0.1:
                symbol = key
                break

        if symbol is None:
            continue
            # raise ValueError("Invalid sound sample: unable to decode.")

        decoded_message.append(symbol)

    return ''.join(decoded_message)

def text_to_binary(text):
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def binary_to_text(binary):
    if len(binary) % 8 != 0:
        raise ValueError("Invalid binary input: must be a multiple of 8 bits.")
    
    text = ''.join(chr(int(binary[i:i + 8], 2)) for i in range(0, len(binary), 8))
    return text

def text_to_decimal(text):
    return ''.join([format(ord(char),'03d') for char in text])

def decimal_to_text(decimal, LDPC=False):
    if len(decimal) % 3 != 0:
        raise ValueError("Invalid decunak input: must be a multiple of 3 bits.")
    if LDPC:
        result = []
        for i in range(0, len(decimal), 6):
            word = decimal[i:i+3].tolist()
            result.append(chr(int("".join([str(x) for x in word]))))
        return "".join(result)
    return ''.join([chr(int(decimal[i:i+3])) for i in range(0, len(decimal), 3)])


def save_sound_samples_to_file(sound_samples, file_name, sample_rate=SAMPLE_RATE):
    # Concatenate sound samples into a single array
    sound_data = np.hstack(sound_samples)
    
    # Ensure sound data is in the range of int16
    sound_data = np.clip(sound_data, -1, 1) * 32767
    sound_data = sound_data.astype(np.int16)
    
    # Write sound data to a WAV file
    wavio.write(file_name, sound_data, sample_rate, sampwidth=2)



def process_realtime_samples(audio_data, symbols, sample_rate=SAMPLE_RATE, duration=0.1):
    # Prepare an empty list to store detected symbols
    detected_symbols = []

    # Calculate the number of samples for each symbol
    samples_per_symbol = int(sample_rate * duration)

    # Process audio data in chunks of samples_per_symbol
    for i in range(0, len(audio_data) - samples_per_symbol + 1, samples_per_symbol):
        chunk = audio_data[i:i + samples_per_symbol]
        # Process the chunk using the existing process_sound_samples function
        processed_samples = process_sound_samples([chunk], sample_rate)
        # Decode the processed samples
        try:
            decoded_binary_message = decode_message(processed_samples)
            print('decoded_binary_message',decoded_binary_message)
            detected_symbols.extend(list(decoded_binary_message))
        except ValueError as e:
            print(e)
            continue

    return detected_symbols



