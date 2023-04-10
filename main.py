import numpy as np
from listen_for_sound_sample import listen_for_sound_samples
from sound_protocol import SAMPLE_RATE, decimal_to_text, define_symbols, encode_message, decode_message, generate_sound_samples, process_realtime_samples, process_sound_samples, save_sound_samples_to_file,text_to_binary, binary_to_text,START_CODE,END_CODE, text_to_decimal,read_sound_file
from LDPC import encode_ldpc_base10,decode_ldpc,H,G
import re

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',"--message", type=str, default="Hello, I am an AI language model created by OpenAI.")
    parser.add_argument("-g", "--generate", action="store_true")
    parser.add_argument("-l", "--listen", action="store_true")
    args = parser.parse_args()
    if args.generate:
        if args.message:
            message = args.message
        else:
            message = "Hello, I am an AI language model created by OpenAI."
        decimal_encoded_message = START_CODE + START_CODE + text_to_decimal(message) + END_CODE + END_CODE
        print('decimal_encoded_message',decimal_encoded_message)
        decimal_message = []
        print('len(decimal_encoded_message)',len(decimal_encoded_message))
        for i in range(0,len(decimal_encoded_message),3):
            word = np.array(list(decimal_encoded_message[i:i+3]),dtype=int)
            # print(word)
            # print(encode_ldpc_base10(word,G))
            decimal_message.append(encode_ldpc_base10(word,G))
        decimal_message = np.concatenate(decimal_message)
        print('decimal_message',decimal_message)
        # decimal_message = np.array([encode_ldpc_base10(decimal_encoded_message[i:i+3]) for i in range(len(decimal_encoded_message)-3,3)],dtype=int)
        encoded_message = encode_message(decimal_message)
        sound_samples = generate_sound_samples(encoded_message)
        save_sound_samples_to_file(sound_samples, "medium_length_message.wav")
    elif args.listen:

        # Listen for a sound message and store the recorded samples
        # recorded_samples = []

        # def record_audio(data):
        #     recorded_samples.append(data)

        # listening_seconds = (len(read_sound_file("medium_length_message.wav")) / SAMPLE_RATE)+3
        # print("Listening for sound message...")
        # listen_for_sound_samples(record_audio, duration=listening_seconds)
        # # print('recorded_samples',recorded_samples[:2])
        # # Convert recorded_samples to a NumPy array
        # recorded_samples = np.concatenate(recorded_samples) 
        # print('recorded_samples',recorded_samples[:2])
        # recorded_samples = recorded_samples.astype(np.float32) / np.iinfo(np.int16).max

        # # Process and decode the recorded samples
        # detected_symbols = process_realtime_samples(recorded_samples, define_symbols())
        
        # detected_symbols_str = ''.join(detected_symbols)
        detected_symbols_str = '111789111789072511101050108658108658111789044022032795073315032795097979109452032795097979110985032795065284073315032795108658097979110985103658117583097979103658101050032795109452111789100256101050108658032795099577114181101050097979116789101050100256032795098773121418032795079119112583101050110985065284073315046620999321999321'
        print('detected_symbols_str',detected_symbols_str)
        start_group = re.search(r'111...111',detected_symbols_str)
        end_group = re.search(r'999...999',detected_symbols_str)
        print('start_group',start_group)
        print('end_group',end_group)
        if start_group and end_group:
            start_index = start_group.span()[1]
            end_index = end_group.span()[0]
            if start_index < end_index:
                decoded_decimal_message = detected_symbols_str[start_index+3:end_index]
                decoded_message = []
                for i in range(0,len(decoded_decimal_message),6):
                    codeword_base10 = np.array(list(decoded_decimal_message[i:i+6]),dtype=int)
                    decoded_message.append(decode_ldpc(codeword_base10,H,20))
                decoded_message = np.concatenate(decoded_message)
                print('decoded_message',decoded_message)
                final_message = decimal_to_text(decoded_message,LDPC=True)
            else:
                final_message = "No message detected"
        print("Decoded message:", final_message)