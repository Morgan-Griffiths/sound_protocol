import numpy as np
import pytest
from LDPC import compute_initial_LLRs_base10
from sound_protocol import define_symbols, encode_message, decode_message, generate_sound_samples, process_sound_samples,text_to_binary, binary_to_text


def test_define_symbols():
    symbols = define_symbols()
    assert isinstance(symbols, dict)
    assert '0' in symbols
    assert '1' in symbols
    assert len(symbols) == 2


def test_encode_message():
    symbols = define_symbols()
    message = "0101"
    encoded_message = encode_message(message)
    
    assert isinstance(encoded_message, list)
    assert len(encoded_message) == len(message)
    
    for symbol, encoding in zip(message, encoded_message):
        assert encoding == symbols[symbol]


@pytest.mark.parametrize("invalid_message", ["", "0012", "01a1"])
def test_encode_message_invalid_input(invalid_message):
    with pytest.raises(ValueError):
        encode_message(invalid_message)

@pytest.mark.parametrize("message", ["0", "1", "01", "1010"])
def test_decode_message(message):
    encoded_message = encode_message(message)
    sound_samples = generate_sound_samples(encoded_message)
    processed_samples = process_sound_samples(sound_samples)
    decoded_message = decode_message(processed_samples)
    assert decoded_message == message
    

@pytest.mark.parametrize("text", ["Hello", "OpenAI", "ChatGPT"])
def test_text_to_binary_and_binary_to_text(text):
    binary = text_to_binary(text)
    decoded_text = binary_to_text(binary)
    assert decoded_text == text


def test_compute_initial_LLRs_base10():
    received_codeword = np.array([1.0, -3.0, 4.0])
    noise_variance = 2.0
    expected_LLRs = np.array([
        [  0., 0.25,   0.,-0.75,  -2.,-3.75,-6.,-8.75 -12., -15.75],
        [  0.,-1.75,-4.,-6.75 -10., -13.75 -18., -22.75 -28., -33.75],
        [0., 1.75, 3., 3.75, 4., 3.75, 3., 1.75, 0.,-2.25]
    ])

    LLRs = compute_initial_LLRs_base10(received_codeword, noise_variance)

    # Check if the computed LLRs are close to the expected LLRs
    assert np.allclose(LLRs, expected_LLRs, rtol=1e-8, atol=1e-8), f"Expected {expected_LLRs}, but got {LLRs}"

