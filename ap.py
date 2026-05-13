import os
import uuid
import numpy as np
import wave
import struct
from scipy.fft import fft, fftfreq
from scipy.io.wavfile import read as wav_read
from flask import Flask, render_template, request, send_file, jsonify
import tempfile

app = Flask(__name__)

# ---------- কনফিগারেশন (পূর্বের মতো) ----------
DURATION_PER_DIGIT = 0.3        # সেকেন্ড
SAMPLE_RATE = 44100
SILENCE_DURATION = 0.05
AMP = 0.5

FREQ_MAP = {
    '0': 200, '1': 300, '2': 400, '3': 500, '4': 600,
    '5': 700, '6': 800, '7': 900, '8': 1000, '9': 1100,
}
REV_FREQ_MAP = {v: k for k, v in FREQ_MAP.items()}
TOLERANCE = 15

def encode_numbers_to_bytes(numbers_str):
    """শুধু সংখ্যা নেবে, মেমরিতে অডিও বাইট রিটার্ন করবে।"""
    if not numbers_str.isdigit():
        raise ValueError("শুধু 0-9 সংখ্যা অনুমোদিত")
    audio_data = []
    for digit in numbers_str:
        freq = FREQ_MAP[digit]
        t = np.linspace(0, DURATION_PER_DIGIT, int(SAMPLE_RATE * DURATION_PER_DIGIT), endpoint=False)
        wave_segment = AMP * np.sin(2 * np.pi * freq * t)
        audio_data.extend(wave_segment)
        silence = np.zeros(int(SAMPLE_RATE * SILENCE_DURATION))
        audio_data.extend(silence)
    audio_array = np.array(audio_data, dtype=np.float32)
    if np.max(np.abs(audio_array)) > 0:
        audio_array = audio_array / np.max(np.abs(audio_array)) * AMP
    audio_int16 = (audio_array * 32767).astype(np.int16)

    # WAV বাইটে রূপান্তর
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        with wave.open(tmpfile, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        tmp_path = tmpfile.name
    return tmp_path

def decode_sound_from_file(filepath):
    """WAV ফাইল পাথ নিয়ে ডিকোড করে সংখ্যার স্ট্রিং ফেরায়।"""
    try:
        samplerate, data = wav_read(filepath)
        if data.dtype != np.int16:
            data = data.astype(np.int16)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1).astype(np.int16)
    except Exception as e:
        raise RuntimeError(f"WAV পড়তে সমস্যা: {e}")
    
    data_float = data / 32767.0
    samples_per_digit = int(samplerate * DURATION_PER_DIGIT)
    samples_silence = int(samplerate * SILENCE_DURATION)
    hop = samples_per_digit + samples_silence

    decoded_digits = []
    pos = 0
    while pos + samples_per_digit <= len(data_float):
        segment = data_float[pos:pos + samples_per_digit]
        window = np.hanning(len(segment))
        windowed = segment * window
        N = len(windowed)
        if N == 0:
            break
        yf = fft(windowed)
        xf = fftfreq(N, 1/samplerate)
        magnitude = np.abs(yf[:N//2])
        freqs = xf[:N//2]
        if len(freqs) == 0:
            break
        dominant_freq = freqs[np.argmax(magnitude)]
        matched_digit = None
        min_diff = TOLERANCE
        for freq, digit in REV_FREQ_MAP.items():
            diff = abs(dominant_freq - freq)
            if diff < min_diff:
                min_diff = diff
                matched_digit = digit
        if matched_digit is not None:
            decoded_digits.append(matched_digit)
        else:
            decoded_digits.append('?')
        pos += hop
    return ''.join(decoded_digits)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode_route():
    data = request.get_json()
    numbers = data.get('numbers', '')
    if not numbers.isdigit():
        return jsonify({'error': 'শুধু 0-9 সংখ্যা দিন'}), 400
    try:
        wav_path = encode_numbers_to_bytes(numbers)
        return send_file(wav_path, as_attachment=True, download_name=f'{numbers}.wav', mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.unlink(wav_path)

@app.route('/decode', methods=['POST'])
def decode_route():
    if 'audiofile' not in request.files:
        return jsonify({'error': 'কোনো ফাইল পাঠানো হয়নি'}), 400
    file = request.files['audiofile']
    if file.filename == '':
        return jsonify({'error': 'ফাইল নির্বাচন করুন'}), 400
    if not file.filename.endswith('.wav'):
        return jsonify({'error': 'শুধু .wav ফাইল সমর্থিত'}), 400
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        result = decode_sound_from_file(tmp_path)
        return jsonify({'decoded': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
