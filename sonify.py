#!/usr/bin/env python3
"""
Math Sound Codec - গণিতকে সাউন্ড দিয়ে এনকোড ও ডিকোড করুন
সংখ্যা → শব্দ (সাইন ওয়েভ, নির্দিষ্ট ফ্রিকোয়েন্সি)
শব্দ → সংখ্যা (FFT দিয়ে ফ্রিকোয়েন্সি সনাক্ত)
"""

import numpy as np
import wave
import struct
import argparse
import os
from scipy.fft import fft, fftfreq
from scipy.io.wavfile import read as wav_read

# ---------- কনফিগারেশন ----------
DURATION_PER_DIGIT = 0.3        # সেকেন্ড, প্রতি সংখ্যার জন্য
SAMPLE_RATE = 44100             # Hz
SILENCE_DURATION = 0.05         # সংখ্যার মধ্যে চুপ (সেকেন্ড)
AMP = 0.5                       # আয়তন (0-1)

# 0-9 সংখ্যার জন্য ফ্রিকোয়েন্সি ম্যাপ (Hz)
FREQ_MAP = {
    '0': 200, '1': 300, '2': 400, '3': 500, '4': 600,
    '5': 700, '6': 800, '7': 900, '8': 1000, '9': 1100,
    # দশমিক পয়েন্ট বা অন্যান্য চিহ্ন চাইলে বাড়ানো যাবে
}
# উল্টো ম্যাপ (ডিকোডিংয়ের জন্য)
REV_FREQ_MAP = {v: k for k, v in FREQ_MAP.items()}

# ফ্রিকোয়েন্সি শনাক্তের সহনশীলতা (Hz)
TOLERANCE = 15

def encode_numbers_to_sound(numbers_str, output_wav):
    """
    "12345" টাইপ স্ট্রিং ইনপুট নিয়ে WAV ফাইল বানায়।
    প্রতিটি সংখ্যা আলাদা ফ্রিকোয়েন্সির সাইন তরঙ্গে রূপান্তরিত হয়।
    """
    audio_data = []
    for digit in numbers_str:
        if digit not in FREQ_MAP:
            raise ValueError(f"অবৈধ অক্ষর: '{digit}'। শুধু 0-9 অনুমোদিত।")
        freq = FREQ_MAP[digit]
        t = np.linspace(0, DURATION_PER_DIGIT, int(SAMPLE_RATE * DURATION_PER_DIGIT), endpoint=False)
        wave_segment = AMP * np.sin(2 * np.pi * freq * t)
        audio_data.extend(wave_segment)
        # সাইলেন্স (সংখ্যার মধ্যে গ্যাপ)
        silence = np.zeros(int(SAMPLE_RATE * SILENCE_DURATION))
        audio_data.extend(silence)
    
    audio_array = np.array(audio_data, dtype=np.float32)
    # স্বাভাবিককরণ (clipping এড়াতে)
    if np.max(np.abs(audio_array)) > 0:
        audio_array = audio_array / np.max(np.abs(audio_array)) * AMP
    # 16-bit PCM এ রূপান্তর
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    with wave.open(output_wav, 'wb') as wf:
        wf.setnchannels(1)          # মনো
        wf.setsampwidth(2)          # 2 বাইট = 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    print(f"✅ এনকোড সম্পন্ন: {output_wav}")

def decode_sound_to_numbers(wav_file):
    """
    WAV ফাইল থেকে সংখ্যা বের করে।
    প্রতিটি সেগমেন্টের প্রভাবশালী ফ্রিকোয়েন্সি মেপে ম্যাপিং টেবিলের সাথে মেলানো হয়।
    """
    try:
        samplerate, data = wav_read(wav_file)
        if data.dtype != np.int16:
            data = data.astype(np.int16)
        # যদি স্টিরিও হয়, মনোতে কনভার্ট
        if len(data.shape) > 1:
            data = np.mean(data, axis=1).astype(np.int16)
    except Exception as e:
        raise RuntimeError(f"WAV ফাইল পড়তে সমস্যা: {e}")
    
    # ফ্লোটে কনভার্ট এবং নরমালাইজ
    data_float = data / 32767.0
    
    # উইন্ডো সাইজ (একটি ডিজিটের নমুনা + সাইলেন্সের ভগ্নাংশ)
    samples_per_digit = int(samplerate * DURATION_PER_DIGIT)
    samples_silence = int(samplerate * SILENCE_DURATION)
    hop = samples_per_digit + samples_silence
    
    decoded_digits = []
    pos = 0
    while pos + samples_per_digit <= len(data_float):
        segment = data_float[pos:pos + samples_per_digit]
        # হ্যানিং উইন্ডো ফাংশন (স্পেকট্রাল লিকেজ কমানোর জন্য)
        window = np.hanning(len(segment))
        windowed = segment * window
        
        # FFT
        N = len(windowed)
        if N == 0:
            break
        yf = fft(windowed)
        xf = fftfreq(N, 1/samplerate)
        magnitude = np.abs(yf[:N//2])
        # শুধু ইতিবাচক ফ্রিকোয়েন্সি
        freqs = xf[:N//2]
        
        # সর্বোচ্চ ম্যাগনিটিউডের ফ্রিকোয়েন্সি সনাক্ত
        dominant_freq = freqs[np.argmax(magnitude)]
        
        # নিকটতম ম্যাপিং খোঁজা
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
            decoded_digits.append('?')   # শনাক্ত করতে পারেনি
        
        pos += hop
    
    decoded_str = ''.join(decoded_digits)
    return decoded_str

def main():
    parser = argparse.ArgumentParser(description="গণিতকে সাউন্ড দিয়ে এনকোড/ডিকোড করুন")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # এনকোড কমান্ড
    encode_parser = subparsers.add_parser("encode", help="সংখ্যা সাউন্ডে কনভার্ট")
    encode_parser.add_argument("digits", type=str, help="সংখ্যার স্ট্রিং (যেমন 12345)")
    encode_parser.add_argument("-o", "--output", default="encoded.wav", help="আউটপুট WAV ফাইলের নাম")
    
    # ডিকোড কমান্ড
    decode_parser = subparsers.add_parser("decode", help="সাউন্ড থেকে সংখ্যা বের করুন")
    decode_parser.add_argument("wavfile", type=str, help="ইনপুট WAV ফাইল")
    
    args = parser.parse_args()
    
    if args.command == "encode":
        encode_numbers_to_sound(args.digits, args.output)
    elif args.command == "decode":
        result = decode_sound_to_numbers(args.wavfile)
        print(f"🔊 ডিকোড করা সংখ্যা: {result}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
