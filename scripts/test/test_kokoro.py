"""
test_kokoro.py

This script tests the Kokoro TTS pipeline by converting a sample text to speech
and saving the resulting audio as a WAV file in the "/home/funboy/virtflux/outputs/tts" folder.
It processes the generator output from Kokoro, ensuring that each audio segment
is flattened to a one-dimensional array before concatenation.
"""

import os
import numpy as np
import soundfile as sf
from kokoro import KPipeline

def flatten_segment(segment):
    """
    Attempts to convert an audio segment to a 1D NumPy array.

    If a direct conversion fails (due to inhomogeneous shapes), it will:
      - Iterate over the items in the segment,
      - Flatten each item individually,
      - Concatenate all flattened items.

    Returns:
      A 1D NumPy array containing the flattened audio data.

    Raises:
      ValueError: If no valid items can be flattened.
    """
    try:
        # Try to convert the entire segment to a NumPy array directly.
        seg_array = np.array(segment, dtype=float)
        return seg_array.flatten()
    except Exception:
        flat_items = []
        for item in segment:
            try:
                item_array = np.array(item, dtype=float).flatten()
                flat_items.append(item_array)
            except Exception as inner_error:
                continue
        if not flat_items:
            raise ValueError("Unable to flatten segment; no valid items found.")
        return np.concatenate(flat_items)

def main():
    # Define the destination folder for audio output.
    output_dir = "/home/funboy/virtflux/outputs/tts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output file path.
    output_file = os.path.join(output_dir, "test_kokoro.wav")
    
    # Initialize the Kokoro pipeline with American English (lang_code "a").
    # (Do not pass a voice parameter during initialization.)
    pipeline = KPipeline(lang_code="a")
    
    # Define a long sample text.
    text = (
        "AIR3 EXPLAINS: WHAT IS AI AND HOW DOES IT WORK? "
        "1) What is Artificial Intelligence and How Does It Work? "
        "Artificial Intelligence (AI) is the ability of a machine to perform tasks that would normally require human intelligence, "
        "such as voice recognition, problem-solving, and data analysis. It operates using advanced algorithms and neural networks that process "
        "vast amounts of information, learning from data to improve its performance over time. AI utilizes machine learning to adapt to new situations "
        "and deep learning to analyze complex data, making it increasingly sophisticated and efficient. "
        "2) What Is the Difference Between AI, Machine Learning, and Deep Learning? "
        "AI is a broad concept that includes any technology capable of simulating human intelligence. Machine learning is a subset of AI that enables systems "
        "to learn from data without being explicitly programmed. Deep learning, on the other hand, is an advanced form of machine learning that uses artificial "
        "neural networks inspired by the human brain, making it possible for AI to perform tasks like voice recognition, computer vision, and natural language processing. "
        "3) Can AI Think on Its Own, or Does It Need to Be Programmed? "
        "AI does not have a mind of its own and cannot think independently like a human. It is always programmed to analyze data and identify patterns, "
        "making decisions based on the information it receives. Even the most advanced AI models, such as ChatGPT, function through algorithms that learn from data "
        "but lack consciousness, emotions, or autonomous creativity. "
        "4) What Are the Different Types of Artificial Intelligence? "
        "There are three main categories of AI: Narrow AI (Weak AI), General AI (Strong AI), and Superintelligence. Currently, only narrow AI exists, while general AI "
        "remains a long-term goal. "
        "5) What Are the Most Common Uses of AI in Everyday Life? "
        "AI is integrated into many aspects of our daily lives. Examples include voice assistants like Siri and Alexa, recommendation algorithms on Netflix and Spotify, "
        "social media personalization, facial recognition in smartphones and security systems, and autonomous vehicles. "
        "6) How Are AI Models Trained? "
        "AI algorithms are trained using vast amounts of data through techniques such as supervised learning, unsupervised learning, and reinforcement learning. "
        "7) Can AI Completely Replace Human Jobs? "
        "Although AI is automating many tasks, jobs that require creativity, empathy, and critical thinking are unlikely to be fully replaced. Instead, AI will transform these roles. "
        "8) Is AI Truly Intelligent, or Does It Just Follow Instructions? "
        "AI processes information and follows complex mathematical models to make decisions but lacks self-awareness or the ability to think independently. "
        "9) What Are the Advantages and Disadvantages of AI? "
        "Advantages include increased productivity, enhanced data analysis, and personalized experiences, while disadvantages involve potential job displacement, data bias, and privacy concerns. "
        "10) What Is the Difference Between Strong AI and Weak AI? "
        "Weak AI is designed for specific tasks, whereas strong AI would be capable of general reasoning, though no true strong AI exists today. "
        "11) What Is an AI Agent? "
        "An AI agent is a system designed to perceive its environment, make decisions, and take actions to achieve specific goals. "
        "AI agents are used in various applications, such as virtual assistants, chatbots, and autonomous vehicles."
    )
    
    # Generate audio by calling the pipeline with the desired voice.
    # This returns a generator of audio segments.
    audio_generator = pipeline(text, voice="af_heart")
    
    # Consume the generator to retrieve all audio segments.
    segments = list(audio_generator)
    if not segments:
        raise ValueError("No audio segments were returned from the Kokoro pipeline.")
    
    # Process each segment using the flatten helper.
    flattened_segments = []
    for seg in segments:
        try:
            flat_seg = flatten_segment(seg)
            flattened_segments.append(flat_seg)
        except Exception as e:
            print(f"Warning: Failed to flatten a segment: {e}")
    
    if not flattened_segments:
        raise ValueError("No valid audio segments after flattening.")
    
    # Debug: Print the shape of each flattened segment.
    print("Flattened audio segment shapes:", [seg.shape for seg in flattened_segments])
    
    # Concatenate all flattened segments along the sample dimension.
    try:
        audio_data = np.concatenate(flattened_segments, axis=0)
    except Exception as e:
        raise ValueError(f"Failed to concatenate audio segments: {e}")
    
    # Ensure that the final audio data is 2D (samples, channels). If mono, reshape to [samples, 1].
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)
    
    # Define the sample rate (Kokoro's default is typically 24000 Hz).
    sample_rate = 24000
    
    # Save the audio data to a WAV file.
    sf.write(output_file, audio_data, sample_rate)
    print(f"Audio file successfully saved to: {output_file}")

if __name__ == "__main__":
    main()
