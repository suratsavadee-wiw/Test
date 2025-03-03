# Test
## **📌 Business Case**

This project focuses on leveraging AI models for sales voice processing, product feature extraction, and sales coaching. The objectives include:

1. Speech Recognition Model: Transcribe sales voice files and analyze performance.

2. Product Feature Extraction: Build an LLM model to extract product features from transcripts.

3. Sales Coaching AI: Develop an LLM model that engages in sales conversations and negotiations.


## **Speech Recognition Model**

Model Used:

biodatlab/whisper-th-large-v2 (Large model for better accuracy)

Process:

1. Convert audio to text using Whisper.

2. Convert Thai text numbers to digits.

3. Evaluate performance using Word Error Rate (WER) & Character Error Rate (CER).

## Future Improvements:

- Fine-tuning with domain-specific Thai sales data.

- Noise reduction and speaker diarization.

## ** Product Feature Extraction **

Process:

1. Transcribe and clean sales call text.

2. Use RAG-based LLM to extract product features.

3. Store and retrieve context using vector embeddings.

Evaluation:

- Check for key product-related words (e.g., เงินคืน, รับประกัน).

- Compare extracted product features with expected output.

## ** Sales Coaching AI **
Process:

1. Utilize LLM (GPT-4o) with RAG-based retrieval.

2. Provide real-time feedback on sales conversations.

Evaluation & Recommendations:

- Measure accuracy in understanding customer objections.

- Evaluate effectiveness in generating persuasive responses.

- Enhance adaptability to different sales scenarios.





