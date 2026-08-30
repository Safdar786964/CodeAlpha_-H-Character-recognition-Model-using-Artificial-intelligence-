# Handwritten Character Recognition

A Streamlit application for recognizing handwritten characters using a pre-trained TensorFlow/Keras CNN model trained on the EMNIST Balanced dataset.

## Project Structure

- `app.py`: Streamlit application
- `handwritten_character_cnn.keras`: Trained CNN model
- `emnist-balanced-mapping.txt`: Class-to-character mapping
- `utils/preprocessing.py`: Image preprocessing helpers
- `assets/profile.jpg`: Developer profile image

## Run the app

```bash
cd handwritten-character-recognition
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The model is used as-is without retraining.
- The EMNIST balanced mapping file is used to convert model class indices into human-readable characters.
