# ReCognize

# ReCognize
In this project, we aim to build a system that screens for Alzheimer's disease using audio data. The system will analyze audio recordings of patients' speech and use machine learning techniques to classify them as either having Alzheimer's disease or not. The project is divided into several key components, including data collection, preprocessing, feature extraction, model training, and evaluation.
## Project Structure
```plaintext
ReCognize/
├── backend/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── context/
├── model/
├── README.md
```

We have built a model using machine learning techniques to classify audio recordings of patients' speech as either having Alzheimer's disease or not. The model is trained on a dataset of audio recordings and their corresponding labels. The model is saved in the `model` directory and can be loaded for inference.

In the `frontend` directory, we have built a web application using React and Vite. The application allows users to upload audio recordings and get predictions from the model. The frontend communicates with the backend to send audio data and receive predictions.

In the backend directory, we have built a Flask API that serves the model and handles requests from the frontend. The API is responsible for loading the model, processing audio data, and returning predictions.

Team Members: Soujanya C. Aryal, Ariel McGee, Mihir Walvekar, Barakah Quader, Siddhant Karmali
