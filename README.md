# MyCrop - Smart Crop Recommendation System

MyCrop is a machine learning-powered web application designed to help farmers and agricultural planners choose the most suitable crop based on soil and environmental conditions. The system analyzes factors such as nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH level, and rainfall to recommend the best crop.

This project combines Flask for the web interface with a trained Random Forest Classifier to provide practical crop suggestions for improved productivity and better farming decisions.

## Author

Abhinav B

## Project Summary

Agriculture depends heavily on the right crop selection, which is influenced by soil health and weather conditions. This project solves that challenge by using machine learning to predict a suitable crop for a given set of environmental conditions.

The application provides:
- Crop recommendation based on real agricultural input values
- User-friendly web interface for entering field conditions
- Recommended crop output along with supporting agronomic guidance
- Simple deployment using Flask

## Features

- Smart crop prediction using machine learning
- Input form for soil and climate parameters
- Clean and responsive frontend design
- Built-in model serialization with pickle
- Suitable for demo and learning projects

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- Pickle
- HTML/CSS/Bootstrap
- JavaScript

## Dataset

The project uses agricultural data stored in the `Data-processed` folder. The dataset contains crop-related features such as:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall
- Crop label

## Machine Learning Model

A Random Forest Classifier is trained on the dataset to classify the best crop based on the provided inputs. The trained model is saved and later loaded for prediction in the Flask app.

## Project Structure

```text
crop/
├── app_run.py
├── model.py
├── README.md
├── Data-processed/
│   ├── crop.csv
│   └── Crop_rec.csv
├── models/
│   └── model.pkl
├── static/
│   ├── css/
│   ├── images/
│   └── scripts/
├── templates/
│   ├── Crop_recommend.html
│   ├── Crop_result.html
│   ├── index.html
│   └── layout.html
└── .gitignore
```

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/mycrop.git
   cd mycrop
   ```

2. Create and activate a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not present, install the required packages manually:
   ```bash
   pip install flask pandas numpy scikit-learn matplotlib seaborn
   ```

4. Run the app
   ```bash
   python app_run.py
   ```

5. Open in your browser
   ```text
   http://127.0.0.1:5000/
   ```

## Usage

- Open the home page
- Navigate to the crop recommendation section
- Enter the required field values
- Submit the form to view the predicted crop and suggestions

## Notes

This project is ideal for educational use, prototype development, and agricultural decision support. It demonstrates how data science and machine learning can be applied to real-world agricultural challenges.

## Future Improvements

- Add crop yield prediction
- Include more accurate region-based recommendations
- Add user login and saved recommendations
- Improve UI/UX for farmers
- Expand dataset with more crops and real-time weather data
