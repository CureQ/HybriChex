# Classification Web Application

This project is a web application that incorporates a classification model. It allows users to interact with the model through a user-friendly interface.

## Project Structure

```
classification-web-app
├── app
│   ├── static
│   │   ├── css
│   │   │   └── styles.css
│   │   └── js
│   │       └── scripts.js
│   ├── templates
│   │   ├── base.html
│   │   └── index.html
│   ├── __init__.py
│   ├── routes.py
│   └── model
│       ├── classifier.py
│       └── __init__.py
├── requirements.txt
├── run.py
└── README.md
```

## Technologies Used

- **Programming Language**: Python
- **Web Framework**: Flask
- **Machine Learning Libraries**: scikit-learn, pandas, numpy
- **Frontend Technologies**: HTML, CSS, JavaScript

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd classification-web-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python run.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:5000` to access the application.

## Usage

- The main page allows users to input data for classification.
- The application processes the input and displays the classification results.
