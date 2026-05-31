#CrossFit Madrid - Top Performer Predictor

## Project Overview

This project aims to help entrepreneurs, investors, and fitness business owners evaluate the potential success of a new CrossFit gym in Madrid using Machine Learning.

The application predicts the probability that a new gym will become a **Top Performer** within its district and provides actionable recommendations to improve its chances of success.

By combining market research, demographic information, business indicators, and predictive analytics, the tool supports data-driven decision-making before launching a new fitness business.

---

## Business Problem

Opening a new gym involves significant uncertainty. Factors such as location, pricing strategy, local demographics, competition, services offered, and class variety can directly impact business performance.

This project addresses the following question:

> Can we predict whether a new CrossFit gym will become a top performer before opening?

The goal is to transform historical business data into strategic insights that help reduce risk and improve investment decisions.

---

## Dataset

The dataset contains information from CrossFit and functional training gyms located in Madrid, enriched with demographic and district-level information.

### Main Features

- Gym name
- District
- Google rating
- Number of reviews
- Price information
- Available classes
- Available services
- Population density
- Median income
- Median age
- Competition metrics
- Geographic information

---

## Target Variable

A gym is classified as a **Top Performer** when:

- Google Rating ≥ 4.8
- Number of Reviews ≥ 75th percentile within its district

This definition identifies gyms that achieve both strong customer satisfaction and high market visibility.

---

## Methodology

### 1. Data Collection

Data was collected and consolidated from multiple sources, including:

- Google Maps
- Public demographic datasets
- District-level statistics

### 2. Data Cleaning

The following preprocessing steps were applied:

- Missing value treatment
- Text normalization
- Duplicate handling
- Data type correction

### 3. Feature Engineering

Several business-oriented features were created:

- Gyms per 10,000 inhabitants
- Population density indicators
- Price-to-income ratio
- Premium pricing indicators
- Competition metrics
- Service availability indicators
- Class diversity indicators

### 4. Machine Learning

The final model was trained using:

- XGBoost Classifier
- Oversampling techniques for class balancing
- Train/Test Split validation

The model outputs the probability that a gym configuration becomes a top performer.

---

## Application Features

The Streamlit application allows users to simulate a new gym and receive an instant prediction.

### User Inputs

- District selection
- Monthly membership price
- Classes offered
- Services offered

### Model Outputs

- Probability of becoming a Top Performer
- Success assessment
- Business recommendations
- Improvement opportunities

---

## Dashboard Workflow

1. Select the characteristics of a hypothetical gym.
2. Submit the information through the interface.
3. The trained model evaluates the configuration.
4. A success probability is generated.
5. Personalized recommendations are displayed.

---

## Technologies Used

### Data Analysis

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-Learn
- XGBoost

### Web Application

- Streamlit

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Jupyter Notebook
- VS Code

---

## Project Structure

```text
Business-Performance-App/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── crossfit_gyms_madrid_enriched.xlsx
│
├── notebooks/
│   ├── data_collection.ipynb
│   ├── data_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb
│
├── models/
│   └── xgboost_model.pkl
│
└── assets/
    └── images/
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Javierpe10/Business-Performance-App.git
cd Business-Performance-App
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Business Value

This project helps entrepreneurs and investors:

- Evaluate market opportunities
- Reduce decision-making uncertainty
- Identify key success drivers
- Optimize pricing strategies
- Assess local competition
- Improve resource allocation

Rather than relying solely on intuition, users can make decisions supported by historical data and predictive analytics.

---

## Key Insights

The analysis suggests that gym success is influenced by a combination of:

- Local demographics
- Market saturation
- Pricing strategy
- Service offerings
- Class diversity
- Customer engagement

Successful gyms typically balance competitive positioning with strong customer experience and visibility.

---

## Future Improvements

Potential future enhancements include:

- Expansion to other cities
- Real-time Google Places integration
- Explainable AI using SHAP values
- Financial forecasting module
- Competitive benchmarking system
- Automated market opportunity scoring

---

## Author

**Javier Peñaloza**

Data Science & AI
---

## License

This project was developed for educational and portfolio purposes as part of the Ironhack Data Analytics & AI Program.
