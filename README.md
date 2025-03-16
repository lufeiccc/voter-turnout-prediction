# **Voter Turnout Analysis in Pennsylvania Counties**

Analyzing voter turnout trends using linear and panel regression models

## **Project Overview**

This project examines voter turnout in Pennsylvania counties for the 2016 and 2020 elections using Python-based linear and panel regression models. The analysis integrates socioeconomic and demographic data to identify key factors influencing voter participation across different regions and time periods.

### **Key Features:**

- Data collection from **APIs, web scraping, and CSV files**, followed by data cleaning and structuring into a unified dataset.
- Implementation of **OLS regression** to assess individual variable correlations with voter turnout.
- Application of **panel regression models** to account for temporal dependencies and control for county-specific effects.
- Visualization of trends using **Matplotlib, Seaborn, and Plotly** to highlight regional and demographic disparities in voter turnout.

---

## **Installation & Setup**

### **Environment Requirements**

This project is best run in a **Jupyter Notebook** or **Spyder** within an **Anaconda environment** for smooth package management.

### **Required Libraries**

Ensure you have the following Python libraries installed before running the scripts:

```bash
pip install pandas requests beautifulsoup4 numpy matplotlib seaborn plotly geopandas
pip install scikit-learn scipy statsmodels linearmodels
```

---

## **Project Structure**

### **1. Data Compilation (********`compiled_data.py`********)**

- Collects and processes data from multiple sources, including APIs, web scraping, and CSV files.
- Aggregates and cleans the dataset into a structured CSV format.

### **2. Regression Analysis (********`regression.py`********)**

- Performs **Ordinary Least Squares (OLS) regression** to identify statistically significant variables influencing voter turnout.
- Evaluates the strength of correlations between turnout rates and socioeconomic factors.

### **3. Panel Regression (********`p_panel.py`********)**

- Applies **panel regression models** to control for county-specific and temporal effects.
- Compares fixed effects and random effects models to determine the best fit for the data.

### **4. Visualization (********`visualization.py`********)**

- Generates key data visualizations, including:
  - **Voter turnout trends across counties and demographic groups**
  - **Urban/rural classification impact on turnout rates**
  - **Democratic vote share distribution maps for 2016 and 2020**
  - **Aggregated county correlation heatmaps**

---

## **How to Run the Project**

1. **Unzip the project folder** into your working directory.
2. **Run data compilation**
   ```bash
   python "1. compiled.py"
   ```
3. **Generate visualizations**
   ```bash
   python "2. visualization.py"
   ```
4. **Perform regression analysis**
   ```bash
   python "3. regression.py"
   ```
5. **Run panel regression model**
   ```bash
   python "4. panel.py"
   ```

---

## **Results & Findings**

- **Regression Analysis:**

  - OLS regression identifies key socioeconomic factors affecting voter turnout.
  - Panel regression provides a better model fit, with **BIC (-570.96 vs. -488.08)**, capturing temporal dependencies.

- **Key Insights from Visualizations:**

  - **Higher education levels** (Bachelor’s degree or higher) positively correlate with voter turnout (\~0.71).
  - **Lower income levels** (< \$50,000) show a **negative correlation** with turnout (-0.76).
  - **Urban vs. rural classification** reveals distinct voting patterns across regions.

---

## **Contributors**

This project was developed by:

- Shao Gu (Shawn) Lu
- Liufei Chen
- Zhixuan (Alex) Jiang
- Kanika Selvapandian

---

##
