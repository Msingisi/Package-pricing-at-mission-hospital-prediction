# Package Pricing at Mission Hospital Prediction

## Business problem
To develop a predictive analytics model that accurately forecasts the cost of medical treatments and procedures. This model aims to enhance financial transparency for patients, optimize pricing strategies for healthcare providers, and contribute to the overall efficiency of the healthcare system by enabling data-driven decision-making. By leveraging historical data and machine learning algorithms, the project seeks to provide reliable price estimates that reflect the complexity and variability of individual patient care, ultimately leading to improved patient satisfaction and trust in healthcare services.

## Resources

- **Project Idea:** https://365datascience.com/career-advice/top-10-data-science-project-ideas/#2
- **Data Source:** https://github.com/rashidesai24/Package-Pricing-at-Mission-Hospital/blob/main/Package%20Pricing%20at%20Mission%20Hospital%20-%20Data%20Supplement.xlsx

## Data Preparation 

- **Age Categories:** <10, 11 25, 26 50, 50 and above (as per domain expertise) 
- **BP Ranges:** Low, Normal, High, Critical (as per the medical charts)
- **BMI:** Underweight, Normal, Overweight, Obese (as per the medical charts)
- **Hemoglobin:** normal - Female 12 to 15.5, Men 13 to 17.5, any value outside these limits will be abnormal
- **Urea:** normal - 7 to 20 mg/dl any value outside these limits will be "abrnormal"
- **Creatinine Categories:** Age <3 & creatinine: 0.3-0.7 -> Normal, Age: 3-18 & creatinine: 0.5-1.0 -> Normal, Age >18 & Female & creatinine: 0.6 - 1.1 ->	Normal,
Age > 18 & Male & creatinine: 0.9 - 1.3 ->	Normal, Else: Abnormal

## Methods

- Exploratory data analysis
- Bivariate analysis
- Multivariate correlation
- Model deployment

## Quick glance at the results
Correlation between the features.
![heatmap](assets/heatmap.png)

Top features of gradient boosting with the best hyperparameters.
![Top 10](assets/top10.png)

Least useful features of gradient boosting with the best hyperparameters.
![Bottom 10](assets/bottom10.png)

Top 3 models (with default parameters)

| Model with the best hyperparameter     	                | RMSE  |
|-------------------	                                    |------------------	|
| Gradient Boosting     	                                    | 37314.11 	    |
| Random Forest   	                                            | 42510.84 	            |
| Extra Trees               	                        | 44862.48	            |




