# Credit Card Default Detection 



## Code and Resources Used <br> 
**Jupiter Notebook**<br> 
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn<br> 
**Data:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients<br> 
**Article:** https://www.semanticscholar.org/paper/The-comparisons-of-data-mining-techniques-for-the-Yeh-Lien/1cacac4f0ea9fdff3cd88c151c94115a9fddcf33<br> 
## Problem Statement: 
This project aims to compare various data mining techniques for detecting defaults. In a well-developed financial system, crisis management is reactive, while risk prediction is proactive. The primary goal of risk prediction is to utilize financial data—such as business financial statements, customer transaction histories, and repayment records—to forecast business performance or assess the credit risk of individual customers, thereby mitigating potential damage and uncertainty.

When a client accepts a credit card from a bank or issuer, they agree to specific terms and conditions, including the obligation to make at least the minimum payment by the due date specified on their credit card statements. If the customer fails to meet this obligation, the issuer may classify the account as in default, impose a penalty rate, reduce the credit limit, and, in cases of significant delinquency, potentially close the account.

This project will explore and evaluate different data mining methodologies to enhance the detection of defaults, ultimately contributing to more effective risk management within financial systems.

## Data Collection:
The data can be downloaded from https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients, which contains the information of 30000 clients of a bank in Taiwan. 

## Data Cleaning: 
** Data Structure**
X1 to X5 are client personal information:
- X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
- X2: Gender (1 = male; 2 = female).
- X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
- X4: Marital status (1 = married; 2 = single; 3 = others).
- X5: Age (year).
- X6–X11: delay of the past payment: History of past payment. We tracked the past monthly payment records (from April to September 2005) as follows: X6 = the repayment status in September 2005; X7 = the repayment status in August 2005; …; X11 = the repayment status in April 2005. The measurement scale for the repayment status is: −1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; …; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
-  X12–X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September 2005; X13 = amount of bill statement in August 2005; …; X17 = amount of bill statement in April 2005.
-  X18–X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; …; X23 = amount paid in April, 2005.
-  Y is the target: credit card holders are defaulters or non-defaulters (1=yes, 0=no), and all features are integers.
To clean the data I made the following changes and created the following variables:
- drop column: "Unnamed:0" which does not contain any useful information.
- Columns X3 and X4 should have 4 and 3 categories, respectively, but I found more categories in the data frame. To correct this:
-- In the "Education" column (X3), replace categories 0, 5, and 6 with 4.
-- In the "Marital Status" column (X4), replace category 0 with 3.
## Exploratory Data Analysis
![Untitled](https://github.com/user-attachments/assets/78a42964-51cc-4d8b-90f9-6a3b3874d017)

The bar chart shows that we have a binary classification problem on a relatively unbalanced dataset. 

![Untitled](https://github.com/user-attachments/assets/8de0ca7f-7849-41da-95b5-696c9032fd9d)

![Untitled](https://github.com/user-attachments/assets/1c86d0f1-c0d9-4012-afb1-4b36eeff65e6)

A question that comes to mind is: Do the proportion of default is the same for both genders? To test this hypothesis, I performed a chi-square test. State the Hypotheses:
- Null Hypothesis (H0): The proportion of defaults and non-defaults is the same for men and women (i.e., gender and default status are independent).
- Alternative Hypothesis (H1): The proportion of defaults and non-defaults differs by gender (i.e., gender and default status are not independent).


Based on p-vales I reject the null hypothesis: There is a significant difference in the default rates between men and women.

![Untitled](https://github.com/user-attachments/assets/db949a2e-771a-406f-ab12-7b9bb69e493b)

The probability of non_default of clients between 26 and 40 is higher than the default of this age group.

![Untitled](https://github.com/user-attachments/assets/f8066451-d5ed-4be7-8787-540c4b48a5a7)

when the bank pays more than 132000$ credit to clients, it is more likely to default.

** Check Collinearity:**<br>

![Untitled](https://github.com/user-attachments/assets/0ebf5e16-4473-48df-94cb-2ee1955a7fd5)

![Untitled](https://github.com/user-attachments/assets/c775602b-d25d-4104-bd58-aa2f45129791)

The plots show the high correlation between columns X12, X13, X14, X15, X16, X17 so I drop columns with high collinearity , X12, X13, X14, X15, X16. 

### Data Preprocessing <br>
For data preprocessing, I performed the following steps:<br>
**Conversion of Categorical Data to Dummy Variables:** I transformed categorical features into dummy variables (one-hot encoding) to enable their inclusion in machine learning models. This process involves creating binary columns for each category in the original categorical feature.<br>
**Splitting Data for Modeling:** I divided the dataset into training and testing subsets. This separation ensures that the model is trained on one portion of the data and evaluated on a separate, unseen portion to assess its performance and generalizability.
**Normalization Using StandardScaler:** I applied normalization to standardize the features by removing the mean and scaling to unit variance. This step, performed using StandardScaler, ensures that each feature contributes equally to the model, improving convergence and performance.


















