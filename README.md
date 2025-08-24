# E-commerce Analytics and Recommendation System

# Project files
Models -
Presenation - https://docs.google.com/presentation/d/1iJbhhwCp9oOaWg0jNp-vrvUzzF5J9N67/edit?usp=sharing&ouid=115392145190850421540&rtpof=true&sd=true 

## Project Overview

This project aims to analyze user behavior on an e-commerce platform, detect anomalous users, predict product properties, and generate personalized recommendations. The workflow is divided into:

- **Exploratory Data Analysis (EDA)**
- **Property Prediction**
- **Anomaly Detection**
- **Recommendation System**
- **Visualization and Reporting**

The analyses answer key business questions such as user conversion, product popularity, engagement, and cart abandonment.

---

## 1. Business Questions and Insights

### 1.1 Where are we losing users in the conversion funnel?

- **Users at each stage:**
  - View: `view_users`
  - Add to Cart: `cart_users`
  - Purchase: `purchase_users`
  
- **Conversion rates:**
  - View → Cart: `view_to_cart`
  - Cart → Purchase: `cart_to_purchase`
  - View → Purchase: `view_to_purchase`

**Insight:** Most users drop off between viewing and adding to cart or between adding to cart and purchasing.

**Visualization:** Conversion funnel using Plotly `px.funnel`.

---

### 1.2 What items are most frequently abandoned in carts?

- **Metric:** `cart_counts - purchase_counts` per item.
- **Top 10 abandoned products:** Highlights products frequently added but not purchased.

**Insight:** Identifies potential friction points; can guide pricing, promotions, or product information improvements.

**Visualization:** Bar chart of abandoned items.

---

### 1.3 Which users are most engaged?

- **Metric:** Number of events per user.
- **Top 10 most active users:** Users with highest interaction counts.

**Insight:** Engagement correlates with potential lifetime value and loyalty.

**Visualization:** Bar chart of top 10 users by activity.

---

### 1.4 Which users are most valuable?

- **Metric:** Number of purchases per user.
- **Top 10 buyers:** Users with the highest number of transactions.

**Insight:** These users are high-priority for retention campaigns and personalized offers.

**Visualization:** Bar chart of top buyers.

---

### 1.5 What are the most viewed products?

- **Metric:** Number of unique viewers per product.
- **Top 10 most viewed products:** Helps identify popular items for marketing or inventory management.

**Visualization:** Bar chart of top viewed products.

---

### 1.6 Which users add the most items to cart but don’t buy?

- **Metric:** Carted items minus purchased items per user.
- **Top 10 cart abandoners:** Users with highest abandonment.

**Insight:** Highlights users with high purchase intent but low completion, useful for remarketing campaigns.

**Visualization:** Bar chart of top cart abandoners.

---

### 1.7 What is the overall conversion rate of the site?

- **Metric:** Percentage of unique viewers who made a purchase.
- **Conversion Rate:** `conversion_rate %`

**Insight:** Provides a high-level performance metric of the platform.

---

## 2. Methodology

### 2.1 Data Loading and Preprocessing

- Data loaded from multiple CSVs: events, item properties, category tree.
- Timestamps converted to datetime, missing values handled.
- User features created: session length, event rate, unique items, time patterns, and day-of-week statistics.

### 2.2 Property Prediction

- Guided pseudo-positives used to simulate scarce labels.
- Features generated per user-item-property combination.
- Models trained and evaluated using AUC metrics.

### 2.3 Anomaly Detection

- **Statistical methods:** Z-score and IQR for outlier detection.
- **ML methods:** Isolation Forest and Local Outlier Factor.
- **Composite anomaly score:** Normalized 0–10 score combining methods.

### 2.4 Recommendation System

- **Collaborative Filtering (CF):** SVD-based matrix factorization.
- **Content-Based Filtering (CBF):** Item properties used to score products.
- **Hybrid approach:** Weighted combination of CF and CBF.
- **Evaluation:** Precision, recall, and F1 metrics for top-N recommendations.

### 2.5 Visualization

- Conversion funnel, user engagement, top products, anomaly scores, and recommendation performance displayed using Plotly.

---

## 3. How to Run

1. **Install dependencies:**

```bash
pip install pandas numpy scipy scikit-learn plotly implicit
Load Data
Update the file paths in the DataLoader class to point to your local CSV files.

Run Analysis
Execute the workflow including:

Exploratory Data Analysis (EDA)

Data preprocessing

Property prediction

Anomaly detection

Recommendation system generation

Generate Dashboards
Use the Visualizer class to produce visualizations for:

User behavior

Conversion funnel

Anomaly scores

Recommendation system performance

4. Key Insights

Most users drop off at the view → cart and cart → purchase stages.

Certain products are frequently abandoned in carts, indicating potential friction points.

Top engaged users and top buyers can be targeted for retention or marketing campaigns.

Overall site conversion rate provides a benchmark for platform performance.

Hybrid recommendations improve personalized user experiences.

5. Future Work

Improve anomaly detection by incorporating temporal behavior patterns.

Enhance the recommendation system using session-based or sequence-aware models.

Integrate additional user demographics and item metadata for better personalization.

Implement automated dashboards for continuous monitoring.
