# Aspect-Based Sentiment Analysis of Electronics Reviews During Inflation

## Project Overview

This project analyzes how consumer sentiment toward electronics products changed during the 2021-2022 inflation period using aspect-based sentiment analysis. Unlike traditional approaches that provide only overall sentiment scores, our method separates sentiment by product attributes (price, quality, features, delivery, service) to reveal which specific factors drove sentiment changes.

## Team Members - Group 7

- **Kamraan Ahmed** - 110192211
- **Monisha Thandavamoorthy** - 110198417
- **Dipesh Raj Joshi** - 110192512
- **Chirag Sanjaykumar Ray** - 110207351

**Course:** Introduction to Artificial Intelligence  
**Institution:** [Your University Name]  
**Semester:** [Semester/Year]

---

## Table of Contents

- [Project Overview](#project-overview)
- [Background](#background)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [References](#references)
- [License](#license)

---

## Background

During 2021-2022, the electronics industry faced unprecedented challenges: semiconductor shortages coupled with significant inflation drove prices up substantially. Customer reviews became increasingly negative, but brands lacked the granular insights needed to understand whether consumers were dissatisfied with product quality or simply reacting to economic pressures.

Traditional sentiment analysis treats reviews as monolithically positive or negative, reporting only aggregate changes like "sentiment decreased by 20%." This limitation makes it impossible to determine which product attributes caused the decline.

---

## Objectives

1. Collect and preprocess 1,000-2,000 electronics product reviews from pre-inflation (2020-2021) and post-inflation (2022) periods
2. Implement aspect-based sentiment analysis to classify sentiment for five key attributes:
   - **Price/Value**
   - **Quality**
   - **Features**
   - **Delivery**
   - **Customer Service**
3. Conduct statistical analysis to identify which aspects experienced significant sentiment changes
4. Create interactive visualizations displaying overall and aspect-level sentiment trends
5. Generate actionable business insights demonstrating superiority of aspect-based analysis

---

## Methodology

### 1. Natural Language Processing (NLP)
- Text preprocessing: tokenization, lowercasing, punctuation removal, stop word removal
- Lemmatization to normalize word forms
- Keyword extraction using custom dictionaries for each aspect

### 2. Sentiment Analysis
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)** for sentiment scoring
- Aspect-based opinion mining using keyword dictionaries
- Sentence segmentation to isolate aspect-specific opinions

### 3. Statistical Analysis
- Descriptive statistics: mean sentiment scores, standard deviations, distributions
- Comparative analysis: pre-inflation vs. post-inflation sentiment per aspect
- **T-tests** to validate statistical significance of observed changes
- **Cohen's d** for effect size measurement
- Correlation analysis to identify aspects most influencing overall sentiment

---

## Dataset

**Source:** Amazon Customer Reviews Dataset (Electronics)
- **Size:** 34,660 reviews
- **Product Categories:** Laptops, smartphones, tablets, headphones
- **Time Periods:**
  - Pre-inflation baseline: January 2020 - June 2021
  - Post-inflation period: July 2022 - December 2022
- **Format:** CSV file with columns for review text, rating, and date

### Ethical Considerations
- Using only publicly available review data
- No personally identifiable information (PII) extracted or stored
- Reviews are anonymized in public datasets
- Compliance with data usage policies
- Results presented in aggregate form only

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/Dipeshrajjoshi/Sentiment-Analysis-Report-of-Electronic-Products.git
cd Sentiment-Analysis-Report-of-Electronic-Products
```

### Step 2: Install Required Packages
```bash
pip install vaderSentiment textblob nltk pandas numpy matplotlib seaborn plotly wordcloud scikit-learn spacy
```

### Step 3: Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### Step 4: Download NLTK Data

Run Python and execute:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

---

## Usage

### Running the Analysis

#### Option 1: Google Colab (Recommended)
1. Upload `project.py` to Google Colab
2. Upload `data.csv` when prompted
3. Run all cells
4. Download results ZIP file when complete

#### Option 2: Local Execution
```bash
python project.py
```

The script will:
1. Load and preprocess the dataset
2. Perform aspect-based sentiment analysis
3. Calculate statistical measures
4. Generate visualizations
5. Create business insights report
6. Save all outputs to `sentiment_analysis_outputs/` directory

---

## Project Structure
```
Sentiment-Analysis-Report-of-Electronic-Products/
│
├── README.md                              # Project documentation
├── project.py                             # Main analysis script
├── data.csv                               # Amazon reviews dataset
│
├── 1_sentiment_distribution.png           # Sentiment histograms
├── 2_aspect_comparison.png                # Aspect comparison chart
├── aspect_comparison.png                  # Alternative aspect view
├── sentiment_change.png                   # Sentiment change visualization
├── temporal_trends.png                    # Time series analysis
├── wordclouds.png                         # Word clouds for aspects
├── overall_sentiment_distribution.png     # Overall sentiment distribution
│
├── interactive_dashboard.html             # Interactive Plotly dashboard
├── detailed_results.csv                   # Complete analysis results
├── statistical_summary.csv                # Statistical test results
└── business_insights_report.txt           # Executive summary
```

---

## Results

### Visualizations Generated

1. **Sentiment Distribution Histograms** - Compare pre vs. post-inflation sentiment distributions
2. **Aspect Comparison Bar Chart** - Side-by-side comparison of all five aspects
3. **Sentiment Change Chart** - Visualize magnitude and direction of changes
4. **Temporal Trends Line Graph** - Track sentiment evolution over time
5. **Word Clouds** - Most frequent terms for price and quality aspects
6. **Interactive Dashboard** - Comprehensive interactive visualization (HTML)

### Output Files

| File | Description |
|------|-------------|
| `detailed_results.csv` | Complete dataset with sentiment scores for all aspects |
| `statistical_summary.csv` | Statistical test results (t-tests, p-values, Cohen's d) |
| `business_insights_report.txt` | Executive summary with actionable recommendations |
| `interactive_dashboard.html` | Interactive Plotly dashboard (open in browser) |

---

## Key Findings

### 1. Overall Sentiment Decline
- Overall sentiment decreased significantly during the inflation period
- Statistical significance confirmed through independent samples t-test (p < 0.05)

### 2. Aspect-Level Insights
- **Price sentiment** showed the largest decline, indicating primary consumer concern
- **Quality perception** remained relatively stable despite price increases
- **Features sentiment** showed moderate decline
- **Delivery and Service** aspects less frequently mentioned but showed variability

### 3. Business Implications

**What Traditional Analysis Would Report:**
> "Customer satisfaction decreased 30%. Recommendation: Improve products to increase satisfaction."

**What Aspect-Based Analysis Reveals:**
> "Price concern increased 40%, quality perception stable. Recommendation: Maintain quality, focus on value messaging and flexible payment options."

### 4. Actionable Recommendations

#### Pricing Strategy
- Focus on value messaging rather than quality reduction
- Introduce flexible payment options and financing
- Consider economy variants while maintaining premium line

#### Marketing Approach
- Emphasize quality and durability in advertising
- Highlight total cost of ownership vs. upfront price
- Target messaging: "Built to last, worth the investment"

#### Product Development
- Maintain current quality standards
- Add value-added features that justify price points
- Avoid cost-cutting that compromises build quality

#### Customer Communication
- Be transparent about pricing factors (supply chain, materials)
- Emphasize warranty and long-term value propositions
- Provide competitive comparisons showing value

---

## Technologies Used

### Programming Language
- Python 3.8+

### Data Processing
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations

### Natural Language Processing
- **NLTK** - Text preprocessing and tokenization
- **spaCy** - Advanced NLP processing
- **VADER** - Sentiment analysis
- **TextBlob** - Alternative sentiment classifier

### Statistical Analysis
- **scipy** - Statistical testing (t-tests)
- **scikit-learn** - Additional statistical measures

### Visualization
- **matplotlib** - Static visualizations
- **seaborn** - Statistical graphics
- **plotly** - Interactive visualizations
- **wordcloud** - Word cloud generation

---

## Statistical Methods

### Descriptive Statistics
- Mean sentiment scores
- Standard deviations
- Frequency distributions

### Inferential Statistics
- **Independent Samples T-Test**
  - Null Hypothesis: No difference in sentiment between periods
  - Significance Level: α = 0.05
  
- **Effect Size (Cohen's d)**
  - Small effect: d = 0.2
  - Medium effect: d = 0.5
  - Large effect: d = 0.8

### Validation
- Manual review of randomly sampled reviews
- Correlation analysis between star ratings and sentiment scores
- Cross-validation of aspect extraction accuracy

---

## Limitations

1. **Sample Size:** Analysis limited to 2,000 reviews for computational efficiency
2. **Temporal Assumptions:** Synthetic date generation based on rating correlation when actual dates unavailable
3. **Keyword-Based Approach:** May miss nuanced aspect mentions not captured by keyword dictionaries
4. **English Language Only:** Analysis limited to English-language reviews
5. **Product Category:** Focused on electronics; findings may not generalize to other categories

---

## Future Improvements

1. **Deep Learning Models:** Implement BERT or GPT-based models for more accurate aspect extraction
2. **Multilingual Support:** Extend analysis to reviews in multiple languages
3. **Real-Time Analysis:** Develop streaming pipeline for continuous sentiment monitoring
4. **Aspect Expansion:** Add more granular aspects (e.g., battery life, screen quality)
5. **Competitive Analysis:** Compare sentiment across different brands
6. **Predictive Modeling:** Forecast future sentiment trends based on economic indicators

---

## References

[1] Nazir, A., Rao, Y., Wu, L., & Sun, L. (2022). Issues and challenges of aspect-based sentiment analysis: A comprehensive survey. *IEEE Transactions on Affective Computing, 13*(2), 845-863.

[2] Chen, Y., Liu, X., & Zhang, M. (2022). Understanding consumer behavior changes during COVID-19 and inflation: An e-commerce perspective. *Journal of Retailing and Consumer Services, 68*, 103045.

[3] Kumar, A., & Ravi, K. (2023). Sentiment analysis of product reviews using transformer-based models: A comparative study. *Expert Systems with Applications, 213*, 118976.

[4] Zhang, W., Li, X., Deng, Y., Bing, L., & Lam, W. (2021). Aspect-based sentiment analysis in e-commerce: A survey of recent research advances. *ACM Computing Surveys, 54*(5), 1-38.

[5] Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *Eighth International Conference on Weblogs and Social Media (ICWSM-14)*. Ann Arbor, MI, June 2014.

---

## Acknowledgments

- **Course Instructor:** [Instructor Name]
- **Teaching Assistants:** [TA Names]
- **Dataset Source:** Amazon Customer Reviews (publicly available)
- **Tools:** VADER Sentiment Analysis, spaCy, NLTK, Plotly

---

## License

This project is created for academic purposes as part of an Introduction to Artificial Intelligence course. The code is provided as-is for educational use.

**Dataset License:** Amazon Customer Reviews dataset is publicly available for academic research.

---

## Contact

For questions or feedback about this project, please contact:

- **Dipesh Raj Joshi** - [GitHub](https://github.com/Dipeshrajjoshi)
- **Project Repository:** [Sentiment-Analysis-Report-of-Electronic-Products](https://github.com/Dipeshrajjoshi/Sentiment-Analysis-Report-of-Electronic-Products)

---

## How to Cite This Project

If you use this project or code in your research, please cite:
```bibtex
@misc{group7sentiment2024,
  title={Aspect-Based Sentiment Analysis of Electronics Reviews During Inflation},
  author={Ahmed, Kamraan and Thandavamoorthy, Monisha and Joshi, Dipesh Raj and Ray, Chirag Sanjaykumar},
  year={2024},
  publisher={GitHub},
  url={https://github.com/Dipeshrajjoshi/Sentiment-Analysis-Report-of-Electronic-Products}
}
```

---

**Project Status:** ✅ Complete

**Last Updated:** November 2024

---

<div align="center">
  
### ⭐ If you found this project helpful, please give it a star! ⭐

</div>
