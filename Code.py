import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # Import numpy

class FounderInvestorMatcher:
    def __init__(self, industry_weight=0.7, funding_weight=0.3):
        self.industry_weight = industry_weight
        self.funding_weight = funding_weight
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.scaler = MinMaxScaler()

    def preprocess_data(self, founders, investors):
        # Fill missing categorical values with "Unknown"
        founders['industry'] = founders['industry'].fillna("Unknown")
        investors['preferred_industry'] = investors['preferred_industry'].fillna("Unknown")
        
        # Fill missing numeric values with column means
        founders['funding_required'] = founders['funding_required'].fillna(founders['funding_required'].mean())
        investors['investment_max'] = investors['investment_max'].fillna(investors['investment_max'].mean())

        # Rename columns for consistency
        investors = investors.rename(columns={'preferred_industry': 'industry'})
        
        # Combine data for consistent one-hot encoding
        combined_industry = pd.concat([founders[['industry']], investors[['industry']]], axis=0)
        industry_encoded = self.encoder.fit_transform(combined_industry).toarray()
        founder_industry_encoded = industry_encoded[:len(founders)]
        investor_industry_encoded = industry_encoded[len(founders):]

        # Combine funding data for consistent scaling
        combined_funding = pd.concat([founders[['funding_required']], investors[['investment_max']]], axis=0)
        funding_normalized = self.scaler.fit_transform(combined_funding)
        founder_funding_normalized = funding_normalized[:len(founders)]
        investor_funding_normalized = funding_normalized[len(founders):]

        # Replace NaN values with 0 in normalized funding data - Addressing the root cause
        founder_funding_normalized = np.nan_to_num(founder_funding_normalized)  
        investor_funding_normalized = np.nan_to_num(investor_funding_normalized)

        return founder_industry_encoded, investor_industry_encoded, founder_funding_normalized, investor_funding_normalized

    def calculate_similarity(self, founder_encoded, investor_encoded):
        return cosine_similarity(founder_encoded, investor_encoded)

    def compute_match_scores(self, founders, investors):
        (founder_industry_encoded, investor_industry_encoded,
         founder_funding_normalized, investor_funding_normalized) = self.preprocess_data(founders, investors)

        industry_similarity = self.calculate_similarity(founder_industry_encoded, investor_industry_encoded)
        funding_similarity = self.calculate_similarity(founder_funding_normalized, investor_funding_normalized)

        # Compute weighted match score
        match_score = (self.industry_weight * industry_similarity +
                       self.funding_weight * funding_similarity)

        # Display match scores
        match_df = pd.DataFrame(match_score, 
                                columns=[f"Investor{i+1}" for i in range(len(investors))], 
                                index=[f"Founder{i+1}" for i in range(len(founders))])
        return match_df

# Sample data
founders = pd.DataFrame({
    'industry': ['Tech', 'Health', None],
    'stage': ['Seed', 'Series A', 'Pre-Seed'],
    'funding_required': [500000, 2000000, None]
})

investors = pd.DataFrame({
    'preferred_industry': ['Tech', 'Health', None],
    'investment_stage': ['Seed', 'Series B', 'Series A'],
    'investment_min': [100000, 500000, 100000],
    'investment_max': [1000000, 3000000, None]
})

matcher = FounderInvestorMatcher()
match_scores = matcher.compute_match_scores(founders, investors)
print(match_scores)

# Heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(match_scores, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f")
plt.title("Founder-Investor Match Scores")
plt.show()
