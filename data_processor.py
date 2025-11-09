# data_processor.py (Updated: Drop YoY_Growth)

import pandas as pd
import os
from functools import cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional: Try kagglehub if installed, else local
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("kagglehub not installed; use local 'cosmetics_reviews.csv'.")

@cache
def load_and_process():
    # Load data: local first, then kagglehub
    local_csv = 'cosmetics_reviews.csv'
    if os.path.exists(local_csv):
        df_raw = pd.read_csv(local_csv)
        print(f"Loaded local {df_raw.shape[0]} reviews.")
    elif KAGGLE_AVAILABLE:
        path = kagglehub.dataset_download("jithinanievarghese/cosmetics-and-beauty-products-reviews-top-brands")
        csv_path = next((os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')), None)
        if not csv_path:
            raise ValueError("No CSV found in dataset.")
        df_raw = pd.read_csv(csv_path)
        print(f"Loaded Kaggle {df_raw.shape[0]} reviews.")
    else:
        raise ValueError("No local CSV found and kagglehub not installed. Download from Kaggle and save as 'cosmetics_reviews.csv'.")
    
    print("Columns:", df_raw.columns.tolist())
    
    # Improved dynamic column finder for product name: prioritize 'title' or 'name'
    product_col = None
    if 'product_title' in df_raw.columns:
        product_col = 'product_title'
    elif 'product_name' in df_raw.columns:
        product_col = 'product_name'
    else:
        candidates = [col for col in df_raw.columns if 'product' in col.lower() and ('title' in col.lower() or 'name' in col.lower())]
        if candidates:
            product_col = candidates[0]
        else:
            # Fallback, but warn
            product_candidates = [col for col in df_raw.columns if 'product' in col.lower()]
            if product_candidates:
                product_col = product_candidates[0]
                print(f"Warning: Using fallback product col: {product_col} (may be ID, not name)")
            else:
                raise ValueError("No product column found.")
    print(f"Using product column: {product_col}")
    
    # Dynamically find rating column: prioritize 'review_rating'
    rating_col = None
    if 'review_rating' in df_raw.columns:
        rating_col = 'review_rating'
    elif 'rating' in df_raw.columns:
        rating_col = 'rating'
    else:
        for col in df_raw.columns:
            if 'rating' in col.lower():
                rating_col = col
                break
    if rating_col is None:
        raise ValueError("No rating column found.")
    print(f"Using rating column: {rating_col}")
    
    # Dynamically find review text column: prioritize 'review_text'
    review_col = None
    if 'review_text' in df_raw.columns:
        review_col = 'review_text'
    else:
        for col in df_raw.columns:
            if 'review' in col.lower() and 'text' in col.lower():
                review_col = col
                break
    if review_col is None:
        review_col = 'review_title' if 'review_title' in df_raw.columns else None
    if review_col is None:
        print("No review text column; using empty for claims.")
    print(f"Using review column: {review_col}")
    
    # Dynamically find date column: prioritize 'review_date'
    date_col = None
    if 'review_date' in df_raw.columns:
        date_col = 'review_date'
    else:
        for col in df_raw.columns:
            if 'date' in col.lower():
                date_col = col
                break
    print(f"Using date column: {date_col}")
    
    # Brand column (optional): prioritize 'brand_name'
    brand_col = None
    if 'brand_name' in df_raw.columns:
        brand_col = 'brand_name'
    else:
        brand_col = next((col for col in df_raw.columns if 'brand' in col.lower()), None)
    print(f"Using brand column: {brand_col}")
    
    # Tags column for better categorization (if available)
    tags_col = 'product_tags' if 'product_tags' in df_raw.columns else None
    print(f"Using tags column: {tags_col}")
    
    # NLP Categorization with tags priority
    def heuristic_category(product_name, brand='', tags=''):
        # Priority: tags if available
        if tags and not pd.isna(tags):
            tags_lower = str(tags).lower()
            if any(w in tags_lower for w in ['skin', 'face', 'moistur', 'cleans', 'serum', 'cream', 'lotion']):
                return 'Skincare'
            elif any(w in tags_lower for w in ['hair', 'shampoo', 'condition', 'dye']):
                return 'Haircare'
            elif any(w in tags_lower for w in ['makeup', 'lip', 'foundation', 'mascara', 'eye', 'blush']):
                return 'Makeup'
            elif any(w in tags_lower for w in ['fragrance', 'perfume', 'cologne']):
                return 'Fragrance'
            elif any(w in tags_lower for w in ['body', 'deodorant', 'lotion', 'wash', 'soap']):
                return 'Bodycare'
        # Fallback to name/brand
        name_lower = str(product_name).lower()
        if any(w in name_lower for w in ['cream', 'serum', 'moisturizer', 'lotion', 'cleanser', 'mask', 'face', 'toner', 'exfoliator', 'sunscreen', 'eye cream', 'face oil', 'facial']):
            return 'Skincare'
        elif any(w in name_lower for w in ['shampoo', 'conditioner', 'hair oil', 'hair serum', 'hair', 'dye', 'styling gel', 'hair mask', 'hair color', 'hair spray', 'dry shampoo']):
            return 'Haircare'
        elif any(w in name_lower for w in ['lipstick', 'foundation', 'mascara', 'kajal', 'eyeliner', 'blush', 'gloss', 'powder', 'concealer', 'primer', 'highlighter', 'bronzer', 'eyeshadow']):
            return 'Makeup'
        elif any(w in name_lower for w in ['perfume', 'fragrance', 'cologne', 'body mist', 'scent', 'eau de', 'toilette']):
            return 'Fragrance'
        elif any(w in name_lower for w in ['body wash', 'body lotion', 'deodorant', 'body cream', 'scrub', 'soap', 'body oil', 'hand cream', 'foot cream']):
            return 'Bodycare'
        else:
            return 'Other'
    
    df_raw['Category_Heuristic'] = df_raw.apply(lambda row: heuristic_category(row[product_col], row.get(brand_col, ''), row.get(tags_col, '')), axis=1)
    
    # Check class distribution
    class_dist = df_raw['Category_Heuristic'].value_counts()
    print("Heuristic class distribution:", class_dist)
    if len(class_dist) < 2:
        print("Warning: Only one class detected. Falling back to heuristic only (no NLP model).")
        df_raw['Category'] = df_raw['Category_Heuristic']
        model = None
    else:
        # Train NLP model
        X = df_raw[product_col].fillna('').astype(str)
        y = df_raw['Category_Heuristic']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = make_pipeline(
            TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2)),
            LogisticRegression(multi_class='multinomial', max_iter=200, random_state=42)
        )
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"NLP Accuracy: {acc:.2f}")
        
        df_raw['Category'] = model.predict(X)
    model = model if 'model' in locals() else None
    
    # Year extraction
    if date_col:
        df_raw['Year'] = pd.to_datetime(df_raw[date_col], errors='coerce').dt.year
    else:
        df_raw['Year'] = datetime.now().year  # Fallback
    df_raw = df_raw.dropna(subset=['Year'])
    
    # Claims extraction
    def extract_claims(text):
        if pd.isna(text):
            return {'Natural Ingredients': 0, 'Hydrating': 0, 'Anti-Aging': 0, 'Long-Lasting': 0, 'Brightening': 0}
        text_lower = str(text).lower()
        return {
            'Natural Ingredients': int(any(w in text_lower for w in ['natural', 'organic', 'herbal'])),
            'Hydrating': int(any(w in text_lower for w in ['hydrat', 'moistur', 'plump'])),
            'Anti-Aging': int(any(w in text_lower for w in ['anti ag', 'wrinkle', 'firm'])),
            'Long-Lasting': int(any(w in text_lower for w in ['long last', 'all day', 'smudge proof'])),
            'Brightening': int(any(w in text_lower for w in ['brighten', 'glow', 'even tone']))
        }
    
    claims_data = []
    for _, row in df_raw.iterrows():
        text = row.get(review_col, '') if review_col else ''
        for claim, count in extract_claims(text).items():
            if count > 0:
                claims_data.append({
                    'Year': row['Year'],
                    'Claim': claim,
                    'Mention_Count': 1,
                    'Avg_Claim_Rating': row[rating_col]
                })
    if claims_data:
        claims_df = pd.DataFrame(claims_data).groupby(['Year', 'Claim']).agg({
            'Mention_Count': 'sum',
            'Avg_Claim_Rating': 'mean'
        }).reset_index()
        claims_df['YoY_Growth'] = claims_df.groupby('Claim')['Mention_Count'].pct_change() * 100
        claims_df['YoY_Growth'] = claims_df['YoY_Growth'].fillna(0)
    else:
        # Empty claims DF with columns
        claims_df = pd.DataFrame(columns=['Year', 'Claim', 'Mention_Count', 'Avg_Claim_Rating', 'YoY_Growth'])
        print("No claims extracted; empty claims DF.")
    
    # Category aggregation
    cat_df = df_raw.groupby(['Year', 'Category']).agg({
        rating_col: ['count', 'mean']
    }).reset_index()
    cat_df.columns = ['Year', 'Category', 'Sales_Volume', 'Avg_Rating']
    cat_df['YoY_Growth'] = cat_df.groupby('Category')['Sales_Volume'].pct_change() * 100
    cat_df['YoY_Growth'] = cat_df['YoY_Growth'].fillna(0)
    
    return cat_df, claims_df, model

if __name__ == "__main__":
    cat_df, claims_df, model = load_and_process()
    cat_df.to_csv('processed_categories.csv', index=False)
    claims_df.to_csv('processed_claims.csv', index=False)
    print("Processed data saved to CSVs.")