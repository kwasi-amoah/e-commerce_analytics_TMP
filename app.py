#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import pandas as pd
import joblib
import sys
from recommendation import RecommendationSystem, DataLoader, DataPreprocessor
import os
import gdown

st.set_page_config(page_title="Hybrid Recommendation System", layout="wide")
st.title("📊 Hybrid Recommendation System (with Preprocessing)")

# --- Upload CSV files ---
st.sidebar.header("Upload CSV files")
events_file = st.sidebar.file_uploader("Events CSV", type=["csv"])
item_props1_file = st.sidebar.file_uploader("Item Properties 1 CSV", type=["csv"])
item_props2_file = st.sidebar.file_uploader("Item Properties 2 CSV", type=["csv"])
category_tree_file = st.sidebar.file_uploader("Category Tree CSV", type=["csv"])

# Sidebar parameters
n_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10)
alpha = st.sidebar.slider("Hybrid Weight (CF vs CBF)", 0.0, 1.0, 0.5)

# --- Load saved model ---
# 👇 Trick: register RecommendationSystem under __main__ for joblib
sys.modules['__main__'].RecommendationSystem = RecommendationSystem

MODEL_PATH = "saved_models/Recommendation_system/recommendation_system.pkl"
MODEL_ID = "11T--qROirrPzlPuj_njrVEVwfdtQJhSV"  # Google Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    download_model()
    return joblib.load(MODEL_PATH)

rec = load_model()
st.success("✅ Loaded saved recommendation system")

if st.sidebar.button("Preprocess & Run"):
    if events_file and item_props1_file and item_props2_file and category_tree_file:
        with st.spinner("🔄 Preprocessing uploaded data..."):
            try:
                # 1. Load data
                loader = DataLoader()
                loader.load_data(events_file, item_props1_file, item_props2_file, category_tree_file)
                loader.convert_timestamps()

                # 2. Preprocess
                preprocessor = DataPreprocessor(loader.events_df, loader.item_props_df, loader.category_tree_df)
                preprocessor.parse_property_values().build_category_hierarchy().create_user_features()

                st.success("✅ Data preprocessed successfully")

                # Debug: show sample of data
                st.subheader("📂 Sample of Processed Events Data")
                st.write(loader.events_df.head())

                # 3. User input for recommendations
                st.subheader("🎯 Generate Recommendations for a User")
                user_id_input = st.text_input("Enter a User ID (visitorid):")

                if user_id_input:
                    if user_id_input in rec.user_ids:
                        user_idx = rec.user_ids[user_id_input]

                        recs = rec.generate_hybrid_recommendations_for_eval(
                            user_idx,
                            train_item_indices=set(),
                            n_recommendations=n_recommendations,
                            alpha=alpha
                        )
                        recs_df = pd.DataFrame(recs, columns=["Item ID", "Score"])
                        st.table(recs_df)
                    else:
                        st.error("⚠️ User ID not found in the trained model.")

            except Exception as e:
                st.error(f"❌ Error during preprocessing: {e}")

    else:
        st.error("⚠️ Please upload all four CSV files before running.")

