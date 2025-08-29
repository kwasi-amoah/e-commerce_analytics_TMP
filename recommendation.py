#!/usr/bin/env python
# coding: utf-8

# In[1]:


# recommendation.py

import pandas as pd
import numpy as np


class DataLoader:
    """Handle data loading and initial preprocessing"""

    def __init__(self):
        self.events_df = None
        self.item_props_df = None
        self.category_tree_df = None

    def load_data(
        self,
        events_path,
        item_props_path1,
        item_props_path2,
        category_tree_path,
        events_sample_size=1000000,  # adjustable
        item_props_sample_size=1000000,  # adjustable
        random_state=42,
    ):
        """Load and optionally sample datasets for faster downstream analysis."""
        print("Loading datasets...")

        # --- Load events data and sample ---
        self.events_df = pd.read_csv(events_path)
        print(f"Events data original shape: {self.events_df.shape}")
        if events_sample_size and events_sample_size < len(self.events_df):
            self.events_df = self.events_df.sample(
                n=events_sample_size, random_state=random_state
            )
            print(f"Events data sampled shape: {self.events_df.shape}")

        # --- Load and combine item properties, then sample ---
        props1 = pd.read_csv(item_props_path1)
        props2 = pd.read_csv(item_props_path2)
        self.item_props_df = pd.concat([props1, props2], ignore_index=True)
        print(f"Item properties combined shape: {self.item_props_df.shape}")
        if item_props_sample_size and item_props_sample_size < len(
            self.item_props_df
        ):
            self.item_props_df = self.item_props_df.sample(
                n=item_props_sample_size, random_state=random_state
            )
            print(f"Item properties sampled shape: {self.item_props_df.shape}")

        # --- Load category tree (no sampling) ---
        self.category_tree_df = pd.read_csv(category_tree_path)
        print(f"Category tree data shape: {self.category_tree_df.shape}")

        return self

    def convert_timestamps(self):
        """Convert timestamps from scientific notation to datetime"""
        print("Converting timestamps...")

        # Convert events timestamps
        self.events_df["timestamp"] = pd.to_datetime(
            self.events_df["timestamp"], unit="s", errors="coerce"
        )

        # Convert item properties timestamps
        self.item_props_df["timestamp"] = pd.to_datetime(
            self.item_props_df["timestamp"], unit="s", errors="coerce"
        )

        # Extract time features
        self.events_df["hour"] = self.events_df["timestamp"].dt.hour
        self.events_df["day_of_week"] = self.events_df["timestamp"].dt.dayofweek
        self.events_df["month"] = self.events_df["timestamp"].dt.month
        self.events_df["day"] = self.events_df["timestamp"].dt.day
        self.events_df["week"] = self.events_df["timestamp"].dt.isocalendar().week

        return self


class DataPreprocessor:
    """Handle data preprocessing and feature engineering"""

    def __init__(self, events_df, item_props_df, category_tree_df):
        self.events_df = events_df.copy()
        self.item_props_df = item_props_df.copy()
        self.category_tree_df = category_tree_df.copy()

    def parse_property_values(self):
        """Parse complex property values"""
        print("Parsing property values...")

        def extract_numeric_values(value_str):
            """Extract numeric values from property string"""
            if pd.isna(value_str):
                return []

            value_str = str(value_str)
            numeric_values = []

            # Find numeric values starting with 'n'
            parts = value_str.split()
            for part in parts:
                if part.startswith("n"):
                    try:
                        num = float(part[1:])  # Remove 'n' prefix
                        numeric_values.append(num)
                    except:
                        continue

            return numeric_values

        # Extract numeric values
        self.item_props_df["numeric_values"] = self.item_props_df["value"].apply(
            extract_numeric_values
        )
        self.item_props_df["num_count"] = self.item_props_df[
            "numeric_values"
        ].apply(len)
        self.item_props_df["avg_numeric"] = self.item_props_df[
            "numeric_values"
        ].apply(lambda x: np.mean(x) if x else np.nan)

        return self

    def build_category_hierarchy(self):
        """Build category hierarchy structure"""
        print("Building category hierarchy...")

        # Create category mapping
        category_map = {}
        for _, row in self.category_tree_df.iterrows():
            category_map[row["categoryid"]] = (
                row["parentid"] if pd.notna(row["parentid"]) else None
            )

        def get_category_path(category_id, visited=None):
            """Get full category path"""
            if visited is None:
                visited = set()

            if category_id in visited or category_id not in category_map:
                return [category_id]

            visited.add(category_id)
            parent = category_map[category_id]

            if parent is None:
                return [category_id]

            return get_category_path(parent, visited) + [category_id]

        # Calculate category levels
        category_levels = {}
        for cat_id in category_map.keys():
            path = get_category_path(cat_id)
            category_levels[cat_id] = len(path)

        self.category_hierarchy = {
            "category_map": category_map,
            "category_levels": category_levels,
        }

        return self

    def create_user_features(self):
        """Create comprehensive user features"""
        print("Creating user features...")

        # Basic user statistics
        user_features = (
            self.events_df.groupby("visitorid")
            .agg(
                {
                    "event": ["count", "nunique"],
                    "itemid": "nunique",
                    "timestamp": ["min", "max"],
                    "hour": "mean",
                    "day_of_week": "mean",
                }
            )
            .round(3)
        )

        # Flatten column names
        user_features.columns = [
            "_".join(col).strip() for col in user_features.columns
        ]

        # Event type counts
        event_counts = (
            self.events_df.groupby(["visitorid", "event"]).size().unstack(fill_value=0)
        )
        user_features = pd.concat([user_features, event_counts], axis=1)

        # Calculate session duration
        user_features["session_duration"] = (
            user_features["timestamp_max"] - user_features["timestamp_min"]
        ).dt.total_seconds() / 3600  # Convert to hours

        # Conversion rates
        user_features["view_to_cart_rate"] = (
            user_features.get("addtocart", 0) / user_features.get("view", 1)
        )
        user_features["cart_to_purchase_rate"] = (
            user_features.get("transaction", 0) / user_features.get("addtocart", 1)
        )
        user_features["overall_conversion_rate"] = (
            user_features.get("transaction", 0) / user_features.get("view", 1)
        )

        # Activity patterns
        user_features["events_per_hour"] = (
            user_features["event_count"] / user_features["session_duration"].clip(lower=0.1)
        )

        self.user_features = user_features.fillna(0)
        return self


# In[2]:


# recommendation.py
import pandas as pd
import numpy as np

class RecommendationSystem:
    """Hybrid recommendation system using SVD (CF) + content-based filtering (CBF)"""
    
    def __init__(self, events_df, item_props_df):
        self.events_df = events_df
        self.item_props_df = item_props_df

    def create_interaction_matrix(self):
        """Create sparse user-item interaction matrix"""
        event_weights = {'view': 1, 'addtocart': 3, 'transaction': 5}
        interactions = self.events_df.copy()
        interactions['weight'] = interactions['event'].map(event_weights)

        # Aggregate interactions
        user_item_matrix = (
            interactions.groupby(['visitorid', 'itemid'])['weight']
            .sum()
            .reset_index()
        )

        # Map user and item IDs to indices
        self.user_ids = {uid: idx for idx, uid in enumerate(user_item_matrix['visitorid'].unique())}
        self.item_ids = {iid: idx for idx, iid in enumerate(user_item_matrix['itemid'].unique())}

        user_index = user_item_matrix['visitorid'].map(self.user_ids)
        item_index = user_item_matrix['itemid'].map(self.item_ids)
        weights = user_item_matrix['weight'].astype(float)

        self.interaction_matrix = csr_matrix((weights, (user_index, item_index)),
                                             shape=(len(self.user_ids), len(self.item_ids)))
        return self

    def build_collaborative_filtering(self, n_components=50):
        """SVD-based collaborative filtering"""
        sparse_matrix = self.interaction_matrix
        n_components = min(n_components, min(sparse_matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = svd.fit_transform(sparse_matrix)
        self.item_factors = svd.components_.T
        return self

    def build_content_based_filtering(self):
        """Simple content-based scoring based on 'available' property"""
        self.item_profiles = {}
        for item_id in self.item_ids:
            props = self.item_props_df[self.item_props_df['itemid'] == item_id]
            # Simple numeric score: use 'available' if exists, else 0
            available = props.loc[props['property'] == 'available', 'value']
            score = float(available.iloc[0]) if not available.empty else 0
            self.item_profiles[item_id] = score
        return self

    def _cf_scores_for_user(self, user_idx):
        """Get CF scores for all items for a single user"""
        user_vector = self.user_factors[user_idx]
        cf_scores = np.dot(self.item_factors, user_vector)
        # Remove already interacted items
        interacted = set(self.interaction_matrix.getrow(user_idx).indices)
        cf_scores[list(interacted)] = -np.inf
        return cf_scores

    def _cb_scores_for_user(self, user_idx):
        """Get CBF scores for all items for a single user"""
        cb_scores = np.zeros(len(self.item_ids))
        for item_id, idx in self.item_ids.items():
            cb_scores[idx] = self.item_profiles.get(item_id, 0)
        # Remove already interacted items
        interacted = set(self.interaction_matrix.getrow(user_idx).indices)
        cb_scores[list(interacted)] = -np.inf
        return cb_scores

    def generate_hybrid_recommendations_for_eval(self, user_idx, train_item_indices, n_recommendations=10, alpha=0.5):
        """
        Generate hybrid recommendations for evaluation, filtering only train items.
        """
        # CF scores
        user_vector = self.user_factors[user_idx]
        cf_scores = np.dot(self.item_factors, user_vector)
        cf_scores = cf_scores / (np.linalg.norm(cf_scores) + 1e-8)
    
        # CBF scores
        cb_scores = np.zeros_like(cf_scores)
        for item_id, idx in self.item_ids.items():
            cb_scores[idx] = self.item_profiles.get(item_id, 0)
        cb_scores = cb_scores / (np.linalg.norm(cb_scores) + 1e-8)
    
        # Hybrid
        hybrid_scores = alpha * cf_scores + (1 - alpha) * cb_scores
    
        # Remove items in train set
        hybrid_scores[list(train_item_indices)] = -np.inf
    
        # Get top N
        top_indices = np.argsort(hybrid_scores)[-n_recommendations:][::-1]
        recommendations = [(list(self.item_ids.keys())[i], hybrid_scores[i]) for i in top_indices]
    
        return recommendations


    def evaluate_hybrid_recommendations(self, n_recommendations=10, alpha=0.5, test_size=0.2, min_interactions=2):
        """
        Evaluate hybrid recommendations using precision, recall, and F1.
        """
        precisions, recalls = [], []
        idx_to_user = {idx: uid for uid, idx in self.user_ids.items()}

        for user_idx in range(len(self.user_ids)):
            user_id = idx_to_user[user_idx]
            user_interactions = set(self.interaction_matrix.getrow(user_idx).indices)
            
            if len(user_interactions) < min_interactions:
                continue

            # Split into train/test
            test_count = max(1, int(len(user_interactions) * test_size))
            test_items = set(np.random.choice(list(user_interactions), test_count, replace=False))
            train_items = user_interactions - test_items

            # Generate recommendations
            recs = self.generate_hybrid_recommendations_for_eval(user_idx, train_items, n_recommendations, alpha)
            recommended_indices = {self.item_ids[item] for item, _ in recs if item in self.item_ids}

            # Compute metrics
            relevant = recommended_indices & test_items
            precision = len(relevant) / len(recommended_indices) if recommended_indices else 0
            recall = len(relevant) / len(test_items) if test_items else 0

            precisions.append(precision)
            recalls.append(recall)

        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        self.evaluation_results = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1,
            'users_evaluated': len(precisions)
        }

        print("Hybrid Recommendation System Evaluation:")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Users evaluated: {len(precisions)}")

        return self


# In[ ]:




