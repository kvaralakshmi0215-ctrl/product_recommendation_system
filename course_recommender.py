"""
AI Career & Course Recommendation System
==========================================
Content-based recommendation engine powered by TF-IDF and Cosine Similarity,
paired with multi-model difficulty classification on the Coursera catalog.

Features:
- Recommend similar courses based on course title
- Recommend career pathways matching target skills
- Filter by difficulty level (Beginner, Intermediate, Advanced, Mixed)
- Benchmark supervised ML models (Linear SVM, Logistic Regression, Random Forest, etc.)
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score


class CourseRecommender:
    def __init__(self, data_path="Coursera.csv"):
        self.data_path = data_path
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """Loads and cleans the dataset, building the TF-IDF representation."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

        print(f"Loading Coursera dataset from: {self.data_path}...")
        self.df = pd.read_csv(self.data_path, encoding='utf-8', encoding_errors='replace')

        # Fill missing values
        self.df['Course Description'] = self.df['Course Description'].fillna('')
        self.df['Skills'] = self.df['Skills'].fillna('')
        self.df['Difficulty Level'] = self.df['Difficulty Level'].fillna('Mixed')
        self.df['Course Rating'] = pd.to_numeric(self.df['Course Rating'], errors='coerce').fillna(4.0)

        # Combined text feature for semantic matching
        self.df['combined_features'] = (
            self.df['Course Name'] + " " +
            self.df['Course Description'] + " " +
            self.df['Skills']
        )

        print(f"Total courses loaded: {len(self.df):,}")

        # Vectorization
        print("Computing TF-IDF representations (max_features=5,000)...")
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])

        print("Computing pairwise cosine similarity matrix...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("Recommendation engine initialized successfully!\n")

    def recommend_by_course(self, course_name, top_n=5):
        """Recommends top-N most similar courses to a given course name."""
        matches = self.df[self.df['Course Name'].str.contains(course_name, case=False, na=False)]
        if matches.empty:
            print(f"No exact course matched for: '{course_name}'. Finding closest semantic match...")
            query_vec = self.tfidf.transform([course_name])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = sim_scores.argsort()[-top_n:][::-1]
        else:
            idx = matches.index[0]
            matched_title = self.df.iloc[idx]['Course Name']
            print(f"Matched Course: '{matched_title}'")
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            top_indices = [i[0] for i in sim_scores]

        results = self.df.iloc[top_indices][['Course Name', 'University', 'Difficulty Level', 'Course Rating', 'Course URL']]
        return results

    def recommend_by_skills(self, skills, top_n=5, difficulty=None):
        """Recommends courses that best teach the given career skill set."""
        if isinstance(skills, list):
            skills_query = " ".join(skills)
        else:
            skills_query = str(skills)

        query_vec = self.tfidf.transform([skills_query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        filtered_df = self.df.copy()
        filtered_df['sim_score'] = sim_scores

        if difficulty:
            filtered_df = filtered_df[filtered_df['Difficulty Level'].str.lower() == difficulty.lower()]

        top_df = filtered_df.sort_values(by='sim_score', ascending=False).head(top_n)
        return top_df[['Course Name', 'University', 'Difficulty Level', 'Course Rating', 'sim_score', 'Course URL']]

    def benchmark_difficulty_classifiers(self):
        """Benchmarks machine learning classifiers predicting course difficulty levels."""
        print("Benchmarking Difficulty Level Classifiers...")
        valid_df = self.df[self.df['Difficulty Level'].isin(['Beginner', 'Intermediate', 'Advanced', 'Mixed'])]

        X = self.tfidf.transform(valid_df['combined_features'])
        y = valid_df['Difficulty Level']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        models = {
            'Multinomial Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Linear SVM': LinearSVC(max_iter=2000, random_state=42),
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            results[name] = {'Accuracy': acc, 'F1-Score': f1}
            print(f"  {name:25s} | Accuracy: {acc*100:.2f}% | F1-Score: {f1*100:.2f}%")

        return results


def main():
    parser = argparse.ArgumentParser(description="AI Career & Course Recommendation System")
    parser.add_argument("--course", type=str, help="Course title to find similar recommendations for")
    parser.add_argument("--skills", type=str, help="Comma-separated target career skills (e.g., 'python, machine learning, sql')")
    parser.add_argument("--difficulty", type=str, choices=['Beginner', 'Intermediate', 'Advanced', 'Mixed'], help="Filter by difficulty level")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--benchmark", action="store_true", help="Run model benchmarking on course difficulty")

    args = parser.parse_args()

    recommender = CourseRecommender(data_path="Coursera.csv")

    if args.benchmark:
        recommender.benchmark_difficulty_classifiers()

    if args.course:
        print(f"\n--- Recommendations based on course: '{args.course}' ---")
        recs = recommender.recommend_by_course(args.course, top_n=args.top_n)
        print(recs.to_string(index=False))

    if args.skills:
        skill_list = [s.strip() for s in args.skills.split(',')]
        print(f"\n--- Career Recommendations for skills: {skill_list} ---")
        recs = recommender.recommend_by_skills(skill_list, top_n=args.top_n, difficulty=args.difficulty)
        print(recs.to_string(index=False))

    if not args.course and not args.skills and not args.benchmark:
        print("\n--- Example: Recommending for Data Science Career Skills ---")
        example_skills = ["machine learning", "python", "data analysis", "statistics"]
        recs = recommender.recommend_by_skills(example_skills, top_n=5)
        print(recs[['Course Name', 'University', 'Difficulty Level', 'Course Rating', 'sim_score']].to_string(index=False))


if __name__ == "__main__":
    main()
