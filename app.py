import re
import string
from collections import Counter
 
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
@st.cache_resource
def load_nltk_resources():
    for resource in ["stopwords", "punkt", "wordnet", "punkt_tab"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    return set(stopwords.words("english")), WordNetLemmatizer()
 
 
STOP_WORDS, LEMMATIZER = load_nltk_resources()
 
SKILLS = [
    "python", "sql", "excel", "power bi", "tableau",
    "machine learning", "deep learning", "tensorflow",
    "keras", "pytorch", "nlp", "data analysis",
    "data science", "statistics", "aws", "azure",
    "git", "docker", "java", "c++", "javascript",
    "html", "css", "react", "flask", "django",
    "mongodb", "mysql", "postgresql", "linux",
]
 
def preprocess_text(text: str) -> str:
    """Lowercase, strip numbers/punctuation, remove stopwords, lemmatize."""
    text = str(text)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
 
    words = nltk.word_tokenize(text)
    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS
    ]
    return " ".join(words)
 
 
def extract_skills(text: str) -> list:
    """Scan text for any skill in the fixed SKILLS vocabulary."""
    text = str(text).lower()
    return [skill for skill in SKILLS if skill in text]

def _read_job_file(path: str) -> pd.DataFrame:
    
    with open(path, "rb") as f:
        header = f.read(8)
 
    is_xlsx = header.startswith(b"PK\x03\x04")          
    is_xls = header.startswith(b"\xd0\xcf\x11\xe0")      
 
    if is_xlsx or is_xls:
       
        engine = "openpyxl" if is_xlsx else "xlrd"
        try:
            return pd.read_excel(path, engine=engine)
        except ImportError as e:
            raise ImportError(
                f"This is a real Excel file, but the '{engine}' package isn't "
                f"installed. Run: pip install {engine}"
            ) from e
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise ValueError(
            "Could not parse this file as CSV or Excel. Please check that it's "
            "a valid job-postings export."
        ) from e
 
 
@st.cache_data
def load_and_prepare_data(csv_path: str):
    df = _read_job_file(csv_path)
 
    if df.empty or df.shape[1] <= 1:
        raise ValueError(
            "The file was read but no usable columns were found. "
            "Check that it's a valid CSV/Excel export."
        )
 
    df = df.drop(columns=[c for c in ["Uniq Id", "Crawl Timestamp"] if c in df.columns])

    for col in ["Job Title", "Key Skills", "Role", "Industry"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
 
    df["Cleaned_Key_Skills"] = df["Key Skills"].apply(preprocess_text)

    df["Extracted Skills"] = df["Key Skills"].apply(extract_skills)

    df["Job_Text"] = (
        df.get("Job Title", "") + " "
        + df.get("Key Skills", "") + " "
        + df.get("Role", "") + " "
        + df.get("Industry", "")
    )
 
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["Job_Text"])
 
    return df, tfidf, tfidf_matrix
 
 
def match_resume(resume_text: str, df: pd.DataFrame, tfidf: TfidfVectorizer,
                  tfidf_matrix, top_n: int = 5):
    """Vectorize the resume and return the top_n best-matching jobs."""
    resume_vector = tfidf.transform([resume_text])
    similarity_scores = cosine_similarity(resume_vector, tfidf_matrix).flatten()
 
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices].copy()
    results["Match Score (%)"] = (similarity_scores[top_indices] * 100).round(2)
    return results, similarity_scores, top_indices
 
def main():
    st.set_page_config(page_title="Resume ↔ Job Matcher", layout="wide")
    st.title("📄 Resume ↔ Job Matcher")
    st.caption(
        "TF-IDF + cosine similarity job matching, based on the NLP_Project notebook."
    )
 
    with st.sidebar:
        st.header("Settings")
        csv_path = st.text_input(
            "Path to job postings file (.csv, .xls, or .xlsx)", value="job.xls"
        )
        top_n = st.slider("Number of matches to show", min_value=1, max_value=10, value=5)
 
    try:
        df, tfidf, tfidf_matrix = load_and_prepare_data(csv_path)
    except FileNotFoundError:
        st.error(
            f"Could not find '{csv_path}'. Place your job postings file "
            "(.csv, .xls, or .xlsx) in the app folder (or update the path "
            "in the sidebar) and reload."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
 
    st.success(f"Loaded {len(df):,} job postings from `{csv_path}`.")
 
    st.subheader("1. Paste your resume text")
    default_resume = (
        "Python SQL Machine Learning Data Analysis\n"
        "Power BI Tableau Excel Statistics"
    )
    resume_text = st.text_area("Resume / skills text", value=default_resume, height=150)
 
    if st.button("Find matching jobs", type="primary"):
        if not resume_text.strip():
            st.warning("Please paste some resume text first.")
            st.stop()
 
        results, similarity_scores, top_indices = match_resume(
            resume_text, df, tfidf, tfidf_matrix, top_n=top_n
        )
 
        st.subheader("2. Top matching jobs")
        display_cols = [c for c in [
            "Job Title", "Location", "Industry", "Job Experience Required", "Match Score (%)"
        ] if c in results.columns]
        st.dataframe(results[display_cols], use_container_width=True)
 
        best_idx = top_indices[0]
        best_job = df.iloc[best_idx]
        best_score = similarity_scores[best_idx] * 100
 
        st.subheader("3. Best match details")
        st.markdown(f"**Job Title:** {best_job.get('Job Title', 'N/A')}")
        st.markdown(f"**Matching Score:** {best_score:.2f}%")
 
        resume_skills = extract_skills(resume_text.lower())
        job_skills = best_job.get("Extracted Skills", [])
 
        matched_skills = sorted(set(job_skills).intersection(set(resume_skills)))
        missing_skills = sorted(set(job_skills) - set(resume_skills))
        skill_match_pct = (
            (len(matched_skills) / len(job_skills)) * 100 if len(job_skills) > 0 else 0
        )
 
        col1, col2, col3 = st.columns(3)
        col1.metric("Skill Match", f"{skill_match_pct:.1f}%")
        col2.metric("Matched Skills", len(matched_skills))
        col3.metric("Missing Skills", len(missing_skills))
 
        st.markdown("**✅ Matched skills:** " + (", ".join(matched_skills) or "None"))
        st.markdown("**❌ Missing skills:** " + (", ".join(missing_skills) or "None"))
 
    st.divider()
    with st.expander("📊 Explore the dataset (word frequency, top skills, etc.)"):
        all_text = " ".join(df["Cleaned_Key_Skills"])
        words = all_text.split()
        word_counts = Counter(words)
        word_freq = pd.DataFrame(word_counts.most_common(20), columns=["Word", "Frequency"])
 
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(word_freq["Word"], word_freq["Frequency"])
        ax.set_title("Top 20 Most Frequent Skills")
        ax.set_xticklabels(word_freq["Word"], rotation=90)
        st.pyplot(fig)
 
        if "Location" in df.columns:
            st.bar_chart(df["Location"].value_counts().head(10))
        if "Industry" in df.columns:
            st.bar_chart(df["Industry"].value_counts().head(10))
 
 
if __name__ == "__main__":
    main()


   

           
    
       
