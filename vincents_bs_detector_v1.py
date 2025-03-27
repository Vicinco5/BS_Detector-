#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:09:58 2025

Vincent's AI-BS detector, version 1.0
Prompted by Spite, Driven by Honesty 

@author: vincentcalia-bogan
"""
import streamlit as st
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import edit_distance
from scipy.stats import ranksums
import matplotlib.pyplot as plt

# Load spaCy model globally
nlp = spacy.load('en_core_web_sm')

# Similarity Functions
def get_embedding(doc):
    return doc.vector

def jaccard_similarity(doc1, doc2):
    tokens1 = set(token.lemma_.lower() for token in doc1 if not token.is_punct and not token.is_stop)
    tokens2 = set(token.lemma_.lower() for token in doc2 if not token.is_punct and not token.is_stop)
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union != 0 else 0

def dependency_structure_similarity(doc1, doc2):
    deps1 = [token.dep_ for token in doc1]
    deps2 = [token.dep_ for token in doc2]
    max_len = max(len(deps1), len(deps2))
    distance = edit_distance(deps1, deps2)
    similarity = 1 - (distance / max_len)
    return similarity

def punctuation_similarity(text1, text2, symbols):
    sim = np.mean([text1.count(p)==text2.count(p) for p in symbols])
    return sim


# Sentence-wise Comparison
def sentence_wise_similarity(sentences1, sentences2, thresholds, symbols):
    min_len = min(len(sentences1), len(sentences2))
    sims = {'Semantic': [], 'Vocabulary': [], 'Dependency': [], 'Punctuation': []}

    for i in range(min_len):
        doc1, doc2 = nlp(sentences1[i]), nlp(sentences2[i])
        sims['Semantic'].append(cosine_similarity([get_embedding(doc1)], [get_embedding(doc2)])[0][0])
        sims['Vocabulary'].append(jaccard_similarity(doc1, doc2))
        sims['Dependency'].append(dependency_structure_similarity(doc1, doc2))
        sims['Punctuation'].append(punctuation_similarity(sentences1[i], sentences2[i], symbols))

    results = {}
    overall_vector = []

    for key in sims:
        sim_vector = np.array(sims[key])
        threshold_vector = np.full(sim_vector.shape, thresholds[key])
        stat, p_value = ranksums(sim_vector, threshold_vector)
        results[key] = {
            'Mean Similarity': np.mean(sim_vector),
            'Statistic': stat,
            'p-value': p_value,
            'Reject Null': p_value < 0.05
        }
        overall_vector.append(sim_vector)

    overall_mean_vector = np.mean(overall_vector, axis=0)
    overall_threshold_vector = np.full(overall_mean_vector.shape, np.mean(list(thresholds.values())))
    stat, p_value = ranksums(overall_mean_vector, overall_threshold_vector)
    results['Overall'] = {
        'Mean Similarity': np.mean(overall_mean_vector),
        'Statistic': stat,
        'p-value': p_value,
        'Reject Null': p_value < 0.05
    }

    return results, sims
# full-text comparison 
def full_text_similarity(text1, text2, thresholds, symbols):
    doc1, doc2 = nlp(text1), nlp(text2)

    sims = {
        'Semantic': cosine_similarity([get_embedding(doc1)], [get_embedding(doc2)])[0][0],
        'Vocabulary': jaccard_similarity(doc1, doc2),
        'Dependency': dependency_structure_similarity(doc1, doc2),
        'Punctuation': punctuation_similarity(text1, text2, symbols)
    }

    results = {}
    sim_vector = np.array(list(sims.values()))
    threshold_vector = np.array(list(thresholds.values()))

    stat, p_value = ranksums(sim_vector, threshold_vector)
    results['Overall'] = {
        'Mean Similarity': np.mean(sim_vector),
        'Statistic': stat,
        'p-value': p_value,
        'Reject Null': p_value < 0.05
    }

    for key in sims:
        results[key] = {
            'Mean Similarity': sims[key],
            'Statistic': None,
            'p-value': None,
            'Reject Null': sims[key] < thresholds[key]  # Simple threshold check
        }

    return results, sims


# Streamlit App Setup
st.set_page_config(page_title="Vincent's BS Detector", layout="wide")

st.markdown("### Vincent's BS detector version 1.0 | Prompted by Spite, Driven by Honesty | input word limit: 2500 words")

st.markdown(
    """
    <div style="
        border: 2px solid #1E90FF;
        padding: 15px;
        border-radius: 8px;
        ">
        Individual similarity measures are meant to be interperted hollistically by the user, with respect to their specific use. 
        This tool cannot confirm that two samples are distinctly different, but seeks to provide insight on linguistic differences in writing when doubt arises. 
        This is not an AI-detection tool, it simply runs an A-B comparison on two text samples and gives a statistical measure of how similar they are. Please use with discretion. 
        Built by Vincent Calia-Bogan at Brandeis University, spring 2025. 
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("Known-student writing", height=400)

with col2:
    text2 = st.text_area("Suspected plagiarism", height=400)

analysis_type = st.radio(
    "Choose analysis type: (note: Sentence by sentence measures are far more sensitive to the substance (eg, topic) of the input.",
    ["Sentence-by-Sentence", "Entire Text"]
)

st.markdown("#### Adjust the thresholds below to set the sensitivity of the statistical significance test (a Mann-Whitney U test). Does not affect the result of the individual similarity rankings. Raising the thresholds will bias the test towards detecting highly-dissimilar text if significant (test hypothesis: all characteristics are similar), and lowering thresholds does the opposite (test hypothesis: all characteristics are different). Adjust based on what you care about.")
thresh_cols = st.columns(4)

with thresh_cols[0]:
    semantic_thresh = st.number_input("Semantic Threshold", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
with thresh_cols[1]:
    vocab_thresh = st.number_input("Vocabulary Threshold", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
with thresh_cols[2]:
    dep_thresh = st.number_input("Dependency Threshold", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
with thresh_cols[3]:
    punctuation_thresh = st.number_input("Punctuation Threshold", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

thresholds = {'Semantic': semantic_thresh, 'Vocabulary': vocab_thresh,
              'Dependency': dep_thresh, 'Punctuation': punctuation_thresh}

# Default symbols for punctuation similarity
default_symbols = '.,;:!?()-[]"\'{}<>/@#$%^&*_~`|+=\\'
default_symbols += '\n•-–—'  # Includes enter/return, bullet points, and common list markers
symbol_explanations = {
    '.': 'Period', ',': 'Comma', ';': 'Semicolon', ':': 'Colon', '!': 'Exclamation mark', '?': 'Question mark',
    '(': 'Left parenthesis', ')': 'Right parenthesis', '-': 'Hyphen/Dash', '[': 'Left bracket', ']': 'Right bracket',
    '\"': 'Quotation mark', "'": 'Apostrophe', '{': 'Left curly brace', '}': 'Right curly brace',
    '<': 'Less than', '>': 'Greater than', '/': 'Forward slash', '@': 'At sign', '#': 'Hash', '$': 'Dollar sign',
    '%': 'Percent', '^': 'Caret', '&': 'Ampersand', '* ': 'Asterisk', '_': 'Underscore', '~': 'Tilde', '`': 'Backtick',
    '|': 'Vertical bar', '+': 'Plus', '=': 'Equals', "\ ": 'Backslash',
    '\n': 'Newline (paragraph break)', '•': 'Bullet point', '–': 'En dash', '—': 'Em dash'
}
# User input for excluding symbols
exclude_input = st.text_input(
    "List symbols to exclude from punctuation analysis "
    "(separate with lowercase 'e'). Leave blank to include all:"
)
excluded_symbols = exclude_input.split('e') if exclude_input else []

# Update symbols based on user exclusions
symbols_to_use = ''.join(sym for sym in default_symbols if sym not in excluded_symbols)


if st.button("Run Analysis"):
    sentences1 = [sent.text for sent in nlp(text1).sents]
    sentences2 = [sent.text for sent in nlp(text2).sents]

    if len(sentences1) == 0 or len(sentences2) == 0:
        st.error("Please ensure both inputs contain valid sentences.")
    else:
        st.markdown("### Null hypothesis: The two texts are similar (if threshold >0.75) / dissimilar (if threshold <0.35) in their characteristics (Semantic, Vocabulary, Dependency, Punctuation). Dependent on user-defined thresholds. ")

        if analysis_type == "Sentence-by-Sentence":
            results, sims = sentence_wise_similarity(sentences1, sentences2, thresholds, symbols_to_use)
        else:
            results, sims = full_text_similarity(text1, text2, thresholds, symbols_to_use)

        results_tab, plot_tab, details_tab, worst_sentence_tab = st.tabs(["Similarity Results", "Visualization", "Similarity Measures Explained", "Worst Sentence"])

        with results_tab:
            measure_cols = st.columns(len(results))
            for idx, (measure, data) in enumerate(results.items()):
                with measure_cols[idx]:
                    st.markdown(f"### {measure} Similarity")
                    mean_score = data['Mean Similarity']
                    st.write(f"- Mean Similarity: {mean_score:.3f}")
                    if mean_score >= 0.8:
                        st.write("- 🔥 High similarity; likely same author")
                    elif mean_score >= 0.6:
                        st.write("- ✅ Moderately similar; could be same author")
                    elif mean_score >= 0.35: 
                        st.write("- ⚠️ Poorly similar, might not be same author")
                    else:
                        st.write("- 🚨 Low similarity; likely not same author")
        
                    if data['Statistic'] is not None:
                        st.write(f"- Statistic: {data['Statistic']:.3f}")
                        st.write(f"- p-value: {data['p-value']:.3f}")
        
                        if measure != 'Overall':
                            threshold = thresholds[measure]
                        else:
                            threshold = np.mean(list(thresholds.values()))
        # there are some incorrect interpertations here
                        if threshold > 0.75:
                            if data['Reject Null']:
                                interpretation = "Based on rejected null hypothesis and high threshold, texts are likely dissimilar."
                            else:
                                interpretation = "Based on accepted null hypothesis and high threshold, texts are likely similar."
                        elif threshold < 0.35:
                            if data['Reject Null']:
                                interpretation = "Based on rejected null hypothesis and low threshold, texts are likely similar."
                            else:
                                interpretation = "Based on accepted null hypothesis and low threshold, texts are likely dissimilar."
                        else:
                            interpretation = "Interpretation ambiguous due to moderate threshold."
        
                        st.write(f"- Reject Null Hypothesis: {'YES' if data['Reject Null'] else 'NO'}")
                        st.write(f"- Interpretation: {interpretation}")
        
                    else:
                        st.write("- Statistic: N/A (Single comparison)")
                        st.write("- p-value: N/A (Single comparison)")
                        st.write("- Reject Null Hypothesis: N/A (Single comparison)")
                        st.write("- Interpretation: N/A (Single comparison)")
# there is still some real statistical tomfoolery going on here-- I have to figure out why interpertations are wrong-- but the thought is there. 
        with details_tab:
            st.markdown("### Detailed Explanation of Similarity Measures")

            st.markdown("**Semantic Similarity (Cosine):** Measures meaning and contextual similarity by comparing embeddings generated by a language model.\n- *Similar example:* \"The cat sat on the mat.\" vs. \"The kitten rested on the rug.\"\n- *Dissimilar example:* \"I enjoy running marathons.\" vs. \"Quantum physics is fascinating.\"")

            st.markdown("**Vocabulary Similarity (Jaccard):** Assesses overlap in unique vocabulary used, independent of word order or grammar.\n- *Similar example:* \"Cooking pasta requires boiling water.\" vs. \"Boiling water is needed for cooking pasta.\"\n- *Dissimilar example:* \"Economics is challenging.\" vs. \"Flowers bloom in spring.\"")

            st.markdown("**Dependency Structure Similarity:** Compares grammatical structure using dependency parsing, focusing on sentence construction.\n- *Similar example:* \"He quickly ran home.\" vs. \"She slowly walked home.\"\n- *Dissimilar example:* \"The sun rises.\" vs. \"Can you help me move this table?\"")

            st.markdown("**Punctuation Similarity:** Evaluates the similarity in punctuation usage, style, and frequency.\n- *Similar example:* \"Wait! Are you coming?\" vs. \"Stop! Where are you going?\"\n- *Dissimilar example:* \"Hello, how are you?\" vs. \"I am fine\"")

            
        with plot_tab:
            measures = list(results.keys())
            mean_scores = [results[measure]['Mean Similarity'] for measure in measures]
        
            # Bar plot remains the same
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(measures, mean_scores, color=['skyblue', 'lightgreen', 'salmon', 'violet', 'gold'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Mean Similarity Score')
            ax.set_title('Mean Similarity Scores by Measure')
        
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
        
            st.pyplot(fig)
        
            # Only plot similarity vectors if sentence-by-sentence analysis
            if analysis_type == "Sentence-by-Sentence":
                fig, axes = plt.subplots(len(measures)-1, 1, figsize=(10, 4 * (len(measures)-1)))
                detailed_measures = ['Semantic', 'Vocabulary', 'Dependency', 'Punctuation']
                for idx, measure in enumerate(detailed_measures):
                    sim_vector = sims[measure]
                    threshold_vector = np.full_like(sim_vector, thresholds[measure])
        
                    axes[idx].plot(sim_vector, label=f"{measure} Similarity", color='blue')
                    axes[idx].plot(threshold_vector, label="Threshold", color='red', linestyle='--')
                    axes[idx].set_ylim(0, 1)
                    axes[idx].set_title(f"{measure} Similarity vs Threshold")
                    axes[idx].set_xlabel("Sentence Index")
                    axes[idx].set_ylabel("Similarity")
                    axes[idx].legend()
        
                plt.tight_layout()
                st.pyplot(fig)
        
        with worst_sentence_tab:
            if analysis_type == "Sentence-by-Sentence":
                st.markdown("### Least Similar Sentences by Measure")
        
                for measure in ['Semantic', 'Vocabulary', 'Dependency', 'Punctuation']:
                    sim_vector = sims[measure]
                    min_idx = np.argmin(sim_vector)
                    st.markdown(f"**{measure} Similarity:** {sim_vector[min_idx]:.3f}")
                    st.markdown(f"- **Known sample sentence:** `{sentences1[min_idx]}`")
                    st.markdown(f"- **Suspected sample sentence:** `{sentences2[min_idx]}`")
                    st.markdown("---")
            else:
                st.markdown("### Least Similar Sentences by Measure")
                st.markdown("N/A (Entire text analysis selected)")

        # Tab to display symbols usage
        with st.expander("Punctuation Symbols Used in Analysis"):
            st.markdown("### Symbols Currently Used for Punctuation Analysis")
            st.write("✅ - Symbol is included in the analysis")
            st.write("❌ - Symbol is excluded from the analysis")
            for sym, explanation in symbol_explanations.items():
                status = "✅" if sym in symbols_to_use else "❌"
                display_sym = sym if sym != '\n' else '\\n'
                st.write(f"{status} **{display_sym}**: {explanation}")

            
st.markdown("""
### 📚 **Packages used in this analysis:**
- **[Streamlit](https://streamlit.io/)** – Interactive web GUI
- **[NumPy](https://numpy.org/)** – Numerical computations
- **[spaCy](https://spacy.io/)** – Natural Language Processing toolkit
- **[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)** – Cosine similarity computation
- **[NLTK](https://www.nltk.org/)** – Edit distance calculation
- **[SciPy](https://docs.scipy.org/doc/scipy/)** – Statistical testing (ranksums)
- **[Matplotlib](https://matplotlib.org/)** – Plotting visualizations
""")
