# Vincent's BS Detector v1.0

*Prompted by Spite, Driven by Honesty*

A powerful and intuitive NLP-based (using spaCy) Streamlit web application designed to detect textual similarity, identify potential plagiarism, and analyze writing samples based on semantic structure, vocabulary, punctuation, and grammatical dependencies. Built & desiged by Vincent, with input from GPT-4.

**THIS IS A WIP PROJECT that has recieved about 3 hours of work.** Many of the measures of similariry were a practice in learning how to use those statisitcs; it is not clear that they are nesecarrily the best tools to use. Having said that, they seem to work alright. 

---

## 📖 Project Overview

Vincent's AI-BS Detector analyzes and compares two text inputs to determine their similarity across multiple linguistic and stylistic dimensions. It features (**WIP**):

- **Semantic Similarity**: Uses cosine similarity of text embeddings.
- **Vocabulary Similarity**: Measures Jaccard similarity based on unique word usage.
- **Dependency Structure Similarity**: Compares grammatical structures via dependency parsing.
- **Punctuation Similarity**: Evaluates similarity in punctuation frequency and usage.

Results include detailed statistics, visualizations, and sentence-level breakdowns.

**WIP**: With enough input data (the more, the better), Dependancy and Punctuation scores may be used to evaluate textual similarites reguardless of substantive content or topic of discussion between the text samples, as these measures evaluate sentence construction and punctuation usage. However, conclusions of this nature are far from certain, and users should remain cautious. 

---

## ⚙️ Architecture

- **Frontend**: Streamlit
- **Backend NLP**: spaCy
- **Similarity measures**:
  - Cosine similarity (`scikit-learn`)
  - Jaccard similarity (custom Python)
  - Dependency structure comparison (`NLTK`'s Edit Distance)
- **Statistical Testing**: Mann-Whitney U test (`SciPy.stats` ranksums)
- **Visualization**: Matplotlib

---

## 🧮 Math & Statistics

- **Cosine Similarity** (Semantic):
  \(\text{Similarity} = \frac{A \cdot B}{\|A\|\|B\|}\)

- **Jaccard Similarity** (Vocabulary):
  \(J(A,B) = \frac{|A \cap B|}{|A \cup B|}\)

- **Dependency Structure Similarity**:
  \(\text{Similarity} = 1 - \frac{\text{Edit Distance}}{\text{Max Length of Dependencies}}\)

- **Statistical Test**:

  - Mann-Whitney U test to determine if two sets of similarity scores differ significantly from user-defined thresholds.

---

## 🛠️ Packages Used

- [Streamlit](https://streamlit.io/) – Web interface
- [spaCy](https://spacy.io/) – Natural Language Processing toolkit
- [NumPy](https://numpy.org/) – Numerical computations
- [scikit-learn](https://scikit-learn.org/) – Cosine similarity
- [NLTK](https://www.nltk.org/) – Edit distance for grammatical comparison
- [SciPy](https://docs.scipy.org/doc/scipy/) – Statistical testing (ranksums)
- [Matplotlib](https://matplotlib.org/) – Plotting visualizations

---

## 🚀 Functionality

- Analyze either:

  - Sentence-by-Sentence: Compares sentences individually, offering detailed insights.
  - Entire Text: Offers overall document-level similarity.

- Visualize similarity scores vs thresholds clearly.

- Identify least similar sentence pairs per similarity measure.

- Customizable thresholds for tailored sensitivity.

- User-defined punctuation analysis.

---

## ▶️ How to Use

### Requirements

- Python 3.8+

### Installation

```bash
# Clone repository
git clone <your-github-repo-url>
cd <your-directory>

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run vincents_bs_detector.py
```

A new browser window will open at `http://localhost:8501` displaying the app.

---

## 📋 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool provides a statistical assessment of text similarity. It should be used alongside professional judgment, and is not a definitive indicator of plagiarism on its own. Similarly this tool does not possess the capability to identify if a piece of writing was produced by generative ai or not. All it is designed to do is A-B test two input texts and, using a variety of measures, adjudicate how likely it is that they share the same author, or a different author. 
---

