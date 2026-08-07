Smart Medicine Identifier

An AI-powered medicine identification system that recognizes medicines from images using OCR and Retrieval-Augmented Generation (RAG). The application extracts text from medicine packaging, identifies the medicine using semantic search, and provides detailed information including composition, uses, side effects, and manufacturer.

Features:

Upload an image of a medicine strip, bottle, or packaging
Extracts text using EasyOCR
Identifies medicines using Sentence Transformers + FAISS
Uses Google Gemini API when the medicine is not found locally
Automatically updates the local medicine dataset with newly identified medicines
Fast semantic search using vector embeddings
User-friendly Flask web interface

Technologies Used:

Programming Language
- Python
  
Machine Learning & AI
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS
- Retrieval-Augmented Generation (RAG)
- Google Gemini API

OCR
- EasyOCR
- OpenCV

Backend
- Flask

Data Processing
- Pandas
- NumPy

Fuzzy Matching
- RapidFuzz

How It Works
1. User uploads an image of a medicine.
2. EasyOCR extracts text from the packaging.
3. The extracted text is cleaned and processed.
4. Sentence Transformers convert the text into embeddings.
5. FAISS searches the local medicine database for the closest match.
6. If a match is found, medicine information is displayed.
7. If no suitable match exists, the application queries Google Gemini.
8. Newly identified medicines are added to the dataset and indexed automatically for future searches.


Project Structure
```
Smart-Medicine-Identifier/
│
├── app.py
├── requirements.txt
├── rag_ingest.py
│
├── dataset/
│   └── Medicine_Details.csv
│
├── utils/
│   ├── ocr.py
│   ├── matcher.py
│   ├── gemini_search.py
│   ├── update_dataset.py
│   ├── rebuild_index.py
│   ├── medicine_data.pkl
│   └── medicine_index.faiss
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── uploads/
│
└── README.md
```

## Author

**Merlyn Ann Mathew**

GitHub: https://github.com/merannmat
