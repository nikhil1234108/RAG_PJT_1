import re
import sys

from langchain_core.tools import tool
from functools import lru_cache
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import json
import os

nltk.download("punkt_tab",quiet=True)
nltk.download("averaged_perceptron_tagger",quiet=True)
nltk.download("stopwords",quiet=True)
nltk.download("wordnet",quiet=True)
nltk.download("punkt",quiet=True)

TECH_DICTIONARY = {
    # AI / LLM
    "LangChain", "LangGraph", "LlamaIndex", "OpenAI", "GPT-4", "GPT-3",
    "Claude", "Anthropic", "Gemini", "Mistral", "Ollama", "HuggingFace",
    "Transformers", "BERT", "DistilBERT", "RoBERTa", "T5", "LLAMA",
    "RAG", "FAISS", "ChromaDB", "Pinecone", "Weaviate", "Qdrant",

    # ML / DL frameworks
    "TensorFlow", "Keras", "PyTorch", "Scikit-learn", "XGBoost",
    "LightGBM", "CatBoost", "Pandas", "NumPy", "Matplotlib", "Seaborn",
    "Streamlit", "Gradio", "FastAPI", "Flask", "Django",

    # Computer Vision
    "YOLOv8", "YOLO", "OpenCV", "CNN", "ResNet", "VGG", "EfficientNet",
    "Detectron2", "MediaPipe", "Roboflow",

    # Dashboards / Monitoring
    "Grafana", "Kibana", "Datadog", "Prometheus", "Tableau", "PowerBI",
    "Metabase", "Looker", "KNIME", "Superset",

    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Neo4j", "GraphDB", "DuckDB",
    "Elasticsearch", "Redis", "Cassandra", "Snowflake", "BigQuery",
    "Supabase", "Firebase", "Airtable",

    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
    "Airflow", "dbt", "Airbyte", "Kafka", "Spark", "Databricks",
    "Lambda", "EC2", "S3", "Blob Storage", "Vercel", "Heroku",

    # Scraping / Automation
    "BeautifulSoup", "Scrapy", "Selenium", "Playwright", "Puppeteer",
    "Make.com", "Zapier", "n8n", "Retool",

    # Languages
    "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "SQL",
    "Node.js", "React", "Next.js", "FastAPI",

    # Other platforms
    "Salesforce", "Cin7", "Workday", "Shopify", "Stripe", "Twilio",
    "BlandAI", "Parquet", "REST API", "GraphQL", "gRPC",
}
@lru_cache(maxsize=1)
def _load_spacy():
    import spacy
    import torch

    model = "en_core_web_trf" if torch.cuda.is_available() else "en_core_web_md"
    try:
        return spacy.load(model)

    except OSError:
        return spacy.blank("en")

@tool
def tech_stack_extractor_tool(text:str) -> Dict[str, Any]:
    """
        Extracts all technology names, frameworks, libraries, databases,
        and platforms mentioned in a consulting project article.
        Uses spaCy NER + curated tech dictionary matching.
        Returns a categorised dict of found technologies.

        Use this when the user asks:
          - 'What tech was used in this project?'
          - 'Which projects use LangChain?'
          - 'Find all articles mentioning FAISS or YOLOv8'
        """

    found = set()
    text_upper = text.upper()

    for tech in TECH_DICTIONARY:
        pattern = r'\b' + re.escape(tech.upper()) + r'\b'
        if re.search(pattern, text_upper):
            found.add(tech)

    nlp = _load_spacy()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    extr_entities = set()
    for chunk in chunks:
        doc = nlp(chunk)
        for entity in doc.ents:
            if entity.label_ in ("ORG", "PRODUCT") and len(entity.text) > 2:
                extr_entities.add(entity.text)

    categories = {
        "AI_LLM": [],
        "ML_DL": [],
        "CV": [],
        "Dashboard": [],
        "Database": [],
        "Cloud_DevOps": [],
        "Scraping": [],
        "Languages": [],
        "Other": [],
    }

    ai_llm = {"LangChain", "LangGraph", "LlamaIndex", "OpenAI", "GPT-4", "GPT-3", "Claude",
              "Anthropic", "Gemini", "Mistral", "Ollama", "HuggingFace", "Transformers",
              "BERT", "DistilBERT", "RoBERTa", "T5", "LLAMA", "RAG", "FAISS", "ChromaDB",
              "Pinecone", "Weaviate", "Qdrant"}
    ml_dl = {"TensorFlow", "Keras", "PyTorch", "Scikit-learn", "XGBoost", "LightGBM",
             "CatBoost", "Pandas", "NumPy", "Streamlit", "Gradio", "FastAPI", "Flask", "Django"}
    cv = {"YOLOv8", "YOLO", "OpenCV", "CNN", "ResNet", "VGG", "EfficientNet",
          "Detectron2", "MediaPipe", "Roboflow"}
    dash = {"Grafana", "Kibana", "Datadog", "Prometheus", "Tableau", "PowerBI",
            "Metabase", "Looker", "KNIME", "Superset"}
    db = {"PostgreSQL", "MySQL", "MongoDB", "Neo4j", "GraphDB", "DuckDB",
          "Elasticsearch", "Redis", "Cassandra", "Snowflake", "BigQuery",
          "Supabase", "Firebase", "Airtable"}
    cloud = {"AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Airflow",
             "dbt", "Airbyte", "Kafka", "Spark", "Databricks", "Lambda", "EC2",
             "S3", "Blob Storage", "Vercel", "Heroku"}
    scraping = {"BeautifulSoup", "Scrapy", "Selenium", "Playwright", "Puppeteer",
                "Make.com", "Zapier", "n8n", "Retool"}
    langs = {"Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "SQL",
             "Node.js", "React", "Next.js"}

    for tech in found:
        if tech in ai_llm:
            categories["AI_LLM"].append(tech)
        elif tech in ml_dl:
            categories["ML_DL"].append(tech)
        elif tech in cv:
            categories["CV"].append(tech)
        elif tech in dash:
            categories["Dashboard"].append(tech)
        elif tech in db:
            categories["Database"].append(tech)
        elif tech in cloud:
            categories["Cloud_DevOps"].append(tech)
        elif tech in scraping:
            categories["Scraping"].append(tech)
        elif tech in langs:
            categories["Languages"].append(tech)
        else:
            categories["Other"].append(tech)


    categories = {k:sorted(v) for k, v in categories.items() if v}

    return {
        "technologies_found":found,
        "total_count":len(found),
        "by_category":categories,
        "extra_entities":extr_entities,
    }

def _count_syllables(word:str) -> int:
    word = word.lower()
    if word.endswith(("es","ed")):
        word = word[:-2]
    return max(1, len(re.findall(r"[aeiou]+",word)))

@lru_cache(maxsize=1)
def _load_stopwords() -> set:
    sw = set(w.upper() for w in stopwords.words('english'))
    sw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__))," .." ,"data", "stopwords")
    if os.path.exists(sw_dir):
        for filename in os.listdir(sw_dir):
            with open(os.path.join(sw_dir, filename), encoding='utf-8', errors="ignore") as f:
                for line in f:
                    word = line.strip().split("|")[0].strip().upper()
                    if word:
                        sw.add(word)
    return sw

def _clean_tokens(text:str) -> list:
    sw = _load_stopwords()
    return [word for word in word_tokenize(text) if word.upper() not in sw and word.isalpha()]

def _fog_stats(text:str) -> dict[str, Any]:
    words = [word for word in word_tokenize(text) if word.isalpha()]
    sentences = sent_tokenize(text)
    if not words or not sentences:
        return {}

    complex_words = [word for word in words if _count_syllables(word) > 2]
    avg_sent_len = len(words)/len(sentences)
    pct_complex = len(complex_words)/len(words)
    fog = 0.4*(avg_sent_len + pct_complex)

    return {
        "fog_index":round(fog,2),
        "avg_sent_len":round(avg_sent_len,2),
        "complex_words_len":len(complex_words),
        "words_len":len(words),
        "pct_complex":round(pct_complex,4)
    }

@tool
def complexity_classifier_tool(text:str) -> Dict[str, Any]:
    """Classifies project as Basic / Intermediate / Advanced using FOG + word stats."""

    stat = _fog_stats(text)
    fog = stat["fog_index"]
    pct_complex = stat["pct_complex"]
    n_words = stat["complex_words_len"]

    total = (3 if fog > 15 else 2 if fog > 10 else 1) + \
            (3 if pct_complex > 0.2 else 2 if pct_complex > 0.01 else 1) + \
            (3 if n_words > 1000 else 2 if n_words > 400 else 1)

    label = "Advanced" if total >=8 else "Intermediate" if total >=5 else "Basic"

    reason = {
        "Advanced": "High FOG index, long article, dense technical vocabulary",
        "Intermediate": "Moderate complexity, mix of technical and explanatory content",
        "Basic": "Low FOG index, shorter article, simpler vocabulary",
    }[label]

    return{
        "reason":reason,
        "fog_index":fog,
        "pct_complex":pct_complex,
        "n_words":n_words,
        "total":total,
        "complexity":label,
        "avg_sentence_len": stat["avg_sent_len"]
    }

TOPIC_KEYWORDS = {
    "AI / LLM / RAG":          ["langchain","rag","llm","gpt","chatbot","embedding",
                                 "vector","prompt","openai","agent","langgraph","llama"],
    "Computer Vision":          ["yolo","object detection","image classification","opencv",
                                 "cnn","vision","camera","video","detection","segmentation"],
    "Dashboard / Monitoring":   ["grafana","dashboard","monitoring","kibana","datadog",
                                 "visualization","metrics","prometheus","tableau","chart"],
    "Database / Graph":         ["neo4j","graphdb","postgresql","mongodb","elasticsearch",
                                 "database","sql","graph","node","relationship","duckdb"],
    "Data Engineering":         ["pipeline","etl","airflow","kafka","spark","airbyte",
                                 "dbt","parquet","ingestion","scraping","connector"],
    "ML / Forecasting":         ["forecasting","prediction","regression","classification",
                                 "model","training","feature engineering","scikit","xgboost"],
    "Automation / Integration": ["automation","make.com","zapier","airtable","integration",
                                 "workflow","trigger","n8n","retool","api"],
    "Research / Consulting":    ["research","analysis","strategy","governance","policy",
                                 "report","insight","cultural","private equity"],
}

@tool
def topic_classifier_tool(text:str)-> Dict[str, Any]:
    """Assigns article to a domain cluster based on keyword frequency."""

    text_lower = text.lower()
    scores = {t: sum(text_lower.count(k) for k in kws)
              for t, kws in TOPIC_KEYWORDS.items()}
    scores = {t:s for t, s in scores.items() if s>=2}
    if not scores:
        return {'primary_topic':'General','scores':{}}

    sorted_scores = dict(sorted(scores.items(), key = lambda x: x[1], reverse = True))
    keys = list(sorted_scores.keys())
    return {
        "primary_topic":keys[0],
        "secondry_topic":keys[1] if len(keys)>1 else None,
        "all_scores":sorted_scores,
        "is_multi_domain":len(scores)>2

    }

@tool
def readability_tool(text:str) -> Dict[str, Any]:
    """FOG index, avg sentence length, syllables per word, readability label."""

    stats = _fog_stats(text)
    if not stats:
        return {}
    words = [word for word in word_tokenize(text) if word.isalpha()]
    syllables_pw = round(sum(_count_syllables(word) for word in words),2)
    return {
        **stats,
        "syllables_pw":syllables_pw,
        "fog_index":"Hard" if stats['fog_index']>12
                    else "Medium" if stats['fog_index'] > 12 else "Easy"
    }

_SUMMARY_PROMPT = PromptTemplate(
    template = """
    You are a technical project analyst. Produce a structured summary.
    Article:{text}
    Output exactly:
    PROJECT: [one sentence — what was built]
    PROBLEM: [one sentence — what problem it solved]
    TECH: [comma-separated main technologies]
    OUTCOME: [one sentence — result or impact]
    DOMAIN: [one word — AI/CV/Dashboard/Database/Automation/Research]""".strip(),
input_variables = ["text"]
)

@tool
def project_summary_tool(text:str) -> Dict[str,str]:
    """LangChain chain that generates structured PROJECT/PROBLEM/TECH/OUTCOME/DOMAIN summary."""

    try:
        from chains.rag_chain import get_llm
        chain = _SUMMARY_PROMPT | get_llm()
        result = chain.invoke({'text':text[:5000]})
        output = result.content if hasattr(result, 'content') else str(result)
        parsed = {}
        for line in output.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':',1)
                parsed[key.strip()] = value.strip()
        return parsed if parsed else {'summary':output}
    except Exception as e:
        return {'error':str(e)}


@tool
def word_count_tool(text:str) -> Dict[str, int]:
    """Clean word count — stopwords and punctuation removed."""

    return {"word_count":len(_clean_tokens(text))}

@tool
def personal_pronouns_tool(text:str) -> Dict[str, int]:
    """Counts I, we, my, ours, us. Excludes standalone US (country)."""

    matches = [m for m in re.findall(r'\b(I|we|my|ours|us)\b',text,re.IGNORECASE)
               if m!='US']
    return {'len_personal_pronouns':len(matches)}

@tool
def syllable_per_word_tool(text:str) -> Dict[str, float]:
    """Average syllables per word. Higher = more technical vocabulary."""

    words = [word for word in word_tokenize(text) if word.isalpha()]
    if not words:
        return{
            'syllables_per_word':0,
        }
    return {'syllables_per_word':round(sum(_count_syllables(word) for word in words)/len(words),2)}

@tool
def avg_word_len_tool(text:str) -> Dict[str, float]:
    """Average word length. Higher = more technical vocabulary."""
    words = [word for word in word_tokenize(text) if word.isalpha()]
    if not words:
        return{
            'avg_word_len':0,
        }
    return {
        'avg_word_len':round(sum(len(word) for word in words)/len(words),4)
    }
@tool
def named_entities_tools(text:str) -> Dict[str, Any]:
    """
        spaCy NER — en_core_web_trf on GPU, en_core_web_sm on CPU.
        Returns entity counts by type (ORG, PRODUCT, GPE, PERSON etc.)
        """
    nlp = _load_spacy()
    doc = nlp(text[:5000])
    counts:Dict[str, int] = {}
    for ent in doc.ents:
        counts[ent.label_] = counts.get(ent.label_, 0)+1
    return {"entities":counts,"total_entities":len(doc.ents)}

_cluster_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","clusters.json")
def _load_clusters() -> dict:
    if not os.path.exists(_cluster_cache):
        return {}
    with open(_cluster_cache,"r") as f:
        return json.load(f)

@tool
def get_article_cluster_tool(url_id:str) -> Dict[str, Any]:
    """
        Returns which KMeans cluster an article belongs to and its peer articles.
        Use when user asks: 'Which cluster is the YOLOv8 article in?'
        """
    data = _load_clusters()
    if not data:
        return {"error":"clusters not built yet."}
    url_ids = data["kmeans"]["url_ids"]
    label_ids = data["kmeans"]["label_ids"]

    if url_id not in url_ids:
        return {"error":f"{url_id} not found"}
    idx = url_ids.index(url_id)
    cluster_id = label_ids[idx]
    peers = data["kmeans"]["peers"].get(f"{cluster_id}",[])
    return {
        "url_id":url_id,
        "cluster_id":cluster_id,
        "peers":[p for p in peers if p!=url_id]
    }

@tool
def find_similar_article_tool(url_id:str,top_k:int=5) ->Dict[str, Any]:
    """
        Finds most semantically similar articles using UMAP 2D coordinates.
        Use when user asks: 'Find articles similar to the YOLOv8 project'
        """
    data = _load_clusters()
    if not data:
        return {"error":"clusters not built yet."}

    km_data = data["kmeans"]
    url_ids = km_data["url_ids"]
    if url_id not in url_ids:
        return {"error":f"{url_id} not found"}
    embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "data", "article_embeddings.npy")
    if not os.path.exists(embeddings_path):
        return {"error":"Article_embeddings not found. Run Build_clusters first"}
    vectors = np.load(embeddings_path)
    norms = np.linalg.norm(vectors,axis=1,keepdims=True)
    vectors_norm = vectors/(norms+1e-9)

    idx = url_ids.index(url_id)
    query_vec = vectors_norm[idx]
    similarities = vectors_norm @ query_vec
    similarities[idx] = -1
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return {
        "url_id":url_id,
        "find_similar_articals":[{
            "url_id":url_ids[i],
            "similarity_score":round(similarities[i],4)
        }
        for i in top_indices
        ]
    }

ALL_TOOLS = ['tech_stack_extractor','complexity_classifier_tool','topic_classifier_tool',
             'readability_tool','project_summary_tool','word_count_tool','personal_pronouns_tool',
             'syllable_per_word_tool', 'avg_word_len_tool','named_entities_tool','get_article_cluster_tool',
             'find_similar_article_tool']

def analyse_article(text:str) -> Dict[str, Any]:
    """return all tools through batch output"""
    return {
        **tech_stack_extractor_tool.run(text),
        **complexity_classifier_tool.run(text),
        **topic_classifier_tool.run(text),
        **readability_tool.run(text),
        **word_count_tool.run(text),
        **project_summary_tool.run(text),
        **personal_pronouns_tool.run(text),
        **syllable_per_word_tool.run(text),
        **avg_word_len_tool.run(text),
        **named_entities_tools.run(text),
        **get_article_cluster_tool(text),
        **find_similar_article_tool(text)

    }





if __name__ == "__main__":
    text = """In this project, we built a YOLOv8-based real-time event detection system
    deployed on AWS using Docker and FastAPI. The client needed to monitor
    industrial camera feeds and trigger alerts when anomalies were detected.
    We trained a custom YOLOv8 model on 5000 labelled images using Python and
    PyTorch, achieving 94% mAP. The system streams video via OpenCV, runs
    inference, and pushes alerts to a Grafana dashboard via Prometheus metrics.
    """
    output = tech_stack_extractor_tool.invoke({"text": text})
    print(output)


