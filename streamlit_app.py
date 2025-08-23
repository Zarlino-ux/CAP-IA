import requests
import trafilatura
# ... rest of your code

# Importiamo le librerie che ci servono
import requests
import trafilatura

# --- Inserisci qui l'URL che vuoi testare ---
url_da_provare = "https://www.vatican.va/roman_curia/pontifical_councils/justpeace/documents/rc_pc_justpeace_doc_20060526_compendio-dott-soc_it.html"

print(f"Sto scaricando la pagina: {url_da_provare}")

# 1. Scarichiamo il contenuto della pagina web
try:
    response = requests.get(url_da_provare, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status() # Controlla se ci sono stati errori HTTP (es. 404)

    # 2. Usiamo trafilatura per estrarre il testo principale.
    #    Questa singola riga sostituisce tutto il lavoro che prima faceva BeautifulSoup!
    print("Estrazione del testo in corso con Trafilatura...")
    testo_pulito = trafilatura.extract(response.text)

    # 3. Stampiamo le prime 1000 lettere del risultato per vedere com'è
    if testo_pulito:
        print("\n--- INIZIO ESTRATTO ---")
        print(testo_pulito[:1000])
        print("--- FINE ESTRATTO ---")
    else:
        print("Trafilatura non ha trovato un corpo di testo principale in questa pagina.")

except requests.exceptions.RequestException as e:
    print(f"Errore durante il download della pagina: {e}")

from google.colab import drive
drive.mount('/content/drive')

# Install Ollama application
# CAP-IA

import streamlit as st
st.title("DSC ASSISTANT")

## Setup

# If your app depends on [Ollama](https://ollama.com/), please install it manually before running the app:

import subprocess
import streamlit as st

# Per eseguire comandi shell
result = subprocess.run(["ls", "-l"], capture_output=True, text=True)
st.write(result.stdout)
import streamlit as st

import streamlit as st
import pandas as pd

st.title("Installa Ollama")
st.markdown("Esegui questo comando nel tuo terminale:")

st.code("curl https://ollama.com/install.sh | sh", language="bash")

st.info("Nota: Questo comando deve essere eseguito in un terminale, non nello script Python.")

# Separatore
st.markdown("---")

# Sezione per l'installazione delle librerie Python
st.header("Installazione Librerie Python")

st.markdown("""
Esegui questi comandi nel tuo terminale per installare le librerie necessarie:
""")

st.code("""
pip install llama-index llama-index-vector-stores-chroma llama-index-embeddings-huggingface sentence-transformers llama-index-llms-ollama ollama pandas streamlit
""", language="bash")

# Separatore
st.markdown("---")

# Sezione per l'importazione delle librerie
st.header("Importazione Librerie")

st.markdown("Dopo l'installazione, puoi importare le librerie nel tuo script Python:")

# Mostra il codice di importazione
st.code("""
import pandas as pd
import streamlit as st

# Le altre importazioni verranno aggiunte qui dopo l'installazione
# import llama_index
# from llama_index.vector_stores import ChromaVectorStore
# from llama_index.embeddings import HuggingFaceEmbedding
# from llama_index.llms import Ollama
""", language="python")

# Separatore
st.markdown("---")

# Esempio di utilizzo del foglio Google
st.header("Caricamento dati da Google Sheets")

url_foglio_google = 'https://docs.google.com/spreadsheets/d/1Oqq2d1YodTM_qwCuQxud1O8AZdfRQIQQ/export?format=csv'

if st.button("Carica dati da Google Sheets"):
    try:
        # Carica i dati dal foglio Google
        df = pd.read_csv(url_foglio_google)
        st.success("Dati caricati correttamente!")
        
        # Mostra anteprima dati
        st.subheader("Anteprima dei dati")
        st.dataframe(df.head())
        
        # Mostra informazioni sul dataset
        st.subheader("Informazioni sul dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Righe", df.shape[0])
        with col2:
            st.metric("Colonne", df.shape[1])
        with col3:
            st.metric("Valori mancanti", df.isnull().sum().sum())
            
    except Exception as e:
        st.error(f"Errore nel caricamento dei dati: {e}")

# Footer
st.markdown("---")
st.caption("Applicazione Streamlit per gestione dati e Ollama")
print("--- STO LEGGENDO IL FOGLIO GOOGLE ONLINE ---")

# Leggiamo i dati in una tabella (DataFrame) senza intestazione
try:
    df = pd.read_csv(url_foglio_google, header=None)
    
    # STAMPIAMO L'INTERA TABELLA LETTA DALLO SCRIPT
    print("--- QUESTA È LA TABELLA ESATTA CHE LO SCRIPT VEDE ---")
    print(df)
    print("----------------------------------------------------")

except Exception as e:
    print(f"Si è verificato un errore durante la lettura del file: {e}")

# Start the Ollama server in the background
st.header("Gestione Servizio Ollama")

st.markdown("""
### Comandi utili per Ollama:

**Avviare il servizio:**
```bash
ollama serve &

# Wait for the server to start (a few seconds)
import time
import requests

print("Waiting for Ollama server to start...")
server_running = False
for i in range(60):
    try:
        response = requests.get("http://localhost:11434")
        if response.status_code == 200:
            server_running = True
            break
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(1)

if not server_running:
    print("Ollama server failed to start.")
    exit()

print("Ollama server is running.")

# Download the smaller Phi3 model
print("Downloading phi3 model...")
!ollama pull phi3
print("Phi3 model download complete.")

# ==================================================================================
# PROGETTO DSC-IA: FASE 1 - DIGITALIZZATORE WEB & PDF (Versione 2.0)
# ==================================================================================
# Questo script legge una lista di fonti da un Foglio Google.
# Se la fonte è un 'link web', estrae il testo dal sito.
# Se la fonte è un 'pdf', estrae il testo dal file corrispondente su Google Drive.
# Salva ogni testo come file .txt in una cartella su Google Drive.
# ==================================================================================

# --- FASE 0: INSTALLAZIONE DELLE LIBRERIE NECESSARIE ---
# Aggiungiamo 'pymupdf' per la gestione dei PDF
!pip install trafilatura pandas pymupdf

# --- FASE 1: IMPORT DELLE LIBRERIE E CONNESSIONE A GOOGLE DRIVE ---
import requests
import trafilatura
import pandas as pd
from google.colab import drive
import os
import fitz  # Questa è la libreria PyMuPDF

print("Connessione a Google Drive in corso...")
drive.mount('/content/drive')
print("✅ Connessione a Google Drive completata.")

# --- FASE 2: DEFINIZIONE DELLE NOSTRE FUNZIONI "OPERAIO" ---

def estrai_testo_da_url(url_da_analizzare):
    """Funzione per estrarre testo da un link web."""
    try:
        print(f"🔎 Sto analizzando l'URL: {url_da_analizzare[:70]}...")
        response = requests.get(url_da_analizzare, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        response.raise_for_status()
        testo_pulito = trafilatura.extract(response.text)
        return testo_pulito
    except requests.exceptions.RequestException as e:
        print(f"    ‼️ Errore durante l'analisi dell'URL: {e}")
        return None

def estrai_testo_da_pdf(percorso_completo_pdf):
    """NUOVA FUNZIONE per estrarre testo da un file PDF."""
    try:
        print(f"📄 Sto analizzando il PDF: {os.path.basename(percorso_completo_pdf)}...")
        testo_completo = ""
        with fitz.open(percorso_completo_pdf) as doc:
            for pagina in doc:
                testo_completo += pagina.get_text()
        return testo_completo
    except Exception as e:
        print(f"    ‼️ Errore durante l'analisi del PDF: {e}")
        return None

# --- FASE 3: LETTURA E ANALISI DEL FOGLIO GOOGLE ---
print("\nLettura dei dati dal Foglio Google...")
url_foglio_google = 'https://docs.google.com/spreadsheets/d/1OgIRL9zIXdqVS_P768JOZAvumIbd6Ejn/export?format=csv'
try:
    df = pd.read_csv(url_foglio_google, header=None)
    print("✅ Dati letti con successo dal Foglio Google!")
    colonna_titolo_pos = 0
    colonna_sorgente_pos = 1  # Ora questa colonna contiene sia link che nomi di file
    colonna_tipo_pos = 2

    # --- FASE 4: CICLO DI ESECUZIONE E SALVATAGGIO DEI FILE ---
    percorso_output = '/content/drive/MyDrive/DSC_Testi_Digitalizzati/'
    percorso_sorgenti_pdf = '/content/drive/MyDrive/Sorgenti_PDF/' # Cartella dove hai caricato i PDF
    if not os.path.exists(percorso_output):
        os.makedirs(percorso_output)

    print("\n--- INIZIO PROCESSO DI DIGITALIZZAZIONE ---")
    file_salvati = 0
    for indice, riga in df.iterrows():
        titolo = riga.get(colonna_titolo_pos)
        sorgente = riga.get(colonna_sorgente_pos)
        tipo_fonte = riga.get(colonna_tipo_pos)
        testo_estratto = None

        if pd.notna(tipo_fonte) and pd.notna(sorgente) and pd.notna(titolo):
            tipo_normalizzato = str(tipo_fonte).strip().lower()

            # NUOVA LOGICA: Scegliamo quale funzione usare in base al tipo
            if 'link' in tipo_normalizzato or 'web' in tipo_normalizzato:
                testo_estratto = estrai_testo_da_url(sorgente)
            
            elif 'pdf' in tipo_normalizzato:
                percorso_pdf = os.path.join(percorso_sorgenti_pdf, sorgente)
                if os.path.exists(percorso_pdf):
                    testo_estratto = estrai_testo_da_pdf(percorso_pdf)
                else:
                    print(f"    ‼️ Errore: File non trovato in Drive: '{sorgente}'")
            
            if testo_estratto:
                nome_file = "".join(c for c in titolo if c.isalnum() or c in (' ', '.', '-')).rstrip() + ".txt"
                percorso_salvataggio = os.path.join(percorso_output, nome_file)
                with open(percorso_salvataggio, "w", encoding='utf-8') as file:
                    file.write(testo_estratto)
                print(f"    ✅ SUCCESSO: '{nome_file}' salvato.")
                file_salvati += 1

    print("\n--- 🏁 PROCESSO DI DIGITALIZZAZIONE COMPLETATO ---")
    print(f"📄 File totali salvati: {file_salvati}")
    print(f"Controlla la tua cartella su Google Drive: {percorso_output}")

except Exception as e:
    print(f"‼️ ERRORE CRITICO: Impossibile leggere o processare il Foglio Google. Causa: {e}")
    print("Verifica che il link sia corretto e che il foglio sia accessibile.")

import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# --- CONFIGURATION ---
DATA_DIR = "/content/drive/My Drive/CAP AI testing/data"
CHROMA_DB_DIR = "/content/drive/My Drive/CAP AI testing/chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "phi3" # Using the smaller model
QUESTION = "What are the core principles of the Social Doctrine of the Church?"

# --- LOAD EMBEDDING MODEL AND CONFIGURE ---
try:
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    Settings.embed_model = embed_model
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit()

# --- INDEX MANAGEMENT (with lazy loading fix) ---
if os.path.exists(CHROMA_DB_DIR) and os.path.isdir(CHROMA_DB_DIR):
    print("ChromaDB index already exists. Loading existing index...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    chroma_collection = chroma_client.get_collection("cap_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    print("Existing index loaded successfully.")
else:
    print("No existing index found. Building index from documents...")
    # This is the lazy-loading fix
    reader = SimpleDirectoryReader(DATA_DIR)
    documents = reader.load_data()

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection("cap_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    print(f"Loaded {len(documents)} documents. Indexing them now...")
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model
    )
    print("Index built successfully.")

# ==============================================================================
# PROGETTO DSC-IA: FASE 1 - DIGITALIZZATORE WEB (Versione Finale)
# ==============================================================================
# Questo script legge un elenco di fonti da un Foglio Google,
# estrae il testo principale dalle fonti di tipo "link web" (e varianti),
# e salva ogni testo come file .txt su Google Drive.
# ==============================================================================

# --- FASE 0: INSTALLAZIONE DELLE LIBRERIE NECESSARIE ---
!pip install trafilatura pandas

# --- FASE 1: IMPORT DELLE LIBRERIE E CONNESSIONE A GOOGLE DRIVE ---
import requests
import trafilatura
import pandas as pd
from google.colab import drive
import os

print("Connessione a Google Drive in corso...")
drive.mount('/content/drive')
print("✅ Connessione a Google Drive completata.")

# --- FASE 2: DEFINIZIONE DELLA FUNZIONE "OPERAIO" PER L'ESTRAZIONE ---
def estrai_testo_da_url(url_da_analizzare):
    """
    Questa funzione prende un URL, scarica la pagina, estrae il testo 
    principale in modo intelligente e lo restituisce.
    """
    try:
        print(f"🔎 Sto analizzando l'URL: {url_da_analizzare[:70]}...")
        response = requests.get(url_da_analizzare, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        response.raise_for_status()
        testo_pulito = trafilatura.extract(response.text)
        return testo_pulito
    except requests.exceptions.RequestException as e:
        print(f"    ‼️ Errore durante l'analisi dell'URL: {e}")
        return None

# --- FASE 3: LETTURA E ANALISI DEL FOGLIO GOOGLE ---
print("\nLettura dei dati dal nuovo Foglio Google...")
# NUOVO LINK: Utilizziamo il link corretto in formato CSV.
url_foglio_google = 'https://docs.google.com/spreadsheets/d/1OgIRL9zIXdqVS_P768JOZAvumIbd6Ejn/export?format=csv'
try:
    df = pd.read_csv(url_foglio_google, header=None)
    print("✅ Dati letti con successo dal Foglio Google!")

    # Stabiliamo la posizione numerica delle nostre colonne
    colonna_titolo_pos = 0
    colonna_link_pos = 1
    colonna_tipo_pos = 2

    # SEZIONE DI DEBUG: Mostriamo cosa c'è nella colonna del "Tipo"
    print("\n--- Analisi Contenuto Colonna 'Tipo' ---")
    valori_unici = set(df[colonna_tipo_pos].dropna().astype(str).str.strip().str.lower())
    print(f"Lo script ha trovato questi valori unici nella colonna 'Tipo': {valori_unici}")
    print("----------------------------------------\n")

    # --- FASE 4: CICLO DI ESECUZIONE E SALVATAGGIO DEI FILE ---
    percorso_output = '/content/drive/MyDrive/DSC_Testi_Digitalizzati/'
    if not os.path.exists(percorso_output):
        os.makedirs(percorso_output)
        print(f"📂 Cartella di output creata: {percorso_output}")

    print("--- INIZIO PROCESSO DI DIGITALIZZAZIONE ---")

    file_salvati = 0
    for indice, riga in df.iterrows():
        titolo = riga.get(colonna_titolo_pos)
        link = riga.get(colonna_link_pos)
        tipo_fonte = riga.get(colonna_tipo_pos)

        if pd.notna(tipo_fonte) and pd.notna(link) and pd.notna(titolo):
            parole_chiave_web = ['link', 'web', 'enciclica', 'esortazione', 'costituzione', 'lettera']
            tipo_normalizzato = str(tipo_fonte).strip().lower()

            if any(parola in tipo_normalizzato for parola in parole_chiave_web):
                testo_estratto = estrai_testo_da_url(link)
                
                if testo_estratto:
                    nome_file = "".join(c for c in titolo if c.isalnum() or c in (' ', '.', '-')).rstrip() + ".txt"
                    percorso_salvataggio = os.path.join(percorso_output, nome_file)
                    
                    with open(percorso_salvataggio, "w", encoding='utf-8') as file:
                        file.write(testo_estratto)
                    print(f"    ✅ SUCCESSO: '{nome_file}' salvato.")
                    file_salvati += 1

    print("\n--- 🏁 PROCESSO DI DIGITALIZZAZIONE COMPLETATO ---")
    print(f"📄 File totali salvati: {file_salvati}")
    print(f"Controlla la tua cartella su Google Drive: {percorso_output}")

except Exception as e:
    print(f"‼️ ERRORE CRITICO: Impossibile leggere o processare il Foglio Google. Causa: {e}")
    print("Verifica che il link sia corretto e che il foglio sia accessibile pubblicamente.")

import google.generativeai as genai
from google.colab import userdata

try:
    # 1. Load the API key from Colab's Secrets Manager
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)

    # 2. Try a simple, basic call to the Google AI service
    print("✅ Key loaded successfully. Attempting to connect to Google's services...")
    model_list = genai.list_models()
    
    # 3. If the call succeeds, print a success message
    print("\n✅ SUCCESS! The connection was established. The API key is valid.")

except Exception as e:
    # 4. If the call fails, print an error message
    print(f"\n‼️ ERROR: The connection failed.")
    print(f"Error details: {e}")

# --- RUNNING QUERY ---
QUESTION = "What are the core principles of the Social Doctrine of the Church?"

custom_system_prompt = (
    "You are an expert on business ethics, the Social Doctrine of the Church, and sustainable development. "
    "Your purpose is to answer questions by synthesizing information from the provided documents. "
    "Your answers should be integrated and human-sounding, connecting the different sources where appropriate. "
    "Always reference the documents from which you draw information to support your claims. "
    "If you cannot find relevant information, state that explicitly and do not invent an answer."
)

try:
    llm = Ollama(model="phi3", request_timeout=360.0)
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    exit()

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="compact",
    similarity_top_k=5,
    system_prompt=custom_system_prompt
)

print(f"Asking the bot: '{QUESTION}'")
response = query_engine.query(QUESTION)
print("\n--- Bot's Answer ---")
print(response)

!pip install trafilatura

# ==================================================================================
# PROGETTO DSC-IA: FASE 2.1 - CREAZIONE DEL DATABASE VETTORIALE (Versione Corretta)
# ==================================================================================
# Questo script legge i testi digitalizzati dalla cartella '/DSC_Testi_Digitalizzati',
# li suddivide in frammenti (chunk), li trasforma in vettori (embedding) usando
# l'API di Google e crea un database di conoscenza FAISS.
# ==================================================================================

# --- FASE 0: INSTALLAZIONE DELLE LIBRERIE NECESSARIE ---
print("⚙️ Installing necessary libraries...")
# We use -q (quiet) and -U (upgrade) for a clean installation
!pip install -q -U google-generativeai langchain-google-genai langchain langchain-community faiss-cpu

# --- FASE 1: IMPORT E CONFIGURAZIONE ---
import os
import time
import math
import google.generativeai as genai
from google.colab import userdata
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from google.colab import drive

print("📚 Imports completed.")

# Securely configure the Google API key from Colab Secrets
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("API Key is empty")
    
    # Configure both genai and set environment variable for langchain
    genai.configure(api_key=GOOGLE_API_KEY)
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    
    print("✅ Google API Key configured successfully.")
    
    # Test the API key with a simple call
    print("🔍 Testing API connection...")
    try:
        # Simple test to verify the API key works
        models = genai.list_models()
        print("✅ API connection test successful.")
    except Exception as test_e:
        print(f"⚠️ API test failed: {test_e}")
        print("🔧 This might indicate quota issues or invalid API key.")
        
except Exception as e:
    print("‼️ ERROR: API Key not found or invalid. Please ensure it is saved in Colab's 'Secrets' with the name GOOGLE_API_KEY.")
    print(f"Error details: {e}")
    # Stop execution if the key is not present
    raise e

# Mount Google Drive to access our files
print("🚗 Connecting to Google Drive...")
drive.mount('/content/drive')
print("✅ Connection to Google Drive completed.")


# --- FASE 2: CARICAMENTO DEI TESTI DIGITALIZZATI ---
# The path to the folder containing the .txt files from Phase 1
text_path = '/content/drive/MyDrive/DSC_Testi_Digitalizzati/'
loaded_documents = []

print(f"\n📖 Loading texts from the folder: {text_path}")
if not os.path.exists(text_path):
    print(f"‼️ ERROR: The folder {text_path} was not found. Please ensure the path is correct.")
else:
    file_list = os.listdir(text_path)
    txt_files = [f for f in file_list if f.endswith(".txt")]
    
    if not txt_files:
        print("⚠️ No .txt files found in the specified folder.")
    else:
        print(f"📁 Found {len(txt_files)} .txt files to process.")
        
        for file_name in txt_files:
            file_path = os.path.join(text_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Add only if the file is not empty
                        loaded_documents.append({'content': content, 'source': file_name})
                        print(f"    ✅ Loaded: {file_name} ({len(content)} characters)")
                    else:
                        print(f"    ⚠️ Skipped empty file: {file_name}")
            except Exception as e:
                print(f"    ❌ Error reading file {file_name}: {e}")
                
        print(f"✅ Successfully loaded {len(loaded_documents)} valid text documents.")


# --- FASE 3: SUDDIVISIONE DEI TESTI (CHUNKING) ---
if loaded_documents:
    print("\n🧩 Splitting texts into chunks...")
    # We create a splitter that breaks text into 1500-character blocks
    # with a 200-character overlap to preserve context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for doc in loaded_documents:
        text_fragments = text_splitter.split_text(doc['content'])
        for i, fragment in enumerate(text_fragments):
            if fragment.strip():  # Only add non-empty fragments
                chunks.append({
                    'content': fragment.strip(),
                    'source': doc['source'],
                    'chunk_id': f"{doc['source']}_chunk_{i+1}"
                })
    
    print(f"✅ Texts split into {len(chunks)} valid fragments.")
    
    # Show some statistics
    avg_chunk_size = sum(len(chunk['content']) for chunk in chunks) / len(chunks)
    print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")
else:
    chunks = []
    print("⚠️ No documents were loaded, so no chunks were created.")


# --- FASE 4: VETTORIZZAZIONE A PACCHETTI (BATCH EMBEDDING) ---
if chunks:
    print("\n🧠 Creating meaning vectors (embedding)...")
    
    try:
        # Initialize embedding model with explicit configuration
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_document"
        )
        print("✅ Embedding model initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize embedding model: {e}")
        raise e
    
    # Set the batch size to respect the API's free tier rate limits
    batch_size = 20  # Further reduced for stability
    vector_db = None
    total_batches = math.ceil(len(chunks) / batch_size)

    print(f"📊 Processing {len(chunks)} chunks in {total_batches} batches of {batch_size}...")
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_texts = [chunk['content'] for chunk in batch_chunks]
        batch_metadata = [{'source': chunk['source'], 'chunk_id': chunk['chunk_id']} for chunk in batch_chunks]
        
        current_batch = i // batch_size + 1
        print(f"    🔄 Processing batch {current_batch}/{total_batches} ({len(batch_texts)} chunks)...")
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if i == 0:
                    # Create the database with the first batch
                    vector_db = FAISS.from_texts(
                        texts=batch_texts, 
                        embedding=embeddings_model,
                        metadatas=batch_metadata
                    )
                    print(f"    ✅ Created initial vector database with {len(batch_texts)} vectors")
                else:
                    # Add subsequent batches to the existing database
                    vector_db.add_texts(
                        texts=batch_texts,
                        metadatas=batch_metadata
                    )
                    print(f"    ✅ Added {len(batch_texts)} vectors to database")
                
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e).lower()
                retry_count += 1
                
                if any(keyword in error_msg for keyword in ["quota", "rate", "limit", "503", "timeout", "unavailable"]):
                    wait_time = min(60 * retry_count, 300)  # Cap at 5 minutes
                    print(f"    ⏰ API limit/timeout (attempt {retry_count}/{max_retries}). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    
                    if retry_count >= max_retries:
                        print(f"    ❌ Max retries reached for batch {current_batch}.")
                        print(f"    💡 Suggestion: Wait a few minutes and restart from this batch.")
                        print(f"    📊 Progress saved: {i} out of {len(chunks)} chunks processed.")
                        raise e
                else:
                    print(f"    ❌ Unexpected error in batch {current_batch}: {e}")
                    print(f"    🔍 Error type: {type(e).__name__}")
                    raise e
        
        # Longer pause between batches for free tier
        if i + batch_size < len(chunks):  # Don't wait after the last batch
            wait_time = 15  # Increased pause
            print(f"    ⏱️ Pausing {wait_time} seconds before next batch...")
            time.sleep(wait_time)

    print("✅ Vector database created successfully!")
    print(f"📊 Total vectors in database: {vector_db.index.ntotal}")

    # --- FASE 5: SALVATAGGIO DEL DATABASE SU DRIVE ---
    db_save_path = '/content/drive/MyDrive/DSC_Vector_DB'
    print(f"\n💾 Saving the vector database to: {db_save_path}")
    
    try:
        if not os.path.exists(db_save_path):
            os.makedirs(db_save_path)
            print(f"    📁 Created directory: {db_save_path}")
        
        vector_db.save_local(db_save_path)
        print("    ✅ Vector database saved successfully!")
        
        # Verify the save was successful
        saved_files = os.listdir(db_save_path)
        print(f"    📋 Files saved: {saved_files}")
        
    except Exception as e:
        print(f"    ❌ Error saving database: {e}")
        raise e

    # --- FASE 6: TEST OPZIONALE DEL DATABASE ---
    print(f"\n🧪 Testing the vector database...")
    try:
        # Test a simple similarity search
        test_query = "filosofia"
        test_results = vector_db.similarity_search(test_query, k=3)
        print(f"    ✅ Test successful! Found {len(test_results)} results for query '{test_query}'")
        
        # Show a sample result
        if test_results:
            sample = test_results[0]
            print(f"    📝 Sample result preview: {sample.page_content[:100]}...")
            if hasattr(sample, 'metadata') and sample.metadata:
                print(f"    📎 Source: {sample.metadata.get('source', 'Unknown')}")
    except Exception as e:
        print(f"    ⚠️ Test failed, but database was saved: {e}")

    print("\n--- 🏁 PROCESSO COMPLETATO CON SUCCESSO! ---")
    print("🎉 Il 'Cervello DSC-IA' è stato creato e salvato con successo!")
    print(f"📊 Statistiche finali:")
    print(f"    • Documenti processati: {len(loaded_documents)}")
    print(f"    • Frammenti creati: {len(chunks)}")
    print(f"    • Vettori nel database: {vector_db.index.ntotal}")
    print(f"    • Posizione database: {db_save_path}")

else:
    print("\n--- 🏁 PROCESSO INTERROTTO ---")
    print("❌ Nessun frammento da vettorializzare. Il database non è stato creato.")
    print("💡 Suggerimenti:")
    print("   • Verifica che la cartella contenga file .txt")
    print("   • Controlla che i file non siano vuoti")
    print("   • Verifica il percorso della cartella")

print("\n" + "="*80)
print("🤖 DSC-IA Vector Database Creation Script - COMPLETED")
print("="*80)

# ==================================================================================
# PROGETTO DSC-IA: FASE 3.1 - INTERFACCIA DI DIALOGO CON IL DATABASE (Versione Corretta)
# ==================================================================================
# Questo script carica il database vettoriale precedentemente creato,
# inizializza un modello di IA generativa e permette di porre domande
# per ricevere risposte basate sui documenti forniti.
# ==================================================================================

# --- FASE 0: INSTALLAZIONE DELLE LIBRERIE NECESSARIE ---
print("⚙️ Installazione delle librerie necessarie...")
!pip install -q -U google-generativeai langchain-google-genai langchain langchain-community faiss-cpu

# --- FASE 1: IMPORT E CONFIGURAZIONE ---
import os
import google.generativeai as genai
from google.colab import userdata
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from google.colab import drive

print("📚 Importazioni completate.")

# Configurazione della chiave API di Google in modo sicuro e robusto
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("API Key è vuota")
    
    # Configura sia genai che la variabile d'ambiente per langchain
    genai.configure(api_key=GOOGLE_API_KEY)
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    
    print("✅ Chiave API di Google configurata con successo.")
    
    # Test della connessione API
    print("🔍 Test della connessione API...")
    try:
        models = list(genai.list_models())
        print("✅ Test della connessione API riuscito.")
    except Exception as test_e:
        print(f"⚠️ Test API fallito: {test_e}")
        print("🔧 Questo potrebbe indicare problemi di quota o chiave API non valida.")
        
except Exception as e:
    print("‼️ Errore: Chiave API non trovata o non valida.")
    print(f"Dettagli errore: {e}")
    print("Assicurati che sia salvata nei 'Secrets' di Colab con il nome 'GOOGLE_API_KEY'.")
    raise e

# Montaggio di Google Drive
print("🚗 Connessione a Google Drive in corso...")
drive.mount('/content/drive')
print("✅ Connessione a Google Drive completata.")


# --- FASE 2: CARICAMENTO DEL "CERVELLO DSC-IA" (DATABASE VETTORIALE) ---
percorso_db = '/content/drive/MyDrive/DSC_Vector_DB'
print(f"\n🧠 Caricamento del database di conoscenza da: {percorso_db}")

if not os.path.exists(percorso_db):
    print(f"‼️ ERRORE CRITICO: La cartella del database vettoriale non è stata trovata.")
    print(f"Percorso cercato: {percorso_db}")
    print("💡 Possibili soluzioni:")
    print("   1. Assicurati di aver completato con successo lo script di vettorializzazione")
    print("   2. Verifica che il percorso sia corretto")
    print("   3. Controlla che i file siano stati salvati correttamente su Google Drive")
    
    # Mostra i contenuti della cartella padre per debug
    parent_dir = os.path.dirname(percorso_db)
    if os.path.exists(parent_dir):
        print(f"\n🔍 Contenuti della cartella {parent_dir}:")
        for item in os.listdir(parent_dir):
            print(f"   - {item}")
    
    raise FileNotFoundError(f"Database vettoriale non trovato in {percorso_db}")

else:
    try:
        print("🔄 Inizializzazione del modello di embedding...")
        # Inizializziamo lo stesso modello di embedding usato per creare il database
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        print("✅ Modello di embedding inizializzato.")
        
        print("🔄 Caricamento del database vettoriale...")
        # Carichiamo il database vettoriale da locale
        vector_db = FAISS.load_local(
            percorso_db, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        
        # Verifica che il database sia caricato correttamente
        num_vectors = vector_db.index.ntotal
        print(f"✅ Database di conoscenza caricato con successo.")
        print(f"📊 Numero di vettori nel database: {num_vectors}")
        
        if num_vectors == 0:
            raise ValueError("Il database è vuoto - nessun vettore trovato")
            
    except Exception as e:
        print(f"❌ Errore durante il caricamento del database: {e}")
        print(f"🔍 Tipo errore: {type(e).__name__}")
        raise e

    # --- FASE 3: PREPARAZIONE DELLA CATENA DI DOMANDA E RISPOSTA (QA CHAIN) ---
    print("\n🗣️ Preparazione del sistema di dialogo (RetrievalQA)...")
    
    try:
        # Inizializziamo il modello generativo che formulerà le risposte
        print("🔄 Inizializzazione del modello di linguaggio...")
        
        # Lista dei modelli da provare in ordine di preferenza
        modelli_da_provare = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro"
        ]
        
        llm = None
        for modello in modelli_da_provare:
            try:
                print(f"🔍 Tentativo con modello: {modello}")
                llm = ChatGoogleGenerativeAI(
                    model=modello, 
                    temperature=0.3,
                    google_api_key=GOOGLE_API_KEY,
                    convert_system_message_to_human=True
                )
                # Test del modello
                test_response = llm.invoke("Test")
                print(f"✅ Modello {modello} funziona correttamente.")
                break
            except Exception as e:
                print(f"❌ Modello {modello} non disponibile: {e}")
                continue
        
        if llm is None:
            raise Exception("Nessun modello Gemini disponibile. Verifica la tua chiave API e i permessi.")
        
        print("✅ Modello di linguaggio inizializzato.")
        
        # Creiamo un "retriever" ottimizzato
        print("🔄 Configurazione del retriever...")
        retriever = vector_db.as_retriever(
            search_type="similarity", 
            search_kwargs={
                "k": 5,  # Numero di documenti da recuperare
                "fetch_k": 20  # Numero di documenti da considerare prima del ranking
            }
        )
        print("✅ Retriever configurato.")
        
        # Template di prompt personalizzato per risposte più accurate
        custom_prompt_template = """Sei un assistente esperto in Dottrina Sociale della Chiesa e filosofia. 
Usa SOLO le informazioni fornite nel contesto per rispondere alla domanda.
Se le informazioni non sono sufficienti per una risposta completa, dillo chiaramente.

Contesto dai documenti:
{context}

Domanda: {question}

Rispondi in modo chiaro, strutturato e preciso, citando quando possibile i concetti specifici dai documenti forniti."""

        custom_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Assembliamo la catena completa con prompt personalizzato
        print("🔄 Creazione della catena QA...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt},
            verbose=False  # Imposta True per debug
        )
        print("✅ Sistema di dialogo pronto e ottimizzato.")

    except Exception as e:
        print(f"❌ Errore durante la preparazione del sistema di dialogo: {e}")
        print(f"🔍 Tipo errore: {type(e).__name__}")
        raise e

    # --- FASE 4: FUNZIONE PER PORRE DOMANDE ---
    def poni_domanda(domanda, mostra_fonti=True, max_lunghezza_estratto=150):
        """
        Funzione per porre una domanda al sistema DSC-IA
        
        Args:
            domanda (str): La domanda da porre
            mostra_fonti (bool): Se mostrare le fonti utilizzate
            max_lunghezza_estratto (int): Lunghezza massima dell'estratto dalle fonti
        """
        print(f"\n❓ Domanda: {domanda}")
        print("\n🔄 Elaborazione della risposta in corso...")
        
        try:
            # Eseguiamo la catena con la domanda
            risultato = qa_chain({"query": domanda})
            
            # Stampiamo la risposta
            print("\n🤖 Risposta dall'IA DSC:")
            print("=" * 60)
            print(risultato['result'])
            print("=" * 60)
            
            # Stampiamo le fonti se richiesto
            if mostra_fonti and risultato.get('source_documents'):
                print(f"\n📚 Fonti utilizzate ({len(risultato['source_documents'])} documenti):")
                print("-" * 50)
                
                for i, doc in enumerate(risultato['source_documents'], 1):
                    # Estrai metadati in modo sicuro
                    source = doc.metadata.get('source', 'Fonte sconosciuta')
                    chunk_id = doc.metadata.get('chunk_id', 'N/A')
                    
                    # Tronca l'estratto se troppo lungo
                    estratto = doc.page_content[:max_lunghezza_estratto]
                    if len(doc.page_content) > max_lunghezza_estratto:
                        estratto += "..."
                    
                    print(f"  {i}. 📄 {source}")
                    print(f"     🆔 Chunk: {chunk_id}")
                    print(f"     📝 Estratto: \"{estratto}\"")
                    print()
            
            return risultato
            
        except Exception as e:
            print(f"❌ Errore durante l'elaborazione della domanda: {e}")
            print(f"🔍 Tipo errore: {type(e).__name__}")
            return None

    # --- FASE 5: ESEMPI DI UTILIZZO ---
    print("\n" + "="*80)
    print("🎉 SISTEMA DSC-IA PRONTO ALL'USO!")
    print("="*80)
    
    # Esempio 1: Domanda predefinita
    print("\n📋 Esempio 1 - Domanda sul principio di sussidiarietà:")
    esempio_domanda_1 = "Spiegami il principio di sussidiarietà con un esempio pratico per una piccola impresa."
    poni_domanda(esempio_domanda_1)
    
    print("\n" + "-"*50)
    
    # Esempio 2: Seconda domanda
    print("\n📋 Esempio 2 - Domanda sulla Dottrina Sociale:")
    esempio_domanda_2 = "Quali sono i principi fondamentali della Dottrina Sociale della Chiesa?"
    poni_domanda(esempio_domanda_2)
    
    # --- ISTRUZIONI PER L'USO CONTINUO ---
    print(f"\n" + "="*80)
    print("💡 COME CONTINUARE AD USARE IL SISTEMA:")
    print("="*80)
    print("Per porre nuove domande, usa la funzione:")
    print("   poni_domanda(\"La tua domanda qui\")")
    print("")
    print("Esempi:")
    print("   poni_domanda(\"Cos'è la destinazione universale dei beni?\")")
    print("   poni_domanda(\"Spiegami il rapporto tra lavoro e dignità umana\")")
    print("   poni_domanda(\"Come si applica la giustizia sociale nell'economia?\", mostra_fonti=False)")
    print("")
    print("Parametri opzionali:")
    print("   - mostra_fonti=True/False (default: True)")
    print("   - max_lunghezza_estratto=numero (default: 150)")
    print("="*80)
    
    # --- STATISTICHE FINALI ---
    print(f"\n📊 STATISTICHE DEL SISTEMA:")
    print(f"   • Database caricato: ✅")
    print(f"   • Vettori disponibili: {num_vectors}")
    print(f"   • Modello LLM: {llm.model_name if hasattr(llm, 'model_name') else 'Gemini (versione rilevata automaticamente)'}")
    print(f"   • Modello Embedding: Google Embedding-001")
    print(f"   • Documenti per query: 5")
    print(f"   • Sistema pronto: ✅")

poni_domanda("come si partecipa alla ricapitolazione in Cristo")

poni_domanda("What is subsidiarity")

# ==================================================================================
# PROGETTO DSC-IA: FASE 3.1 - INTERFACCIA DI DIALOGO CON IL DATABASE (Versione Corretta)
# ==================================================================================
# Questo script carica il database vettoriale precedentemente creato,
# inizializza un modello di IA generativa e permette di porre domande
# per ricevere risposte basate sui documenti forniti.
# ==================================================================================

# --- FASE 0: INSTALLAZIONE DELLE LIBRERIE NECESSARIE ---
print("⚙️ Installazione delle librerie necessarie...")
!pip install -q -U google-generativeai langchain-google-genai langchain langchain-community faiss-cpu

# --- FASE 1: IMPORT E CONFIGURAZIONE ---
import os
import google.generativeai as genai
from google.colab import userdata
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from google.colab import drive
from langchain_community.embeddings import HuggingFaceEmbeddings  # Added missing import

print("📚 Importazioni completate.")

# Configurazione della chiave API di Google in modo sicuro e robusto
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("API Key è vuota")
    
    # Configura sia genai che la variabile d'ambiente per langchain
    genai.configure(api_key=GOOGLE_API_KEY)
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    
    print("✅ Chiave API di Google configurata con successo.")
    
    # Test della connessione API
    print("🔍 Test della connessione API...")
    try:
        models = list(genai.list_models())
        print("✅ Test della connessione API riuscito.")
    except Exception as test_e:
        print(f"⚠️ Test API fallito: {test_e}")
        print("🔧 Questo potrebbe indicare problemi di quota o chiave API non valida.")
        
except Exception as e:
    print("‼️ Errore: Chiave API non trovata o non valida.")
    print(f"Dettagli errore: {e}")
    print("Assicurati che sia salvata nei 'Secrets' di Colab con il nome 'GOOGLE_API_KEY'.")
    raise e

# Montaggio di Google Drive
print("🚗 Connessione a Google Drive in corso...")
drive.mount('/content/drive')
print("✅ Connessione a Google Drive completata.")


# --- FASE 2: CARICAMENTO DEL "CERVELLO DSC-IA" (DATABASE VETTORIALE) ---
percorso_db = '/content/drive/MyDrive/DSC_Vector_DB'
print(f"\n🧠 Caricamento del database di conoscenza da: {percorso_db}")

if not os.path.exists(percorso_db):
    print(f"‼️ ERRORE CRITICO: La cartella del database vettoriale non è stata trovata.")
    print(f"Percorso cercato: {percorso_db}")
    print("💡 Possibili soluzioni:")
    print("   1. Assicurati di aver completato con successo lo script di vettorializzazione")
    print("   2. Verifica che il percorso sia corretto")
    print("   3. Controlla che i file siano stati salvati correttamente su Google Drive")
    
    # Mostra i contenuti della cartella padre per debug
    parent_dir = os.path.dirname(percorso_db)
    if os.path.exists(parent_dir):
        print(f"\n🔍 Contenuti della cartella {parent_dir}:")
        for item in os.listdir(parent_dir):
            print(f"   - {item}")
    
    raise FileNotFoundError(f"Database vettoriale non trovato in {percorso_db}")

else:
    try:
        print("🔄 Inizializzazione del modello di embedding...")
        # Inizializziamo lo stesso modello di embedding usato per creare il database
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        print("✅ Modello di embedding inizializzato.")
        
        print("🔄 Caricamento del database vettoriale...")
        # Carichiamo il database vettoriale da locale
        vector_db = FAISS.load_local(
            percorso_db, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        
        # Verifica che il database sia caricato correttamente
        num_vectors = vector_db.index.ntotal
        print(f"✅ Database di conoscenza caricato con successo.")
        print(f"📊 Numero di vettori nel database: {num_vectors}")
        
        if num_vectors == 0:
            raise ValueError("Il database è vuoto - nessun vettore trovato")
            
    except Exception as e:
        print(f"❌ Errore durante il caricamento del database: {e}")
        print(f"🔍 Tipo errore: {type(e).__name__}")
        raise e

    # --- FASE 3: PREPARAZIONE DELLA CATENA DI DOMANDA E RISPOSTA (QA CHAIN) ---
    print("\n🗣️ Preparazione del sistema di dialogo (RetrievalQA)...")
    
    try:
        # Inizializziamo il modello generativo che formulerà le risposte
        print("🔄 Inizializzazione del modello di linguaggio...")
        
        # Lista dei modelli da provare in ordine di preferenza
        modelli_da_provare = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro"
        ]
        
        llm = None
        for modello in modelli_da_provare:
            try:
                print(f"🔍 Tentativo con modello: {modello}")
                llm = ChatGoogleGenerativeAI(
                    model=modello, 
                    temperature=0.3,
                    google_api_key=GOOGLE_API_KEY,
                    convert_system_message_to_human=True
                )
                # Test del modello
                test_response = llm.invoke("Test")
                print(f"✅ Modello {modello} funziona correttamente.")
                break
            except Exception as e:
                print(f"❌ Modello {modello} non disponibile: {e}")
                continue
        
        if llm is None:
            raise Exception("Nessun modello Gemini disponibile. Verifica la tua chiave API e i permessi.")
        
        print("✅ Modello di linguaggio inizializzato.")
        
        # Creiamo un "retriever" ottimizzato
        print("🔄 Configurazione del retriever...")
        retriever = vector_db.as_retriever(
            search_type="similarity", 
            search_kwargs={
                "k": 5,  # Numero di documenti da recuperar
                "fetch_k": 20  # Numero di documenti da considerare prima del ranking
            }
        )
        print("✅ Retriever configurato.")
        
        # Template di prompt personalizzato per risposte più accurate
        custom_prompt_template = """Sei un assistente esperto in Dottrina Sociale della Chiesa e filosofia. 
Usa SOLO le informazioni fornite nel contesto per rispondere alla domanda.
Se le informazioni non sono sufficienti per una risposta completa, dillo chiaramente.

Contesto dai documenti:
{context}

Domanda: {question}

Rispondi in modo chiaro, strutturato e preciso, citando quando possibile i concetti specifici dai documenti forniti."""

        custom_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Assembliamo la catena completa con prompt personalizzato
        print("🔄 Creazione della catena QA...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt},
            verbose=False  # Imposta True per debug
        )
        print("✅ Sistema di dialogo pronto e ottimizzato.")

    except Exception as e:
        print(f"❌ Errore durante la preparazione del sistema di dialogo: {e}")
        print(f"🔍 Tipo errore: {type(e).__name__}")
        raise e

    # --- FASE 4: FUNZIONE PER PORRE DOMANDE CON SUPPORTO MULTILINGUE ---
    def poni_domanda(domanda, mostra_fonti=True, max_lunghezza_estratto=150, ricerca_multilingue=True):
        """
        Funzione per porre una domanda al sistema DSC-IA con supporto multilingue
        
        Args:
            domanda (str): La domanda da porre
            mostra_fonti (bool): Se mostrare le fonti utilizzate
            max_lunghezza_estratto (int): Lunghezza massima dell'estratto dalle fonti
            ricerca_multilingue (bool): Se effettuare ricerca anche con traduzione
        """
        print(f"\n❓ Domanda: {domanda}")
        print("\n🔄 Elaborazione della risposta in corso...")
        
        # Dizionario di traduzioni per termini chiave
        traduzioni_chiave = {
            # Italiano -> Inglese
            "sussidiarietà": ["subsidiarity", "subsidiary"],
            "dignità": ["dignity", "human dignity"],
            "solidarietà": ["solidarity"],
            "bene comune": ["common good"],
            "destinazione universale": ["universal destination"],
            "proprietà privata": ["private property"],
            "giustizia sociale": ["social justice"],
            "lavoro": ["work", "labor"],
            "famiglia": ["family"],
            "società civile": ["civil society"],
            "stato": ["state", "government"],
            "mercato": ["market"],
            "economia": ["economy", "economic"],
            "sviluppo": ["development"],
            "pace": ["peace"],
            "guerra": ["war"],
            "poveri": ["poor", "poverty"],
            "ricchi": ["rich", "wealth"],
            
            # Inglese -> Italiano
            "subsidiarity": ["sussidiarietà"],
            "dignity": ["dignità"],
            "solidarity": ["solidarietà"],
            "common good": ["bene comune"],
            "universal destination": ["destinazione universale"],
            "private property": ["proprietà privata"],
            "social justice": ["giustizia sociale"],
            "work": ["lavoro"],
            "labor": ["lavoro"],
            "family": ["famiglia"],
            "civil society": ["società civile"],
            "state": ["stato"],
            "government": ["governo", "stato"],
            "market": ["mercato"],
            "economy": ["economia"],
            "development": ["sviluppo"],
            "peace": ["pace"],
            "war": ["guerra"],
            "poor": ["poveri"],
            "poverty": ["povertà"],
            "rich": ["ricchi"],
            "wealth": ["ricchezza"]
        }
        
        def genera_query_alternative(query_originale):
            """Genera versioni alternative della query con traduzioni"""
            queries = [query_originale]
            
            # Converte in minuscolo per il matching
            query_lower = query_originale.lower()
            
            # Per ogni termine chiave, sostituisce con le traduzioni
            for termine, traduzioni in traduzioni_chiave.items():
                if termine.lower() in query_lower:
                    for traduzione in traduzioni:
                        # Sostituisce mantenendo il case
                        query_tradotta = query_originale.replace(termine, traduzione)
                        query_tradotta = query_tradotta.replace(termine.capitalize(), traduzione.capitalize())
                        query_tradotta = query_tradotta.replace(termine.lower(), traduzione.lower())
                        if query_tradotta not in queries:
                            queries.append(query_tradotta)
            
            # Aggiunge versiones tradotte complete per domande comuni
            if any(parola in query_lower for parola in ["cos'è", "cosa è", "spiegami", "definisci"]):
                if "english" not in query_lower:
                    queries.append(query_originale + " (English version)")
            
            if any(parola in query_lower for parola in ["what is", "explain", "define"]):
                if "italiano" not in query_lower and "italian" not in query_lower:
                    queries.append(query_originale + " (versione italiana)")
            
            return queries[:5]  # Limita a 5 query per evitare overhead
        
        try:
            documenti_trovati = []
            
            if ricerca_multilingue:
                print("🌍 Ricerca multilingue attivata...")
                query_alternatives = genera_query_alternative(domanda)
                print(f"🔍 Effettuo ricerca con {len(query_alternatives)} varianti della query")
                
                # Cerca con tutte le varianti
                for i, query_var in enumerate(query_alternatives, 1):
                    try:
                        docs_temp = vector_db.similarity_search(query_var, k=3)  # Meno doc per query
                        if docs_temp:
                            print(f"   ✅ Query {i}: trovati {len(docs_temp)} documenti")
                            documenti_trovati.extend(docs_temp)
                        else:
                            print(f"   ⚪ Query {i}: nessun documento")
                    except Exception as e:
                        print(f"   ❌ Errore query {i}: {e}")
                        continue
                
                # Rimuove duplicati basandosi sul contenuto
                seen_contents = set()
                documenti_unici = []
                for doc in documenti_trovati:
                    content_hash = hash(doc.page_content[:100])  # Usa i primi 100 char come hash
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        documenti_unici.append(doc)
                
                # Prende i migliori 5 documenti
                documenti_finali = documenti_unici[:5]
                print(f"📊 Documenti unici selezionati: {len(documenti_finali)}")
                
                if not documenti_finali:
                    print("⚠️ Nessun documento trovato con ricerca multilingue. Provo ricerca standard...")
                    risultato = qa_chain({"query": domanda})
                else:
                    # Crea un contesto combinato dai documenti trovati
                    contesto_combinato = "\n\n".join([doc.page_content for doc in documenti_finali])
                    
                    # Crea una query migliorata
                    query_migliorata = f"""
                    Basandoti sul seguente contesto che include informazioni in italiano e inglese, 
                    rispondi alla domanda in italiano in modo completo e preciso.
                    Se trovi informazioni rilevanti in inglese, traducile e incorporale nella risposta.
                    
                    Contesto:
                    {contesto_combinato}
                    
                    Domanda: {domanda}
                    """
                    
                    # Usa il modello LLM direttamente con il contesto migliorato
                    risposta = llm.invoke(query_migliorata)
                    
                    # Simula il formato di qa_chain
                    risultato = {
                        'result': risposta.content if hasattr(risposta, 'content') else str(risposta),
                        'source_documents': documenti_finali
                    }
            else:
                # Ricerca standard
                risultato = qa_chain({"query": domanda})
            
            # Stampiamo la risposta
            print("\n🤖 Risposta dall'IA DSC:")
            print("=" * 60)
            print(risultato['result'])
            print("=" * 60)
            
            # Stampiamo le fonti se richiesto
            if mostra_fonti and risultato.get('source_documents'):
                print(f"\n📚 Fonti utilizzate ({len(risultato['source_documents'])} documenti):")
                print("-" * 50)
                
                for i, doc in enumerate(risultato['source_documents'], 1):
                    # Estrai metadati in modo sicuro
                    source = doc.metadata.get('source', 'Fonte sconosciuta')
                    chunk_id = doc.metadata.get('chunk_id', 'N/A')
                    
                    # Rileva la lingua del documento
                    content_sample = doc.page_content[:50].lower()
                    is_english = any(word in content_sample for word in ['the', 'and', 'of', 'to', 'in', 'is', 'that'])
                    is_italian = any(word in content_sample for word in ['la', 'il', 'di', 'che', 'per', 'è', 'della'])
                    
                    lingua = "🇬🇧 EN" if is_english and not is_italian else "🇮🇹 IT" if is_italian else "❓"
                    
                    # Tronca l'estratto se troppo lungo
                    estratto = doc.page_content[:max_lunghezza_estratto]
                    if len(doc.page_content) > max_lunghezza_estratto:
                        estratto += "..."
                    
                    print(f"  {i}. 📄 {source} {lingua}")
                    print(f"     🆔 Chunk: {chunk_id}")
                    print(f"     📝 Estratto: \"{estratto}\"")
                    print()
            
            return risultato
            
        except Exception as e:
            print(f"❌ Errore durante l'elaborazione della domanda: {e}")
            print(f"🔍 Tipo errore: {type(e).__name__}")
            return None

    # --- FASE 5: ESEMPI DI UTILIZZO ---
    print("\n" + "="*80)
    print("🎉 SISTEMA DSC-IA PRONTO ALL'USO!")
    print("="*80)
    
    # Esempio 1: Domanda predefinita
    print("\n📋 Esempio 1 - Domanda sul principio di sussidiarietà:")
    esempio_domanda_1 = "Spiegami il principio di sussidiarietà con un esempio pratico per una piccola impresa."
    poni_domanda(esempio_domanda_1)
    
    print("\n" + "-"*50)
    
    # Esempio 2: Seconda domanda
    print("\n📋 Esempio 2 - Domanda sulla Dottrina Sociale:")
    esempio_domanda_2 = "Quali sono i principi fondamentali della Dottrina Sociale della Chiesa?"
    poni_domanda(esempio_domanda_2)
    
    # --- ISTRUZIONI PER L'USO CONTINUO ---
    print(f"\n" + "="*80)
    print("💡 COME CONTINUARE AD USARE IL SISTEMA:")
    print("="*80)
    print("Per porre nuove domande, usa la funzione:")
    print("   poni_domanda(\"La tua domanda qui\")")
    print("")
    print("🌍 RICERCA MULTILINGUE (predefinita):")
    print("   poni_domanda(\"Cos'è la sussidiarietà?\")  # Trova info in IT e EN")
    print("   poni_domanda(\"What is subsidiarity?\")    # Trova info in IT e EN")
    print("")
    print("⚙️ OPZIONI AVANZATE:")
    print("   poni_domanda(\"domanda\", ricerca_multilingue=False)  # Solo ricerca standard")
    print("   poni_domanda(\"domanda\", mostra_fonti=False)         # Nascondi fonti")
    print("   poni_domanda(\"domanda\", max_lunghezza_estratto=200) # Estratti più lunghi")
    print("")
    print("📚 ESEMPI DI DOMANDE MULTILINGUE:")
    print("   • \"Spiegami il principio di sussidiarietà\"")
    print("   • \"What is the universal destination of goods?\"")
    print("   • \"Come si applica la solidarietà nell'economia?\"")
    print("   • \"Explain the relationship between work and human dignity\"")
    print("="*80)
    
    # --- STATISTICHE FINALI ---
    print(f"\n📊 STATISTICHE DEL SISTEMA:")
    print(f"   • Database caricato: ✅")
    print(f"   • Vettori disponibili: {num_vectors}")
    print(f"   • Modello LLM: {llm.model_name if hasattr(llm, 'model_name') else 'Gemini (versione rilevata automaticamente)'}")
    print(f"   • Modello Embedding: Google Embedding-001")
    print(f"   • Documenti per query: 5")
    print(f"   • Sistema pronto: ✅")

poni_domanda("come faccio a migliorare la sussidiarietà nella mia azienda di 13 dipendenti che produce scarpe, in Veneto")

pip install streamlit langchain google-generativeai faiss-cpu

import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configurazione dell'interfaccia
st.set_page_config(
    page_title="DSC-IA Assistente",
    page_icon="🤖",
    layout="wide"
)

# Titolo dell'applicazione
st.title("🤖 DSC-IA Assistente")
st.markdown("Interfaccia per dialogare con il database di conoscenza sulla Dottrina Sociale della Chiesa")

# Sidebar per le impostazioni
with st.sidebar:
    st.header("⚙️ Configurazione")
    api_key = st.text_input("Inserisci la tua Google API Key:", type="password")
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
        genai.configure(api_key=api_key)
        st.success("API Key configurata!")
    
    st.markdown("---")
    st.header("ℹ️ Informazioni")
    st.markdown("""
    Questo sistema utilizza:
    - Modello Gemini di Google
    - Database vettoriale FAISS
    - Documenti sulla Dottrina Sociale della Chiesa
    """)

# Funzione per inizializzare il sistema
@st.cache_resource
def inizializza_sistema(percorso_db, api_key):
    try:
        # Inizializza il modello di embedding
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Carica il database vettoriale
        vector_db = FAISS.load_local(
            percorso_db, 
            embeddings_model, 
            allow_dangerous_deserialization=True
        )
        
        # Inizializza il modello LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        
        # Configura il retriever
        retriever = vector_db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        # Crea il prompt personalizzato
        custom_prompt_template = """Sei un assistente esperto in Dottrina Sociale della Chiesa e filosofia. 
Usa SOLO le informazioni fornite nel contesto per rispondere alla domanda.

Contesto dai documenti:
{context}

Domanda: {question}

Rispondi in modo chiaro, strutturato e preciso."""
        
        custom_prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Crea la catena QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt}
        )
        
        return qa_chain, vector_db
        
    except Exception as e:
        st.error(f"Errore durante l'inizializzazione: {e}")
        return None, None

# Interfaccia principale
if api_key:
    percorso_db = st.text_input("Percorso del database vettoriale:", "DSC_Vector_DB")
    
    if st.button("Inizializza Sistema"):
        with st.spinner("Caricamento del database e inizializzazione del modello..."):
            qa_chain, vector_db = inizializza_sistema(percorso_db, api_key)
            
        if qa_chain:
            st.session_state.qa_chain = qa_chain
            st.session_state.vector_db = vector_db
            st.success("Sistema pronto all'uso!")
            
    if 'qa_chain' in st.session_state:
        st.markdown("---")
        st.header("💬 Fai una domanda")
        
        # Input della domanda
        domanda = st.text_area("Scrivi la tua domanda:", height=100)
        
        if st.button("Cerca risposta"):
            if domanda:
                with st.spinner("Ricerca della risposta..."):
                    try:
                        risultato = st.session_state.qa_chain({"query": domanda})
                        
                        # Mostra la risposta
                        st.subheader("Risposta:")
                        st.write(risultato['result'])
                        
                        # Mostra le fonti
                        with st.expander("Visualizza fonti utilizzate"):
                            for i, doc in enumerate(risultato['source_documents'], 1):
                                source = doc.metadata.get('source', 'Fonte sconosciuta')
                                estratto = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                
                                st.markdown(f"**Fonte {i}:** {source}")
                                st.caption(f"Estratto: {estratto}")
                                st.markdown("---")
                                
                    except Exception as e:
                        st.error(f"Errore durante la ricerca: {e}")
            else:
                st.warning("Per favore, inserisci una domanda.")
else:
    st.warning("Inserisci una Google API Key nella sidebar per iniziare.")

# Footer
st.markdown("---")
st.caption("DSC-IA Assistente - Sistema di domande e risposte basato sulla Dottrina Sociale della Chiesa")
