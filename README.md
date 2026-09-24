# Persona Adaptive Customer Support Agent

## Project Overview

Persona Adaptive Customer Support Agent is an AI-powered support system that delivers personalized customer support responses based on customer persona.

The system identifies the user persona from the incoming query, retrieves relevant knowledge from a support knowledge base using Retrieval-Augmented Generation (RAG), generates persona-adaptive responses, and escalates complex or sensitive issues to human support when necessary.

Supported Personas:

* Technical Expert
* Frustrated User
* Business Executive

Core Capabilities:

* Persona Detection
* Knowledge Retrieval (RAG)
* Adaptive Response Generation
* Escalation Detection
* Human Handoff Summary
* Interactive Streamlit UI

---

# Tech Stack

## Backend

* Python 3.12

## LLM

* Google Gemini 2.5 Flash

## Agent Framework / Libraries

* LangChain Text Splitters 0.3+
* Custom RAG Pipeline

## Embedding Model

* Sentence Transformers (`all-MiniLM-L6-v2`)

## Vector Database

* ChromaDB 1.0+

## UI

* Streamlit 1.47+

## Libraries

* google-genai 2.9.0
* sentence-transformers 5.0+
* chromadb 1.0+
* langchain 0.3+
* langchain-text-splitters 0.3+
* pypdf 5.8+
* python-dotenv 1.1+

---

# Architecture Diagram

```plaintext
User Query
   ↓
Persona Detection
   ↓
Knowledge Retrieval (RAG)
   ↓
Response Generation
   ↓
Escalation Check
   ↓
Human Handoff / Final Response
```

Detailed Flow:

1. User submits support query.
2. Gemini classifies customer persona.
3. RAG pipeline retrieves relevant knowledge base chunks.
4. Gemini generates persona-aware response.
5. Escalation logic checks confidence and issue sensitivity.
6. If needed, issue is escalated to human support.

---

# Persona Detection Strategy

## Classification Method

Persona classification is performed using Google Gemini 2.5 Flash.

Each user query is analyzed based on:

* Tone
* Emotional state
* Technical vocabulary
* Business context
* Urgency level

The classifier returns:

* Persona
* Confidence Score
* Reasoning

Example JSON output:

```json
{
  "persona": "Technical Expert",
  "confidence": 0.91,
  "reasoning": "Query contains API and authentication related terminology."
}
```

---

## Prompt Design

Gemini is instructed to classify queries into exactly one of three personas:

### Technical Expert

Characteristics:

* Technical vocabulary
* Logs, APIs, configurations
* Debugging mindset

### Frustrated User

Characteristics:

* Emotional language
* Urgent issues
* Negative sentiment

### Business Executive

Characteristics:

* Concise communication
* Business impact focus
* ROI / downtime concerns

---

## Rules Used

* Only one persona can be assigned.
* Output must be valid JSON.
* Confidence score ranges from 0 to 1.
* Classification reasoning is mandatory.

---

# RAG Pipeline Design

The project uses Retrieval-Augmented Generation (RAG) for knowledge retrieval.

---

## Document Loading

Supported formats:

* TXT
* Markdown (.md)
* PDF

Knowledge base includes:

* Account Security
* Billing Policies
* Refund Policies
* API Authentication
* Login Troubleshooting
* Payment Issues

---

## Chunking Strategy

Documents are split using RecursiveCharacterTextSplitter.

Configuration:

* Chunk Size: 500
* Chunk Overlap: 50

Purpose:

* Preserve context continuity
* Improve retrieval quality

---

## Embedding Model

Embedding Model:

* Sentence Transformers (`all-MiniLM-L6-v2`)

Purpose:

* Convert text chunks into vector embeddings
* Capture semantic meaning

---

## Vector Database Choice

Database:

* ChromaDB

Reasons:

* Lightweight
* Easy integration
* Persistent local storage
* Fast similarity search

---

## Retrieval Strategy

Steps:

1. Convert user query into embedding.
2. Perform vector similarity search in ChromaDB.
3. Retrieve top-k relevant chunks.

Configuration:

* Top K = 3

Returned Data:

* Chunk text
* Source document
* Similarity score

---

# Escalation Logic

The system escalates issues to human support when needed.

---

## Escalation Triggers

### 1. No Relevant Documents Found

If RAG fails to retrieve relevant context.

### 2. Low Confidence Persona Detection

If confidence score < 0.65

### 3. Sensitive Topics

Sensitive issues include:

* Refund
* Billing
* Payment
* Legal issues
* Subscription cancellation

### 4. High User Frustration

Escalates if user shows strong negative sentiment.

Example keywords:

* angry
* frustrated
* terrible
* useless
* worst

---

# Setup Instructions

## Step 1: Clone Repository

```bash
git clone <your_repository_url>
cd persona-support-agent
```

---

## Step 2: Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Add Environment Variables

Create `.env` file.

---

## Step 5: Run Application

```bash
streamlit run app.py
```

---

# Environment Variables

Required variables:

```env
GEMINI_API_KEY=your_gemini_api_key
```

API Key Source:
Google AI Studio

---

# Example Queries

## Technical Expert

1. API authentication is failing with 401 unauthorized error.
2. I need details about OAuth token validation.

## Frustrated User

3. I have tried everything and login still fails.
4. This service is terrible. Nothing is working.

## Business Executive

5. Payment gateway downtime is affecting customer transactions.
6. What business impact will service outage cause?

---

# Known Limitations

Current limitations:

* Single-turn conversation only
* Limited knowledge base size
* No persistent user memory
* Basic escalation rules
* No deployment monitoring

---

# Future Improvements

Planned enhancements:

* Multi-turn support conversations
* Conversation memory
* Advanced analytics dashboard
* Better escalation intelligence
* Cloud deployment
* Real-time monitoring
"# AI-Persona-Adaptive-Customer-Support-Agent-main" 
