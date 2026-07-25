# AI Recruiter Assistant

An AI-powered recruitment automation platform that covers the entire hiring lifecycle — from CV parsing and job understanding to intelligent candidate-job matching, AI interviews (text/voice/video), explainable reports, and a full recruiter dashboard — with bilingual support (English/Arabic).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Development (Docker)](#development-docker)
  - [Development (Manual)](#development-manual)
  - [Production](#production)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [AI Pipeline](#ai-pipeline)
  - [CV Parsing](#cv-parsing)
  - [Matching Engine](#matching-engine)
  - [Interview System](#interview-system)
  - [Explainability](#explainability)
- [Database](#database)
- [Scripts](#scripts)
- [Testing](#testing)
- [License](#license)

---

## 📌 Overview

**AI Recruiter Assistant** automates the recruitment pipeline end-to-end:

1. **CV Parsing** — Extract skills, experience, education from CVs (PDF/DOCX/TXT) using rule-based + LLM-enhanced extraction.
2. **Job Understanding** — Analyze job descriptions to extract required/optional skills, seniority, and experience requirements.
3. **Hybrid Matching** — Combine ESCO skill matching, semantic similarity (embeddings), structured scoring, cross-encoder LLM reranking, and historical feedback into an explainable score.
4. **AI Interviews** — Interactive text/voice (STT/TTS) and video (WebRTC) interviews with automatic answer evaluation.
5. **Explainable Reports** — Detailed candidate reports with score breakdowns, skill gap analysis, and candidate comparison.
6. **Continuous Learning** — Improve matching over time via recruiter feedback.
7. **Recruiter Dashboard** — Comprehensive management UI for candidates, jobs, matches, interviews, and reports.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, SQLAlchemy 2.0 (async), Alembic |
| **Frontend** | React 19, TypeScript, Vite 8, Tailwind CSS 4, Recharts, Lucide Icons |
| **Database** | SQLite (dev), PostgreSQL + pgvector (prod) |
| **AI/ML** | sentence-transformers, Ollama (llama3.2, gemma3:4b, nomic-embed-text), rapidfuzz, stanza |
| **Vector Store** | pgvector / NumPy cosine similarity |
| **Auth** | JWT (python-jose + bcrypt) |
| **Task Queue** | Redis (CV processing queue) |
| **Infrastructure** | Docker, docker-compose, nginx, gunicorn |
| **Voice** | faster-whisper (STT), edge-tts (TTS), WebRTC (video interviews) |
| **OCR** | pytesseract, pdf2image |

---

## 🏗 Architecture

```
┌──────────────────────────┐     ┌──────────────────────────────┐
│     React Frontend       │◄───►│       FastAPI Backend         │
│  (Vite + Tailwind)       │     │   ┌──────────────────────┐   │
│   ┌─────────────────┐    │     │   │   API Layer          │   │
│   │ Recruiter UI     │    │     │   │  (REST + WebSocket)  │   │
│   │ Candidate UI     │    │     │   └──────┬───────────────┘   │
│   │ Public Interview │    │     │   ┌──────┴───────────────┐   │
│   └─────────────────┘    │     │   │   Services Layer      │   │
└──────────────────────────┘     │   │  (CV, Matching, NLP,  │   │
                                 │   │   Interview, RAG...)  │   │
┌──────────────────────────┐     │   └──────┬───────────────┘   │
│     External Services    │     │   ┌──────┴───────────────┐   │
│  ┌────────────────────┐  │     │   │   Data Layer         │   │
│  │   Ollama (LLM)     │◄─┤     │   │  (SQLAlchemy +       │   │
│  │   Embeddings       │  │     │   │   pgvector/SQLite)   │   │
│  └────────────────────┘  │     │   └──────────────────────┘   │
│  ┌────────────────────┐  │     └──────────────────────────────┘
│  │   Redis (Queue)    │◄─┤
│  └────────────────────┘  │
│  ┌────────────────────┐  │
│  │   PostgreSQL       │◄─┤
│  │   + pgvector       │  │
│  └────────────────────┘  │
└──────────────────────────┘
```

---

## ✨ Features

### CV Parsing
- Extract text from PDF (with OCR fallback), DOCX, TXT
- Split CV into sections (Experience, Education, Skills, Projects, Certifications...)
- Dual-path skill extraction:
  - **Rule-based**: match against a curated catalog of 500+ skills with negation/learning detection
  - **LLM-enhanced**: bilingual LLM extracts structured skills with context, then grounds them in CV text
- Normalize skills against the **ESCO** taxonomy (European classification)
- Detect skill level (beginner/intermediate/advanced) and status (acquired/learning)
- Structured experience and education entry extraction

### Job Understanding
- Parse job descriptions → required/optional skills, seniority level, experience years
- Enrich with ESCO taxonomy

### Hybrid Matching Engine
- **ESCO Skill Matching** — exact, synonym, URI, and related-skill matching via the ESCO taxonomy graph
- **Semantic Similarity** — cosine similarity between job and candidate embeddings (sentence-transformers or Ollama)
- **Cross-Encoder Reranking** — Ollama LLM reranks top candidates with bounded score adjustment
- **Structured Scoring** — experience years, seniority fit, education level
- **RAG Enrichment** — knowledge base lookups for skill definitions
- **Feedback Integration** — historical recruiter acceptance/rejection affects future matches
- **Explainable Scores** — transparent breakdown of every scoring component with reasoning trace

### AI Interviews
- Skill-based question generation (templates + LLM-grounded)
- Text, voice (STT/TTS), and video (WebRTC) interviews
- Automatic answer evaluation (quick keyword + deep LLM background analysis)
- Public interview support (no authentication required for candidates)
- Follow-up question generation

### Reports & Explainability
- Detailed candidate reports with per-component score breakdown
- Skill gap analysis (missing vs. matched skills)
- Strengths & weaknesses identification
- Side-by-side candidate comparison

### Recruiter Dashboard
- Manage candidates, jobs, interviews, and matches from a single UI
- Single and bulk CV upload
- Match results with full traceability
- Feedback submission to improve the matching model
- Interview session monitoring

### Voice & Video
- Speech-to-text via faster-whisper
- Text-to-speech via edge-tts
- Video interviews via WebRTC with real-time chat

### Security & Production
- JWT authentication with access/refresh token rotation
- Role-based access control (admin, recruiter, candidate)
- Staff workspace isolation: owner, admin, and recruiter accounts only see their own jobs, candidates, matches, reports, feedback, and interviews
- Rate limiting middleware
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
- Trusted hosts validation
- Audit logging (all sensitive operations tracked)
- Production readiness checks

---

## 🚀 Getting Started

### Development (Docker)

```bash
git clone https://github.com/jadAkeel/AI-recuriter.git
cd AI-recuriter

docker-compose up --build
```

- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **Redis**: localhost:6379

### Development (Manual)

#### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

### Production

```bash
docker-compose -f docker-compose.prod.yml up --build
```

Full production stack includes: nginx → API (x2 replicas) + CV Worker + PostgreSQL + pgvector + Redis + Ollama (with GPU support).

Ollama models are pulled automatically on startup: `llama3.2`, `gemma3:4b`, `nomic-embed-text`.

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./app.db` | Database connection string |
| `EMBEDDING_PROVIDER` | `hash` | `hash`, `sentence-transformers`, or `ollama` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `LLM_PROVIDER` | `ollama` | LLM provider for parsing & interviews |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Default LLM model |
| `OLLAMA_INTERVIEW_MODEL` | `gemma3:4b` | Model for interviews |
| `OLLAMA_PARSING_MODEL` | `llama3.2` | Model for CV parsing |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings |
| `JWT_SECRET_KEY` | `dev-only-...` | JWT signing secret (min 32 chars) |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `CORS_ORIGINS_STR` | `http://localhost:5173,...` | Allowed CORS origins |
| `ESCO_API_ENABLED` | `false` | Enable ESCO taxonomy enrichment |
| `APP_BASE_URL` | `http://localhost:5173` | App base URL (emails, links) |
| `MAX_UPLOAD_BYTES` | `15728640` | Max upload size (15MB) |
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting middleware |
| `RUN_CV_WORKER_IN_API` | `true` | Run CV queue worker in API process |
| `KEEP_ALIVE_ENABLED` | `false` | Enable periodic public health pings for Render demos |
| `KEEP_ALIVE_INTERVAL_SECONDS` | `270` | Keep-alive interval in seconds; values below 60 are clamped |
| `KEEP_ALIVE_URL` | auto from `RENDER_EXTERNAL_HOSTNAME` | Optional explicit URL to ping for keep-alive |
| `FIRST_USER_OWNER_ENABLED` | `false` | Promote the first registered account to owner when no users exist |
| `INITIAL_OWNER_EMAIL` / `INITIAL_OWNER_PASSWORD` | not set | Optional fixed owner account created/promoted on startup when no owner exists |
| `SMTP_*` | — | Email configuration for interview invitations |

---

## 🌐 API Endpoints

All endpoints are prefixed with `/api/v1`. Full interactive docs at http://localhost:8000/docs.

### Health & Readiness
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Basic health check |
| GET | `/ready` | Full readiness check (DB, Redis, Ollama) |
| GET | `/health/embeddings` | Embedding provider info |

### Auth
| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login (returns JWT access + refresh tokens) |
| POST | `/auth/refresh` | Refresh access token |
| GET | `/auth/me` | Current user profile |
| GET | `/auth/users` | List all users (admin) |
| PATCH | `/auth/users/{id}/role` | Change user role |

### CV Parsing
| Method | Path | Description |
|--------|------|-------------|
| POST | `/cv/parse` | Upload and parse a CV file |

### Candidates
| Method | Path | Description |
|--------|------|-------------|
| GET | `/skills/categories` | Skill catalog grouped by category |
| GET | `/skills` | All known skill names |
| POST | `/candidates` | Upload CV and create/update candidate (sync) |
| POST | `/candidates/async` | Queue CV for background processing |
| GET | `/candidates/async/{task_id}` | Poll async CV result |
| GET | `/candidates` | List candidates with search, filter, sort, pagination |
| GET | `/candidates/me` | Candidate's own profile |
| GET | `/candidates/{id}` | Get candidate details |
| GET | `/candidates/{id}/cv` | Download or preview CV |
| DELETE | `/candidates/{id}` | Delete candidate + all dependencies |
| DELETE | `/candidates` | Delete ALL candidates (admin) |
| POST | `/candidates/stream` | Stream multi-CV upload with NDJSON results |

### Jobs
| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs/parse` | Parse job description text (no save) |
| GET | `/jobs` | List all saved jobs |
| POST | `/jobs` | Create job (parse → enrich → save → embed) |
| PATCH | `/jobs/{id}` | Update job + re-embed |
| DELETE | `/jobs/{id}` | Delete job + dependencies |

### Matching
| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs/{id}/match` | Run full matching pipeline against job |
| GET | `/jobs/{id}/matches` | Get saved matches for a job |
| POST | `/matching/feedback` | Submit recruiter feedback on a match |
| GET | `/matching/feedback/stats` | Feedback statistics |

### Interviews
| Method | Path | Description |
|--------|------|-------------|
| POST | `/interviews/start` | Create interview session |
| POST | `/interviews/invite` | Create session + send email invitation |
| GET | `/interviews/public/{session_id}` | Get public interview state |
| POST | `/interviews/public/{session_id}/answer` | Submit text answer (public) |
| POST | `/interviews/public/{session_id}/voice-answer` | Submit voice answer (public) |
| POST | `/interviews/public/{session_id}/evaluate` | Get public evaluation |
| POST | `/interviews/answer` | Submit answer (authenticated) |
| POST | `/interviews/chat-answer` | Chat-style answer with next question |
| POST | `/interviews/followup` | Generate follow-up question |
| GET | `/interviews/{session_id}` | Interview status |
| GET | `/interviews/dashboard-results` | Recruiter dashboard data |
| DELETE | `/interviews/{session_id}` | Delete interview |
| POST | `/interviews/evaluate` | Evaluate interview session |

### Reports
| Method | Path | Description |
|--------|------|-------------|
| POST | `/reports/candidate` | Generate candidate report |
| POST | `/reports/compare` | Compare multiple candidates |
| DELETE | `/reports/candidate` | Delete a report |

### RAG (Knowledge Base)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/rag/ingest` | Add knowledge document |
| POST | `/rag/seed` | Seed built-in knowledge base |
| POST | `/rag/query` | Query knowledge base semantically |

### ESCO
| Method | Path | Description |
|--------|------|-------------|
| GET | `/esco/skills/count` | ESCO skill count |
| POST | `/esco/extract` | Extract ESCO skills from text |
| POST | `/esco/refresh` | Refresh ESCO cache |

### Voice
| Method | Path | Description |
|--------|------|-------------|
| POST | `/voice/start/{session_id}` | Start voice session |
| POST | `/voice/process` | Process base64 audio |
| POST | `/voice/process/upload` | Process uploaded audio file |
| GET | `/voice/status/{session_id}` | Voice session status |

### WebSocket
| Path | Description |
|------|-------------|
| `/ws/cv-notifications` | Real-time CV processing status updates |
| `/ws/interview/{session_id}` | Live interview chat with voice & WebRTC support |

---

## 📁 Project Structure

```
AI-recuriter/
├── backend/
│   ├── app/
│   │   ├── main.py                 # App bootstrap, lifespan, middleware
│   │   ├── worker.py               # Standalone CV queue worker
│   │   ├── api/                    # Route handlers
│   │   │   ├── router.py           # Master router
│   │   │   ├── auth.py             # Registration, login, token refresh
│   │   │   ├── cv.py               # CV parsing endpoint
│   │   │   ├── candidates.py       # Candidate CRUD + streaming upload
│   │   │   ├── jobs.py             # Job CRUD
│   │   │   ├── matching.py         # Candidate-job matching
│   │   │   ├── interviews.py       # Interview sessions & evaluation
│   │   │   ├── reports.py          # Reports & comparison
│   │   │   ├── feedback.py         # Matching feedback
│   │   │   ├── rag.py              # RAG knowledge base
│   │   │   ├── esco.py             # ESCO skill extraction
│   │   │   ├── voice.py            # Voice STT/TTS
│   │   │   ├── health.py           # Health & readiness
│   │   │   └── ws.py               # WebSocket endpoints
│   │   ├── core/                   # Infrastructure
│   │   │   ├── config.py           # Pydantic settings (env-based)
│   │   │   ├── db.py               # Async DB engine & session factory
│   │   │   ├── deps.py             # FastAPI dependencies (auth, roles)
│   │   │   ├── logging.py          # Async queue-based logging
│   │   │   ├── redis.py            # Redis client with caching
│   │   │   └── security.py         # Security headers & rate limiting
│   │   ├── models/                 # SQLAlchemy models
│   │   │   ├── base.py             # DeclarativeBase
│   │   │   ├── candidate.py        # Candidate (CV data)
│   │   │   ├── embedding.py        # Embedding (pgvector + JSON)
│   │   │   ├── interview.py        # InterviewSession
│   │   │   ├── job.py              # Job
│   │   │   ├── match_result.py     # MatchResult
│   │   │   ├── report.py           # Report
│   │   │   ├── report_version.py   # Report versioning
│   │   │   ├── audit_log.py        # Audit logging
│   │   │   ├── skill_evidence.py   # Skill evidence
│   │   │   ├── skill_feedback.py   # Recruiter feedback
│   │   │   ├── knowledge.py        # RAG knowledge documents
│   │   │   └── user.py             # Users
│   │   ├── schemas/                # Pydantic schemas
│   │   │   ├── candidate.py
│   │   │   ├── job.py
│   │   │   ├── match.py
│   │   │   ├── interview.py
│   │   │   ├── report.py
│   │   │   ├── auth.py
│   │   │   ├── esco.py
│   │   │   ├── rag.py
│   │   │   └── health.py
│   │   └── services/              # Business logic
│   │       ├── cv_parser.py       # Rule-based CV parsing
│   │       ├── enhanced_cv_parser.py # LLM-enhanced CV parsing
│   │       ├── embedding.py       # Embedding providers
│   │       ├── vector_store.py    # Vector DB (pgvector / in-memory)
│   │       ├── matching.py        # Legacy matching
│   │       ├── hybrid_matcher.py  # Core hybrid matching engine
│   │       ├── job_parser.py      # Job description parsing
│   │       ├── interview.py       # Template + LLM interview questions
│   │       ├── enhanced_interview.py # LLM answer evaluation
│   │       ├── interview_analysis.py # Background async analysis
│   │       ├── explainability.py  # Reports & comparison
│   │       ├── skill_catalog.py   # 500+ skill taxonomy
│   │       ├── esco_service.py    # ESCO taxonomy integration
│   │       ├── ollama_cross_encoder.py # Cross-encoder reranker
│   │       ├── rag.py             # RAG knowledge base
│   │       ├── task_queue.py      # Redis CV processing queue
│   │       ├── candidate_text.py  # Embedding text builder
│   │       ├── skill_evidence.py  # Skill evidence storage
│   │       ├── continuous_learning.py # Feedback → dynamic weights
│   │       ├── audit.py           # Audit logging
│   │       ├── ai_metadata.py     # AI provider metadata
│   │       ├── auth.py            # Authentication logic
│   │       ├── bilingual_llm.py   # English/Arabic LLM service
│   │       ├── voice_service.py   # STT/TTS
│   │       ├── stanza_nlp.py      # Stanza NLP pipeline
│   │       ├── production_backfill.py # Production data backfill
│   │       └── readiness.py       # Production readiness checks
│   ├── alembic/                   # Database migrations
│   │   └── versions/
│   │       ├── 0001_initial_schema.py
│   │       ├── 0002_embedding_metadata.py
│   │       ├── 0003_skill_feedback.py
│   │       └── 0004_audit_evidence_versioning.py
│   ├── tests/                     # Test suite
│   │   ├── test_cv_parser.py
│   │   ├── test_health.py
│   │   ├── test_jobs.py
│   │   ├── test_matching.py
│   │   ├── test_benchmark.py
│   │   ├── test_regression.py
│   │   ├── test_fix_*.py
│   │   ├── test_production_readiness_foundation.py
│   │   ├── test_readiness_and_production_guards.py
│   │   └── fixtures/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── PRODUCTION_RUNBOOK.md
├── frontend/
│   ├── src/
│   │   ├── main.tsx               # Entry point
│   │   ├── App.tsx                 # Router configuration
│   │   ├── api/
│   │   │   └── client.ts           # Axios with JWT interceptor
│   │   ├── components/
│   │   │   ├── Layout.tsx          # Sidebar + navbar layout
│   │   │   ├── Sidebar.tsx         # Navigation sidebar
│   │   │   ├── Navbar.tsx          # Top navbar
│   │   │   ├── ProtectedRoute.tsx  # Auth guard
│   │   │   └── VoiceRecorder.tsx   # Voice recording UI
│   │   ├── context/
│   │   │   ├── AuthContext.tsx      # Auth state
│   │   │   └── auth.ts             # Auth hook
│   │   ├── hooks/
│   │   │   ├── useWebRTC.ts        # WebRTC video hook
│   │   │   └── useVoiceRecorder.ts # MediaRecorder hook
│   │   ├── pages/
│   │   │   ├── Login.tsx
│   │   │   ├── Register.tsx
│   │   │   ├── PublicInterview.tsx
│   │   │   ├── LiveInterview.tsx
│   │   │   ├── VideoInterview.tsx
│   │   │   ├── candidate/
│   │   │   │   ├── Dashboard.tsx
│   │   │   │   ├── UploadCV.tsx
│   │   │   │   ├── Interview.tsx
│   │   │   │   └── Results.tsx
│   │   │   └── recruiter/
│   │   │       ├── Dashboard.tsx
│   │   │       ├── Jobs.tsx
│   │   │       ├── Candidates.tsx
│   │   │       ├── BulkUpload.tsx
│   │   │       ├── Matching.tsx
│   │   │       ├── MatchResults.tsx
│   │   │       ├── Interviews.tsx
│   │   │       └── Reports.tsx
│   │   ├── types/
│   │   │   └── api.ts
│   │   └── utils/
│   │       ├── network.ts
│   │       └── errors.ts
│   ├── package.json
│   └── vite.config.ts
├── scripts/                       # PowerShell & Python utility scripts
├── nginx/                         # Production nginx config
├── nlpPrompt/                     # Project development prompts
├── docker-compose.yml             # Development stack
├── docker-compose.prod.yml        # Production stack
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🤖 AI Pipeline

### CV Parsing Pipeline

```
Upload CV (PDF/DOCX/TXT)
        │
        ▼
Text Extraction
├── PDF → pdfplumber + OCR (pytesseract fallback)
├── DOCX → python-docx
└── TXT → direct decode
        │
        ▼
Section Splitting (regex-based headers)
├── Experience, Education, Skills, Projects, Certifications...
        │
        ▼
Skill Extraction (Dual Path)
├── Rule-based: normalize text → catalog match (500+ skills)
│   └── negation/learning detection, years/level estimation
└── LLM-enhanced: bilingual LLM → structured skills
    └── ground each skill in CV text → merge with rule-based
        │
        ▼
ESCO Normalization → URI mapping → related skills
        │
        ▼
Experience & Education Parsing
├── Dates, titles, companies, institutions, degrees
        │
        ▼
Profile Assembly → CandidateProfile
        │
        ▼
Embedding Generation (hash / sentence-transformers / Ollama)
        │
        ▼
Skill Evidence Storage
```

### Matching Pipeline

```
Job Description
        │
        ▼
Job Parser → required/optional skills, seniority, experience years
        │
        ▼
Pre-compute Job Embedding
Pre-compute Candidate Embeddings (batch, with vector store cache)
        │
        ▼
For each candidate:
├── ESCO Skill Matching
│   ├── exact match (normalized)
│   ├── synonym match (SYNONYM_MAP)
│   ├── ESCO URI match (same taxonomy node)
│   ├── ESCO related skill match (broader/narrower/related)
│   ├── historical feedback match
│   └── text evidence fallback
├── Negation Detection → exclude denied skills
├── Evidence-Adjusted Confidence
├── Semantic Score (cosine similarity)
├── Junior Project Semantic Bonus
├── Seniority Score (years fit vs. job band)
├── Experience Score (normalized years)
        │
        ▼
Final Score = 0.55(required) + 0.20(optional) + 0.15(semantic)
              + 0.05(experience) + 0.05(seniority)
        │
        ▼
Score Capped by Required-Skill Coverage (0.3 + 0.7 * required_score)
        │
        ▼
Cross-Encoder Reranking (Ollama LLM, top-K pairs, ±0.05 adjustment)
        │
        ▼
Save MatchResult with full reasoning trace
├── scoring model, weights, contributions, penalties, trace
```

### Interview Pipeline

```
Start Interview
        │
        ▼
Generate Questions
├── Template-based (pre-defined questions per skill)
└── LLM-grounded (generated from CV + job context)
        │
        ▼
Collect Answers
├── Text (form input)
├── Voice (STT via faster-whisper)
└── Video (WebRTC with real-time chat)
        │
        ▼
Evaluate
├── Quick score (keyword matching + similarity)
└── Background LLM analysis (deep evaluation, async)
        │
        ▼
Generate Follow-up Questions (if needed)
        │
        ▼
Save to Interview Session
```

### Explainability Pipeline

```
Match Results
        │
        ▼
Report Generator
├── Score breakdown per category
├── Skill gap analysis (matched vs. missing)
├── Strengths & weaknesses
└── Candidate comparison (side-by-side)
        │
        ▼
Saved as Report with versioning
```

---

## 💾 Database

### Models

| Model | Table | Purpose |
|-------|-------|---------|
| `User` | `users` | Authentication & role-based access |
| `Candidate` | `candidates` | Parsed CV data (skills, experience, education) |
| `Embedding` | `embeddings` | Vector embeddings (pgvector or JSON) |
| `Job` | `jobs` | Job descriptions with parsed requirements |
| `MatchResult` | `match_results` | Scored candidate-job matches with reasoning trace |
| `InterviewSession` | `interview_sessions` | Interview sessions, questions, answers |
| `Report` | `reports` | Candidate reports with score breakdowns |
| `ReportVersion` | `report_versions` | Report version history |
| `AuditLog` | `audit_logs` | Security audit trail for sensitive operations |
| `SkillEvidence` | `skill_evidence` | Evidence text for each extracted skill |
| `SkillFeedback` | `skill_feedback` | Recruiter feedback for continuous learning |
| `KnowledgeDocument` | `knowledge_documents` | RAG knowledge base documents |

### Migrations (Alembic)

| Version | Description |
|---------|-------------|
| `0001_initial_schema.py` | Initial schema (users, candidates, jobs, matches, etc.) |
| `0002_embedding_metadata.py` | Embedding metadata fields |
| `0003_skill_feedback.py` | Feedback loop tables |
| `0004_audit_evidence_versioning.py` | Audit logs, skill evidence, report versioning |

---

## 📜 Scripts

| Script | Purpose |
|--------|---------|
| `scripts/start-public-interview.ps1` | Start frontend + ngrok tunnel + backend for public interviews |
| `scripts/start-backend-tunnel.ps1` | Start ngrok tunnel to backend |
| `scripts/start-ngrok-tunnel.ps1` | Start ngrok tunnel |
| `scripts/backup-production.ps1` | Production database backup |
| `scripts/restore-production.ps1` | Production database restore |
| `scripts/download_esco_data.py` | Download ESCO taxonomy data |
| `scripts/evaluate_cross_encoder_impact.py` | Evaluate cross-encoder scoring impact |
| `scripts/generate_skill_catalog_extender.py` | Generate extended skill catalog entries |
| `backend/scripts/backfill_production_readiness.py` | Backfill production data |
| `backend/check_candidates.py` | Diagnostic: inspect candidate data |
| `backend/find_triple.py` | Diagnostic: find skill triples |
| `backend/reset_pass.py` | Reset user password |
| `check_db.py` | Inspect database contents |
| `copy_candidates.py` | Copy candidates between environments |
| `generate_report_final.py` | Generate final audit report |
| `reset_password.py` | Password reset utility |

---

## 🧪 Testing

```bash
# From the backend directory
cd backend

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_cv_parser.py -v

# Run with coverage report
pytest --cov=app --cov-report=term-missing

# Run production readiness tests
pytest tests/test_production_readiness_foundation.py -v
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

MIT

### Staff Workspace Isolation

Each job and candidate created by an authenticated owner, admin, or recruiter is assigned to that user's private workspace. Matching, reports, feedback, interviews, CV downloads, and delete/update operations enforce the same ownership boundary. A staff user cannot read, edit, delete, or match another staff user's data.

The candidate-facing job board remains shared so candidates can discover available positions. Candidate self-service data is still restricted to the signed-in candidate. User accounts and role management remain global for owner/admin administration.

Migration `0005_user_data_isolation` assigns existing legacy jobs, candidates, and feedback records to the first staff account during deployment. New API-created records always receive an owner. Render runs Alembic migrations automatically on startup.
