# Deployment Guide / دليل النشر والتشغيل (Render)

This project is configured to be deployed on **Render** using a single-click blueprint setup.

يحتوي هذا المشروع على تهيئة جاهزة للنشر المباشر على منصة **Render** باستخدام ملف الـ Blueprint (`render.yaml`).

---

## 🛠️ Deployment Steps / خطوات النشر

### 1. Update GitHub / تحديث مستودع جيت هاب
The changes have been pushed to your GitHub repository:
`https://github.com/jadAkeel/HR_recruiter_assistant.git`

تم تحديث ورفع كافة التعديلات إلى مستودع GitHub الخاص بك.

### 2. Deploy on Render / النشر على Render
1. Go to the [Render Dashboard](https://dashboard.render.com/).
2. Click **New** (top right) and select **Blueprint**.
3. Connect your GitHub account and select the **HR_recruiter_assistant** repository.
4. Render will automatically detect the `render.yaml` blueprint. Click **Apply**.
5. Render will now provision:
   - A PostgreSQL Database (with `pgvector` support).
   - A Redis Instance (for task queuing).
   - FastAPI Backend Web Service.
   - Vite React Frontend Static Site.

1. توجه إلى [لوحة تحكم Render](https://dashboard.render.com/).
2. اضغط على زر **New** في أعلى اليمين واختر **Blueprint**.
3. قم بربط حساب GitHub الخاص بك واختيار مستودع **HR_recruiter_assistant**.
4. سيقرأ Render ملف `render.yaml` تلقائياً. اضغط على **Apply**.
5. سيقوم Render بإنشاء وتشغيل المكونات التالية تلقائياً:
   - قاعدة بيانات PostgreSQL (مع دعم `pgvector`).
   - خادم Redis (لإدارة طوابير المهام الخلفية).
   - الخدمة الخلفية FastAPI Backend.
   - واجهة المستخدم الثابتة Vite React Frontend.

The backend Blueprint and external keep-alive workflow use the lightweight
`/api/v1/health` endpoint so the web service can stay awake independently of
optional AI-provider availability. Check the deeper `/api/v1/ready` endpoint
separately when validating PostgreSQL, Redis, embeddings, and the configured
LLM provider. In production, CORS accepts only the frontend URL configured
through `CORS_ORIGINS_STR`.

يستخدم الـBackend وGitHub workflow المسار السريع `/api/v1/health` كي تبقى
الخدمة مستيقظة بشكل مستقل عن توفّر مزوّد الـAI. افحص المسار الأعمق
`/api/v1/ready` بشكل منفصل عند التحقق من PostgreSQL وRedis والـembeddings
ومزوّد الـLLM المضبوط. وفي الإنتاج، يقبل CORS فقط رابط الواجهة المضبوط في
`CORS_ORIGINS_STR`.

---

## ⚙️ Post-Deployment Configurations / الإعدادات اللازمة بعد النشر

Once the services are deployed, make sure to configure the following environment variables in the Render dashboard:

بعد اكتمال النشر، يرجى ضبط المتغيرات البيئية التالية في لوحة تحكم Render:

### Backend Environment Variables (`ai-recruiter-backend`):
1. **`OPENAI_API_KEY`**: 
   - Add your OpenAI API key in the environment settings of the backend service (OpenAI is highly recommended for production as Ollama requires massive resources not available on Render Free tier).
   - قم بإضافة مفتاح OpenAI API الخاص بك في إعدادات البيئة للخدمة الخلفية (يُنصح باستخدام OpenAI للإنتاج لأن تشغيل Ollama يتطلب موارد ضخمة غير متوفرة في الخدمة المجانية لـ Render).

2. **`CORS_ORIGINS_STR`**:
   - Once your frontend service is deployed, copy its URL (e.g., `https://ai-recruiter-frontend.onrender.com`) and paste it as the value for `CORS_ORIGINS_STR` in the backend service.
   - بمجرد نشر واجهة المستخدم، انسخ رابطها (مثال: `https://ai-recruiter-frontend.onrender.com`) وضعه كقيمة لـ `CORS_ORIGINS_STR` في إعدادات الخدمة الخلفية.

3. **`TRUSTED_HOSTS_STR`**:
   - Set this to the host name of your backend service (e.g., `ai-recruiter-backend.onrender.com`).
   - اضبط هذا المتغير ليكون اسم النطاق الخاص بالخدمة الخلفية (مثال: `ai-recruiter-backend.onrender.com`).

---

4. **Keep-alive for Render Free**:
   - `KEEP_ALIVE_ENABLED=true` and `KEEP_ALIVE_INTERVAL_SECONDS=270` are configured in `render.yaml`.
   - The backend pings its public `/api/v1/health` endpoint every 4.5 minutes using `RENDER_EXTERNAL_HOSTNAME`.
   - `.github/workflows/render-keep-alive.yml` also pings the endpoint externally every 5 minutes, so a stopped instance can be woken from outside the service.
   - Set `KEEP_ALIVE_URL` only if you want to override the auto-detected Render URL.
   - GitHub Actions schedules can occasionally start late. For guaranteed always-on hosting, use a paid Render web-service instance.

5. **Owner/admin bootstrap**:
   - `FIRST_USER_OWNER_ENABLED=true` is configured in `render.yaml`, so the first registered account on an empty database becomes `owner`.
   - For an existing database with users but no owner, set both `INITIAL_OWNER_EMAIL` and `INITIAL_OWNER_PASSWORD` in the backend Render environment. On startup, that account will be created or promoted to `owner`.

---

## ✅ Render fix checklist (demo hardening)

What the live services showed before this fix, and why:

- `/api/v1/ready` reported `llm_provider=ollama` (unreachable) and hash embeddings, and the API
  echoed arbitrary `*.onrender.com` origins with credentials. So the live backend runs **without**
  `ENVIRONMENT=production` and without the blueprint's env vars.
- The live backend is named `ai-recuriter-yoy5`, not the blueprint's `ai-recruiter-backend`. It was
  most likely created by hand, so `render.yaml` never applied to it. Earlier `render.yaml` versions also
  used `fromService ... property: url`, which Render does not support (valid properties: `host`,
  `port`, `hostport`, `slug`, `connectionString`, `connectionPoolString`), plus a `plan` on the static site. That
  can make a Blueprint sync fail. Both are fixed in `render.yaml`.

Decision for the demo: **`LLM_PROVIDER=openai` + `EMBEDDING_PROVIDER=sentence-transformers`,
`ENVIRONMENT=production`.** OpenAI is implemented for all async chat/generation paths
(`backend/app/services/bilingual_llm.py`). Only the optional Ollama cross-encoder rerank
(`cross_encoder_top_k`, default 0) needs Ollama, so leave it at 0 on Render.

### 1. Backend service `ai-recuriter-yoy5` → Environment

Set exactly these (Dashboard → service → Environment → Edit), then **Save, rebuild, and deploy**:

| Key | Value |
|-----|-------|
| `ENVIRONMENT` | `production` |
| `LLM_PROVIDER` | `openai` |
| `OPENAI_API_KEY` | your key (the account needs billing enabled) |
| `EMBEDDING_PROVIDER` | `sentence-transformers` |
| `AUTO_DETECT_LANG` | `false` (Free has 512 MB RAM; a second multilingual model would OOM) |
| `HF_HOME` | `/opt/render/project/src/backend/.hf_cache` |
| `CORS_ORIGINS_STR` | `https://ai-recruiter-frontend.onrender.com` (exact origin, no trailing slash) |
| `TRUSTED_HOSTS_STR` | `localhost,127.0.0.1` (the Render hostname is added automatically from `RENDER_EXTERNAL_HOSTNAME`) |
| `JWT_SECRET_KEY` | a random value of 32+ characters (production refuses the dev default) |
| `DATABASE_URL` | the Postgres **internal** connection string (production refuses SQLite) |
| `REDIS_URL` | the Render Redis/Key Value internal URL (production refuses to start without Redis) |
| `RUN_CV_WORKER_IN_API` | `true` |
| `KEEP_ALIVE_ENABLED` | `true` |

Delete these if present: `OLLAMA_*`, `CORS_ALLOW_TUNNELS`, `ALLOW_UNOWNED_RESOURCES`, and any
`EMBEDDING_PROVIDER=hash`.

Also set **Build Command** to match `render.yaml` (it pre-fetches the embedding weights):

```
pip install -r requirements-prod.txt && (python scripts/download_models.py --skip-whisper --no-load-check || echo "WARNING - model pre-download failed; models will download on first use")
```

### 2. Verify the backend

```bash
curl -s https://ai-recuriter-yoy5.onrender.com/api/v1/ready
```

Expect HTTP 200 and `"status": "ok"`, with `llm_provider.provider = "openai"` ("openai key configured")
and `embedding_provider.active_provider = "sentence-transformers"`. A `degraded` answer (HTTP 503) names
the failing check. `fallback_models` lists any model served by hash vectors.

CORS must no longer reflect foreign origins. This should print **no** `access-control-allow-origin` header:

```bash
curl -si -X OPTIONS https://ai-recuriter-yoy5.onrender.com/api/v1/candidates -H "Origin: https://x.onrender.com" -H "Access-Control-Request-Method: GET"
```

If the instance restarts with "out of memory" in the logs, sentence-transformers does not fit in 512 MB.
Upgrade the instance (Starter or larger), or fall back to `EMBEDDING_PROVIDER=hash`. Hash is only
allowed with `ENVIRONMENT` unset/`development`, and semantic scores are then off (readiness will say so).

### 3. Frontend `ai-recruiter-frontend` ("Service Suspended")

1. Dashboard → `ai-recruiter-frontend`. If the page shows **Suspended**, click **Resume Service**.
   If it was suspended by Render (not by you), the banner gives the reason: usually billing, free-tier
   limits, or a policy flag. Fix that first (Billing page / support).
2. Environment: `VITE_API_BASE_URL=https://ai-recuriter-yoy5.onrender.com/api/v1`. It must be a full
   `https://` URL, because the build bakes it in.
3. **Manual Deploy → Clear build cache & deploy.** Vite reads env vars only at build time.
4. Open https://ai-recruiter-frontend.onrender.com, register, and confirm the network tab calls
   `ai-recuriter-yoy5.onrender.com` without CORS errors.
5. If the service was deleted rather than suspended, create a new **Static Site** from the repo with
   root `frontend`, build `npm install && npm run build`, publish `dist`, the rewrite `/* → /index.html`,
   and the env var above. If its URL changes, update `CORS_ORIGINS_STR` on the backend to match.

### 4. Database and blueprint

- Check the `ai-recruiter-db` page for its expiry date (see the note under "Workspace isolation").
- Optional: to make future changes come from `render.yaml`, go to Blueprints → New Blueprint Instance.
  This creates *new* services named as in the file, so either rename them in `render.yaml` to the live
  names first, or keep managing the live services by hand using the tables above.

---

## 🔄 Updating Frontend Redirect Rules / تحديث قواعد التوجيه في الواجهة
In the Render dashboard for the **Frontend** service (`ai-recruiter-frontend`), go to **Redirects/Rewrites** and ensure you have:
- **Source**: `/api/*`
- **Destination**: `https://<YOUR-BACKEND-NAME>.onrender.com/api/*`
- **Action**: `Rewrite`

*This is pre-configured in `render.yaml` but can be updated manually if you change the backend service name.*

في لوحة تحكم خدمة الواجهة (`ai-recruiter-frontend`)، اذهب إلى قسم **Redirects/Rewrites** وتأكد من تهيئتها كالتالي:
- **المصدر**: `/api/*`
- **الوجهة**: رابط الخدمة الخلفية الخاص بك `https://<YOUR-BACKEND-NAME>.onrender.com/api/*`
- **الإجراء**: `Rewrite`

---

## 📊 Database & Runbook / قاعدة البيانات والتشغيل
- Schema migrations will run automatically upon startup using Alembic (`alembic upgrade head`).
- If you need to backfill existing models or run scripts, you can open the Render shell in the backend service and run commands like:
 `python scripts/backfill_production_readiness.py --dry-run`

### Workspace isolation

Migration `0005_user_data_isolation` adds ownership to jobs, candidates, and feedback. Existing legacy rows are assigned to the first staff account during deployment so they are not exposed to every new staff user. New jobs and CV uploads are automatically assigned to the authenticated owner, admin, or recruiter who created them.

The current demo uses a Free Render Postgres database named `ai-recruiter-db`. Free Render Postgres databases expire 30 days after creation unless upgraded. An earlier note said this instance expires on August 24, 2026, but it was still reachable after that date, so **the expiry date needs verification**: check the database page in the Render dashboard (it shows the expiry, if any) before relying on it for a demo. The web service must use its internal URL through `DATABASE_URL`. Migration `0006_cv_binary_storage` stores uploaded CV bytes in Postgres so downloads survive web-service restarts. Those files count against the free database's 1GB storage limit.

After the migration:

- Staff users see only their own jobs, candidates, matches, reports, feedback, and interview records.
- Cross-user read, update, delete, matching, report, and feedback requests return `404`.
- Candidates still see the shared job board so they can apply to available positions.
- Owner/admin user and role administration remains global by design.

- سيتم تشغيل تحديثات قاعدة البيانات (Migrations) تلقائياً عند إقلاع الخدمة باستخدام Alembic.
- إذا كنت بحاجة لتهيئة النماذج أو تشغيل السكربتات، يمكنك فتح الـ Shell المتاح في Render للخدمة الخلفية وكتابة الأوامر مثل:
  `python scripts/backfill_production_readiness.py --dry-run`
