# Deployment Guide / دليل النشر والتشغيل (Render)

This project is configured to be deployed on **Render** using a single-click blueprint setup.

يحتوي هذا المشروع على تهيئة جاهزة للنشر المباشر على منصة **Render** باستخدام ملف الـ Blueprint (`render.yaml`).

---

## 🛠️ Deployment Steps / خطوات النشر

### 1. Update GitHub / تحديث مستودع جيت هاب
The changes have been pushed to your GitHub repository:
`https://github.com/jadAkeel/AI-recuriter.git`

تم تحديث ورفع كافة التعديلات إلى مستودع GitHub الخاص بك.

### 2. Deploy on Render / النشر على Render
1. Go to the [Render Dashboard](https://dashboard.render.com/).
2. Click **New** (top right) and select **Blueprint**.
3. Connect your GitHub account and select the **AI-recuriter** repository.
4. Render will automatically detect the `render.yaml` blueprint. Click **Apply**.
5. Render will now provision:
   - A PostgreSQL Database (with `pgvector` support).
   - A Redis Instance (for task queuing).
   - FastAPI Backend Web Service.
   - Vite React Frontend Static Site.

1. توجه إلى [لوحة تحكم Render](https://dashboard.render.com/).
2. اضغط على زر **New** في أعلى اليمين واختر **Blueprint**.
3. قم بربط حساب GitHub الخاص بك واختيار مستودع **AI-recuriter**.
4. سيقرأ Render ملف `render.yaml` تلقائياً. اضغط على **Apply**.
5. سيقوم Render بإنشاء وتشغيل المكونات التالية تلقائياً:
   - قاعدة بيانات PostgreSQL (مع دعم `pgvector`).
   - خادم Redis (لإدارة طوابير المهام الخلفية).
   - الخدمة الخلفية FastAPI Backend.
   - واجهة المستخدم الثابتة Vite React Frontend.

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

The current demo uses a Free Render Postgres database named `ai-recruiter-db`. Free Render Postgres databases expire after 30 days; this instance expires on **August 24, 2026** unless upgraded or replaced. The web service must use its internal URL through `DATABASE_URL`. Migration `0006_cv_binary_storage` stores uploaded CV bytes in Postgres so downloads survive web-service restarts. Those files count against the free database's 1GB storage limit.

After the migration:

- Staff users see only their own jobs, candidates, matches, reports, feedback, and interview records.
- Cross-user read, update, delete, matching, report, and feedback requests return `404`.
- Candidates still see the shared job board so they can apply to available positions.
- Owner/admin user and role administration remains global by design.

- سيتم تشغيل تحديثات قاعدة البيانات (Migrations) تلقائياً عند إقلاع الخدمة باستخدام Alembic.
- إذا كنت بحاجة لتهيئة النماذج أو تشغيل السكربتات، يمكنك فتح الـ Shell المتاح في Render للخدمة الخلفية وكتابة الأوامر مثل:
  `python scripts/backfill_production_readiness.py --dry-run`
