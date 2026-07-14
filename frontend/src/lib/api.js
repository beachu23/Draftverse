// Base URL for the FastAPI backend. In dev this defaults to the local server;
// in prod (Vercel) set VITE_API_BASE to the Railway backend URL. Vite reads env
// vars only at startup, so restart the dev server after changing .env.local.
export const API_BASE =
  import.meta.env.VITE_API_BASE || 'http://localhost:8000'
