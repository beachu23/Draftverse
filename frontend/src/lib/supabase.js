import { createClient } from '@supabase/supabase-js'

// Public anon key — safe to expose in the client bundle. Access is protected by
// an insert-only Row-Level Security policy on the rookie_ladder_predictions table.
// Set these in frontend/.env.local (dev) and in Vercel project settings (prod).
const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

// True only when both env vars are present, so the UI can show a clear message
// instead of throwing when Supabase hasn't been configured yet.
export const supabaseReady = Boolean(url && anonKey)

export const supabase = supabaseReady ? createClient(url, anonKey) : null
