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

// Ensure there's an (anonymous) auth session so each device maps to a stable
// user_id. Reuses the persisted session on reload; only signs in when none
// exists. This backs the one-prediction-per-user unique constraint on the
// rookie_ladder_predictions table (predictions default user_id = auth.uid()).
export async function ensureAnonSession() {
  if (!supabase) return null
  const { data: { session } } = await supabase.auth.getSession()
  if (session?.user) return session.user
  const { data, error } = await supabase.auth.signInAnonymously()
  if (error) throw error
  return data.user
}
