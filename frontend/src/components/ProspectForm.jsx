import { useState, useEffect } from 'react'
import { supabase, supabaseReady } from '../lib/supabase'

const POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']
const FEET_OPTIONS = [5, 6, 7, 8]
const ARCADE = '"Press Start 2P", monospace'

// Fields the /similarity endpoint accepts
const BACKEND_FIELDS = new Set([
  'position', 'height_inches', 'weight', 'age_at_draft',
  'combine_max_vertical', 'combine_lane_agility', 'combine_shuttle',
  'combine_three_qtr_sprint', 'combine_wingspan_inches',
  'pg_g', 'pg_mp', 'pg_fg_pct', 'pg_ft_pct',
  'p36_pts', 'p36_reb', 'p36_ast', 'p36_blk', 'p36_stl', 'p36_to', 'p36_pf',
  'adv1_ts_pct', 'adv1_3pa_rate', 'adv1_fta_rate', 'adv1_proj_nba_3p',
  'adv1_usg_pct', 'adv1_ast_usg', 'adv1_ast_to',
  'adv2_per', 'adv2_ows_40', 'adv2_dws_40', 'adv2_obpm', 'adv2_dbpm',
])

function prospectToPayload(p) {
  const out = {}
  for (const [key, val] of Object.entries(p)) {
    if (BACKEND_FIELDS.has(key) && val != null) out[key] = val
  }
  return out
}

function fmtMedianHeight(inches) {
  if (!inches) return ''
  const ft = Math.floor(inches / 12)
  const ins = (inches % 12).toFixed(2).replace(/\.?0+$/, '')
  return `${ft}′ ${ins}″`
}

function SectionToggle({ label, open, onToggle }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      style={{
        display: 'flex', alignItems: 'center', gap: '8px',
        background: 'none', border: 'none', color: '#a0a0b0',
        fontSize: '11px', fontWeight: 500, letterSpacing: '0.08em',
        textTransform: 'uppercase', cursor: 'pointer',
        padding: '8px 0 4px', width: '100%', userSelect: 'none',
      }}
    >
      <span style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: '16px', height: '16px', border: '1px solid #2a2a3a',
        borderRadius: '3px', fontSize: '12px', lineHeight: 1,
        color: '#6060a0', flexShrink: 0,
      }}>
        {open ? '−' : '+'}
      </span>
      {label}
    </button>
  )
}

const inputStyle = {
  background: '#0f0f1a', border: '1px solid #1e1e30', borderRadius: '6px',
  color: '#e0e0f0', fontSize: '13px', padding: '8px 10px', width: '100%',
  outline: 'none', transition: 'border-color 0.15s',
  appearance: 'none', WebkitAppearance: 'none',
}

const labelStyle = {
  display: 'block', fontSize: '11px', fontWeight: 500,
  letterSpacing: '0.06em', textTransform: 'uppercase',
  color: '#6a6a8a', marginBottom: '4px',
}

function Field({ label, children }) {
  return (
    <div>
      <label style={labelStyle}>{label}</label>
      {children}
    </div>
  )
}

function NumericInput({ name, value, onChange, placeholder, step, min, max }) {
  return (
    <input
      type="number" name={name} value={value} onChange={onChange}
      placeholder={placeholder} step={step} min={min} max={max}
      style={inputStyle}
      onFocus={e => { e.target.style.borderColor = '#3a3a5a' }}
      onBlur={e => { e.target.style.borderColor = '#1e1e30' }}
    />
  )
}

function GridRow({ cols = 2, children }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: '10px' }}>
      {children}
    </div>
  )
}

// ── Submit button (shared style) ────────────────────────────────────────────
function SubmitButton({ disabled, label, playersReady }) {
  return (
    <button
      type="submit"
      disabled={disabled}
      style={{
        marginTop: '8px', width: '100%', padding: '14px 12px',
        background: disabled ? '#0f0f1e' : 'rgba(0,4,18,0.96)',
        border: `2px solid ${disabled ? '#1a1a28' : '#3a8fff'}`,
        boxShadow: disabled ? 'none' : '4px 4px 0 #001040, 0 0 16px rgba(58,143,255,0.2)',
        borderRadius: '4px',
        color: disabled ? '#3a3a50' : '#3a8fff',
        fontFamily: ARCADE, fontSize: '7px',
        letterSpacing: '1px', lineHeight: 2,
        cursor: disabled ? 'default' : 'pointer',
        transition: 'color 0.15s, box-shadow 0.15s',
      }}
      onMouseEnter={e => {
        if (!disabled) {
          e.currentTarget.style.color = '#ffffff'
          e.currentTarget.style.boxShadow = '4px 4px 0 #001040, 0 0 24px rgba(58,143,255,0.5)'
        }
      }}
      onMouseLeave={e => {
        if (!disabled) {
          e.currentTarget.style.color = '#3a8fff'
          e.currentTarget.style.boxShadow = '4px 4px 0 #001040, 0 0 16px rgba(58,143,255,0.2)'
        }
      }}
    >
      {!playersReady ? 'LOADING UNIVERSE...' : label}
    </button>
  )
}

// ── Main component ───────────────────────────────────────────────────────────

export default function ProspectForm({ visible, playersReady, medians, onSubmit, prospects2026 = [] }) {
  const [tab, setTab] = useState('2026')
  const [selectedSlug, setSelectedSlug] = useState('')
  const [customMode, setCustomMode] = useState(false)   // swap 2026 dropdown ↔ custom stat entry

  // Rookie ladder state
  const [ladder, setLadder]   = useState(['', '', '', '', ''])   // 5 slugs, rank 1–5
  const [contact, setContact] = useState('')
  const [rookieStatus, setRookieStatus] = useState('idle')       // idle | submitting | success | error
  const [rookieError,  setRookieError]  = useState('')

  // Custom form state
  const [open, setOpen] = useState({ combine: false, perGame: false, per36: false, advanced: false })
  const [vals, setVals] = useState({
    name: '',
    position: '', heightFt: '', heightIn: '', weight: '', age: '',
    maxVertical: '', laneAgility: '', shuttle: '', sprint: '',
    wingspanFt: '', wingspanIn: '',
    pg_g: '', pg_mp: '', pg_fg_pct: '', pg_ft_pct: '',
    p36_pts: '', p36_reb: '', p36_ast: '', p36_blk: '', p36_stl: '', p36_to: '', p36_pf: '',
    ts_pct: '', usg_pct: '', three_pa_rate: '', fta_rate: '', proj_3p: '',
    ast_usg: '', ast_to: '', per: '', ows_40: '', dws_40: '', obpm: '', dbpm: '',
  })

  // Auto-select first prospect once data loads
  useEffect(() => {
    if (prospects2026.length > 0 && !selectedSlug) {
      setSelectedSlug(prospects2026[0].slug)
    }
  }, [prospects2026.length])

  const selectedProspect = prospects2026.find(p => p.slug === selectedSlug) ?? null

  const set = (field) => (e) => setVals(v => ({ ...v, [field]: e.target.value }))
  const toggle = (section) => setOpen(o => ({ ...o, [section]: !o[section] }))

  const m = medians
  const medH = m.height_inches ? fmtMedianHeight(m.height_inches) : ''

  // ── 2026 submit ────────────────────────────────────────────────────────────
  const handle2026Submit = (e) => {
    e.preventDefault()
    if (!selectedProspect || !playersReady) return
    onSubmit({ ...prospectToPayload(selectedProspect), name: selectedProspect.name })
  }

  // ── Custom submit ──────────────────────────────────────────────────────────
  const canSubmitCustom = playersReady && vals.position && vals.heightFt && vals.weight && vals.age

  const handleCustomSubmit = (e) => {
    e.preventDefault()
    if (!canSubmitCustom) return
    const heightInches = parseFloat(vals.heightFt) * 12 + parseFloat(vals.heightIn || 0)
    const wingspanInches = vals.wingspanFt
      ? parseFloat(vals.wingspanFt) * 12 + parseFloat(vals.wingspanIn || 0)
      : null
    onSubmit({
      ...(vals.name && { name: vals.name }),
      position: vals.position,
      height_inches: heightInches,
      weight: parseFloat(vals.weight),
      age_at_draft: parseFloat(vals.age),
      ...(vals.maxVertical  && { combine_max_vertical: parseFloat(vals.maxVertical) }),
      ...(vals.laneAgility  && { combine_lane_agility: parseFloat(vals.laneAgility) }),
      ...(vals.shuttle      && { combine_shuttle: parseFloat(vals.shuttle) }),
      ...(vals.sprint       && { combine_three_qtr_sprint: parseFloat(vals.sprint) }),
      ...(wingspanInches    && { combine_wingspan_inches: wingspanInches }),
      ...(vals.pg_g         && { pg_g: parseFloat(vals.pg_g) }),
      ...(vals.pg_mp        && { pg_mp: parseFloat(vals.pg_mp) }),
      ...(vals.pg_fg_pct    && { pg_fg_pct: parseFloat(vals.pg_fg_pct) }),
      ...(vals.pg_ft_pct    && { pg_ft_pct: parseFloat(vals.pg_ft_pct) }),
      ...(vals.p36_pts      && { p36_pts: parseFloat(vals.p36_pts) }),
      ...(vals.p36_reb      && { p36_reb: parseFloat(vals.p36_reb) }),
      ...(vals.p36_ast      && { p36_ast: parseFloat(vals.p36_ast) }),
      ...(vals.p36_blk      && { p36_blk: parseFloat(vals.p36_blk) }),
      ...(vals.p36_stl      && { p36_stl: parseFloat(vals.p36_stl) }),
      ...(vals.p36_to       && { p36_to: parseFloat(vals.p36_to) }),
      ...(vals.p36_pf       && { p36_pf: parseFloat(vals.p36_pf) }),
      ...(vals.ts_pct       && { adv1_ts_pct: parseFloat(vals.ts_pct) }),
      ...(vals.usg_pct      && { adv1_usg_pct: parseFloat(vals.usg_pct) }),
      ...(vals.three_pa_rate && { adv1_3pa_rate: parseFloat(vals.three_pa_rate) }),
      ...(vals.fta_rate     && { adv1_fta_rate: parseFloat(vals.fta_rate) }),
      ...(vals.proj_3p      && { adv1_proj_nba_3p: parseFloat(vals.proj_3p) }),
      ...(vals.ast_usg      && { adv1_ast_usg: parseFloat(vals.ast_usg) }),
      ...(vals.ast_to       && { adv1_ast_to: parseFloat(vals.ast_to) }),
      ...(vals.per          && { adv2_per: parseFloat(vals.per) }),
      ...(vals.ows_40       && { adv2_ows_40: parseFloat(vals.ows_40) }),
      ...(vals.dws_40       && { adv2_dws_40: parseFloat(vals.dws_40) }),
      ...(vals.obpm         && { adv2_obpm: parseFloat(vals.obpm) }),
      ...(vals.dbpm         && { adv2_dbpm: parseFloat(vals.dbpm) }),
    })
  }

  // ── Rookie ladder submit ─────────────────────────────────────────────────────
  const setLadderSlot = (i, slug) => setLadder(prev => prev.map((s, idx) => (idx === i ? slug : s)))

  const canSubmitRookie =
    playersReady &&
    contact.trim() !== '' &&
    ladder.every(Boolean) &&
    new Set(ladder).size === 5

  const handleRookieSubmit = async (e) => {
    e.preventDefault()
    if (!canSubmitRookie || rookieStatus === 'submitting') return
    if (!supabaseReady) {
      setRookieError('Predictions database is not configured yet (missing Supabase env vars).')
      setRookieStatus('error')
      return
    }
    setRookieStatus('submitting')
    setRookieError('')
    const predictions = ladder.map((slug, i) => {
      const p = prospects2026.find(x => x.slug === slug)
      return { rank: i + 1, slug, name: p?.name ?? slug }
    })
    const { error } = await supabase
      .from('rookie_ladder_predictions')
      .insert([{ contact_method: contact.trim(), predictions }])
    if (error) {
      setRookieError(error.message)
      setRookieStatus('error')
    } else {
      setRookieStatus('success')
    }
  }

  return (
    <div style={{
      position: 'absolute', inset: 0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: '#0a0a0f', zIndex: 10, overflow: 'auto',
      opacity: visible ? 1 : 0,
      pointerEvents: visible ? 'auto' : 'none',
      transition: 'opacity 0.3s ease',
    }}>
      <div style={{
        width: '100%', maxWidth: '520px',
        margin: '40px 20px', padding: '32px',
        background: 'rgba(0,4,18,0.97)',
        border: '2px solid #3a8fff',
        borderRadius: '4px',
        boxShadow: '6px 6px 0 #001040, 0 0 36px rgba(58,143,255,0.15)',
      }}>

        {/* Header */}
        <div style={{ marginBottom: '24px' }}>
          <h1 style={{ margin: '0 0 6px', fontSize: '20px', fontWeight: 600, color: '#d0d0e8', letterSpacing: '-0.02em', lineHeight: 1.2 }}>
            Enter the Draft Universe
          </h1>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid #112244', marginBottom: '24px' }}>
          {[['2026', '2026 CLASS'], ['rookie', 'ROOKIE LADDER']].map(([key, label]) => {
            const active = tab === key
            return (
              <button
                key={key}
                type="button"
                onClick={() => { setTab(key); setCustomMode(false) }}
                style={{
                  fontFamily: ARCADE, fontSize: '6px', letterSpacing: '1px',
                  background: 'none', border: 'none', cursor: 'pointer',
                  padding: '8px 16px 10px',
                  color: active ? '#3a8fff' : '#334466',
                  borderBottom: active ? '2px solid #3a8fff' : '2px solid transparent',
                  marginBottom: '-1px',
                  transition: 'color 0.15s',
                }}
              >
                {label}
              </button>
            )
          })}
        </div>

        {/* ── 2026 CLASS tab (dropdown view) ──────────────────────────────────── */}
        {tab === '2026' && !customMode && (
          <form onSubmit={handle2026Submit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {prospects2026.length === 0 ? (
              <div style={{ fontFamily: ARCADE, fontSize: '6px', color: '#334466', textAlign: 'center', padding: '24px 0' }}>
                LOADING PROSPECTS...
              </div>
            ) : (
              <>
                {/* Dropdown + custom-entry switch */}
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <label style={labelStyle}>Select Prospect</label>
                    <button
                      type="button"
                      onClick={() => setCustomMode(true)}
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        color: '#3a8fff', fontFamily: ARCADE, fontSize: '6px',
                        letterSpacing: '0.5px', padding: '2px 4px',
                      }}
                      onMouseEnter={e => { e.currentTarget.style.color = '#ffffff' }}
                      onMouseLeave={e => { e.currentTarget.style.color = '#3a8fff' }}
                    >
                      + CUSTOM STATS
                    </button>
                  </div>
                  <select
                    value={selectedSlug}
                    onChange={e => setSelectedSlug(e.target.value)}
                    style={{ ...inputStyle, cursor: 'pointer', fontSize: '13px' }}
                    onFocus={e => { e.target.style.borderColor = '#3a3a5a' }}
                    onBlur={e => { e.target.style.borderColor = '#1e1e30' }}
                  >
                    <option value="" disabled>— pick a prospect —</option>
                    {prospects2026.map((p, i) => (
                      <option key={p.slug} value={p.slug}>
                        {i + 1}. {p.name}  ·  {p.position}  ·  {p.school}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Prospect card */}
                {selectedProspect && (
                  <div style={{
                    background: 'rgba(0,4,18,0.97)',
                    border: '2px solid #1a2a44',
                    borderRadius: '4px',
                    padding: '14px 18px',
                    boxShadow: '3px 3px 0 #000d28',
                  }}>
                    <div style={{ fontFamily: ARCADE, fontSize: '8px', color: '#ffdd00', letterSpacing: '1px', marginBottom: '6px', lineHeight: 1.8 }}>
                      {selectedProspect.name.toUpperCase()}
                    </div>
                    <div style={{ fontSize: '12px', color: '#556688', letterSpacing: '0.5px' }}>
                      {selectedProspect.position} · {selectedProspect.school}
                    </div>
                  </div>
                )}
              </>
            )}

            <SubmitButton
              disabled={!selectedProspect || !playersReady}
              label="ENTER THE DRAFT UNIVERSE ▶"
              playersReady={playersReady}
            />
          </form>
        )}

        {/* ── CUSTOM stat entry (swapped in from the 2026 tab) ────────────────── */}
        {tab === '2026' && customMode && (
          <form onSubmit={handleCustomSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            <button
              type="button"
              onClick={() => setCustomMode(false)}
              style={{
                alignSelf: 'flex-start', background: 'none', border: 'none', cursor: 'pointer',
                color: '#3a8fff', fontFamily: ARCADE, fontSize: '6px', letterSpacing: '0.5px', padding: '0',
              }}
              onMouseEnter={e => { e.currentTarget.style.color = '#ffffff' }}
              onMouseLeave={e => { e.currentTarget.style.color = '#3a8fff' }}
            >
              ◀ BACK TO 2026 CLASS
            </button>

            <div style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#3a3a58', paddingBottom: '4px', borderBottom: '1px solid #141420' }}>
              Required
            </div>

            <Field label="Name (optional)">
              <input
                type="text"
                value={vals.name}
                onChange={set('name')}
                style={inputStyle}
                onFocus={e => { e.target.style.borderColor = '#3a3a5a' }}
                onBlur={e => { e.target.style.borderColor = '#1e1e30' }}
              />
            </Field>

            <Field label="Position">
              <select
                value={vals.position}
                onChange={set('position')}
                required
                style={{ ...inputStyle, cursor: 'pointer', color: vals.position ? '#e0e0f0' : '#3a3a58' }}
                onFocus={e => { e.target.style.borderColor = '#3a3a5a' }}
                onBlur={e => { e.target.style.borderColor = '#1e1e30' }}
              >
                <option value="" disabled>Select position</option>
                {POSITIONS.map(p => <option key={p} value={p}>{p}</option>)}
              </select>
            </Field>

            <div>
              <label style={labelStyle}>Height</label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div style={{ position: 'relative' }}>
                  <NumericInput value={vals.heightFt} onChange={set('heightFt')} placeholder="6" min={5} max={8} step={1} />
                  <span style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', color: '#4a4a68', fontSize: '12px', pointerEvents: 'none' }}>ft</span>
                </div>
                <div style={{ position: 'relative' }}>
                  <NumericInput value={vals.heightIn} onChange={set('heightIn')} placeholder={medH ? medH.split('′')[1].replace('″', '').trim() : ''} step={0.25} min={0} max={11.75} />
                  <span style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', color: '#4a4a68', fontSize: '12px', pointerEvents: 'none' }}>in</span>
                </div>
              </div>
            </div>

            <GridRow cols={2}>
              <Field label="Weight (lbs)">
                <NumericInput value={vals.weight} onChange={set('weight')} placeholder={m.weight != null ? String(Math.round(m.weight)) : ''} />
              </Field>
              <Field label="Age at Draft">
                <NumericInput value={vals.age} onChange={set('age')} placeholder={m.age_at_draft != null ? parseFloat(m.age_at_draft).toFixed(1) : ''} step={0.1} />
              </Field>
            </GridRow>

            {/* Combine */}
            <div>
              <SectionToggle label="Combine Measurements" open={open.combine} onToggle={() => toggle('combine')} />
              {open.combine && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '10px' }}>
                  <GridRow cols={2}>
                    <Field label="Max Vertical (in)">
                      <NumericInput value={vals.maxVertical} onChange={set('maxVertical')} placeholder={m.combine_max_vertical?.toFixed(1) ?? ''} step={0.5} />
                    </Field>
                    <Field label="Lane Agility (s)">
                      <NumericInput value={vals.laneAgility} onChange={set('laneAgility')} placeholder={m.combine_lane_agility?.toFixed(2) ?? ''} step={0.01} />
                    </Field>
                  </GridRow>
                  <GridRow cols={2}>
                    <Field label="Shuttle (s)">
                      <NumericInput value={vals.shuttle} onChange={set('shuttle')} placeholder={m.combine_shuttle?.toFixed(2) ?? ''} step={0.01} />
                    </Field>
                    <Field label="3/4 Sprint (s)">
                      <NumericInput value={vals.sprint} onChange={set('sprint')} placeholder={m.combine_three_qtr_sprint?.toFixed(2) ?? ''} step={0.01} />
                    </Field>
                  </GridRow>
                  <Field label="Wingspan">
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                      <select
                        value={vals.wingspanFt}
                        onChange={set('wingspanFt')}
                        style={{ ...inputStyle, cursor: 'pointer', color: vals.wingspanFt ? '#e0e0f0' : '#3a3a58' }}
                        onFocus={e => { e.target.style.borderColor = '#3a3a5a' }}
                        onBlur={e => { e.target.style.borderColor = '#1e1e30' }}
                      >
                        <option value="">ft</option>
                        {FEET_OPTIONS.map(f => <option key={f} value={f}>{f} ft</option>)}
                      </select>
                      <NumericInput value={vals.wingspanIn} onChange={set('wingspanIn')} placeholder="inches" step={0.25} min={0} max={11.75} />
                    </div>
                  </Field>
                </div>
              )}
            </div>

            {/* Per Game */}
            <div>
              <SectionToggle label="Stats — Per Game" open={open.perGame} onToggle={() => toggle('perGame')} />
              {open.perGame && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', marginTop: '10px' }}>
                  <Field label="Games"><NumericInput value={vals.pg_g} onChange={set('pg_g')} placeholder={m.pg_g ? String(Math.round(m.pg_g)) : ''} /></Field>
                  <Field label="Minutes"><NumericInput value={vals.pg_mp} onChange={set('pg_mp')} placeholder={m.pg_mp?.toFixed(1) ?? ''} step={0.1} /></Field>
                  <Field label="FG%"><NumericInput value={vals.pg_fg_pct} onChange={set('pg_fg_pct')} placeholder={m.pg_fg_pct?.toFixed(3) ?? ''} step={0.001} min={0} max={1} /></Field>
                  <Field label="FT%"><NumericInput value={vals.pg_ft_pct} onChange={set('pg_ft_pct')} placeholder={m.pg_ft_pct?.toFixed(3) ?? ''} step={0.001} min={0} max={1} /></Field>
                </div>
              )}
            </div>

            {/* Per 36 */}
            <div>
              <SectionToggle label="Stats — Per 36" open={open.per36} onToggle={() => toggle('per36')} />
              {open.per36 && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', marginTop: '10px' }}>
                  {[
                    ['PTS', 'p36_pts', 'p36_pts'], ['REB', 'p36_reb', 'p36_reb'],
                    ['AST', 'p36_ast', 'p36_ast'], ['BLK', 'p36_blk', 'p36_blk'],
                    ['STL', 'p36_stl', 'p36_stl'], ['TO',  'p36_to',  'p36_to'],
                    ['PF',  'p36_pf',  'p36_pf'],
                  ].map(([lbl, field, mkey]) => (
                    <Field key={field} label={lbl}>
                      <NumericInput value={vals[field]} onChange={set(field)} placeholder={m[mkey]?.toFixed(1) ?? ''} step={0.1} />
                    </Field>
                  ))}
                </div>
              )}
            </div>

            {/* Advanced */}
            <div>
              <SectionToggle label="Advanced Stats" open={open.advanced} onToggle={() => toggle('advanced')} />
              {open.advanced && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', marginTop: '10px' }}>
                  {[
                    ['TS%',         'ts_pct',        'adv1_ts_pct',        0.001],
                    ['USG%',        'usg_pct',        'adv1_usg_pct',       0.001],
                    ['3PA Rate',    'three_pa_rate',  'adv1_3pa_rate',      0.001],
                    ['FTA Rate',    'fta_rate',       'adv1_fta_rate',      0.001],
                    ['Proj NBA 3P%','proj_3p',        'adv1_proj_nba_3p',   0.001],
                    ['AST/USG',     'ast_usg',        'adv1_ast_usg',       0.01],
                    ['AST/TO',      'ast_to',         'adv1_ast_to',        0.01],
                    ['PER',         'per',            'adv2_per',           0.1],
                    ['OWS/40',      'ows_40',         'adv2_ows_40',        0.01],
                    ['DWS/40',      'dws_40',         'adv2_dws_40',        0.01],
                    ['OBPM',        'obpm',           'adv2_obpm',          0.1],
                    ['DBPM',        'dbpm',           'adv2_dbpm',          0.1],
                  ].map(([lbl, field, mkey, step]) => (
                    <Field key={field} label={lbl}>
                      <NumericInput value={vals[field]} onChange={set(field)} placeholder={m[mkey]?.toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : 1) ?? ''} step={step} />
                    </Field>
                  ))}
                </div>
              )}
            </div>

            <SubmitButton
              disabled={!canSubmitCustom}
              label="ENTER THE DRAFT UNIVERSE ▶"
              playersReady={playersReady}
            />
          </form>
        )}

        {/* ── ROOKIE LADDER tab ──────────────────────────────────────────────── */}
        {tab === 'rookie' && (
          rookieStatus === 'success' ? (
            <div style={{ textAlign: 'center', padding: '24px 8px' }}>
              <div style={{ fontFamily: ARCADE, fontSize: '9px', color: '#ffdd00', letterSpacing: '1px', lineHeight: 2 }}>
                PREDICTION LOGGED
              </div>
              <div style={{ fontSize: '12px', color: '#7aadff', marginTop: '14px', lineHeight: 1.6 }}>
                Thanks — your rookie ladder is saved. We&apos;ll reach out via the contact you gave.
              </div>
              <button
                type="button"
                onClick={() => { setLadder(['', '', '', '', '']); setContact(''); setRookieStatus('idle') }}
                style={{
                  marginTop: '20px', background: 'none', border: 'none', cursor: 'pointer',
                  color: '#3a8fff', fontFamily: ARCADE, fontSize: '6px', letterSpacing: '0.5px',
                }}
              >
                + SUBMIT ANOTHER
              </button>
            </div>
          ) : (
          <form onSubmit={handleRookieSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            <div style={{ fontSize: '12px', color: '#7aadff', lineHeight: 1.6 }}>
              Predict the 2026 <span style={{ color: '#ffdd00' }}>Rookie Ladder</span>!
            </div>

            {prospects2026.length === 0 ? (
              <div style={{ fontFamily: ARCADE, fontSize: '6px', color: '#334466', textAlign: 'center', padding: '24px 0' }}>
                LOADING PROSPECTS...
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {ladder.map((slug, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ fontFamily: ARCADE, fontSize: '9px', color: '#ffdd00', width: '14px', textAlign: 'right', flexShrink: 0 }}>
                      {i + 1}
                    </div>
                    <select
                      value={slug}
                      onChange={e => setLadderSlot(i, e.target.value)}
                      style={{ ...inputStyle, cursor: 'pointer', color: slug ? '#e0e0f0' : '#3a3a58' }}
                      onFocus={e => { e.target.style.borderColor = '#3a3a5a' }}
                      onBlur={e => { e.target.style.borderColor = '#1e1e30' }}
                    >
                      <option value="" disabled>— pick a rookie —</option>
                      {prospects2026.map(p => {
                        const taken = ladder.includes(p.slug) && slug !== p.slug
                        return (
                          <option
                            key={p.slug}
                            value={p.slug}
                            disabled={taken}
                            style={taken
                              ? { color: '#2b2b38', background: '#08080d' }
                              : { color: '#e0e0f0', background: '#0f0f1a' }}
                          >
                            {p.name}  ·  {p.position}
                          </option>
                        )
                      })}
                    </select>
                  </div>
                ))}
              </div>
            )}

            <Field label="Preferred contact method">
              <input
                type="text"
                value={contact}
                onChange={e => setContact(e.target.value)}
                placeholder="email, phone, Instagram, Discord"
                style={inputStyle}
                onFocus={e => { e.target.style.borderColor = '#3a3a5a' }}
                onBlur={e => { e.target.style.borderColor = '#1e1e30' }}
              />
            </Field>

            {rookieStatus === 'error' && (
              <div style={{ color: '#ff4444', fontSize: '11px', lineHeight: 1.5 }}>
                {rookieError || 'Something went wrong — please try again.'}
              </div>
            )}

            <SubmitButton
              disabled={!canSubmitRookie || rookieStatus === 'submitting'}
              label={rookieStatus === 'submitting' ? 'SUBMITTING...' : 'SUBMIT PREDICTION ▶'}
              playersReady={playersReady}
            />
          </form>
          )
        )}
      </div>
    </div>
  )
}
