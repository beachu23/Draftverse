import { useState, useEffect, useRef } from 'react'

const POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']
const FEET_OPTIONS = [5, 6, 7, 8]

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
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        background: 'none',
        border: 'none',
        color: '#a0a0b0',
        fontSize: '11px',
        fontWeight: 500,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        cursor: 'pointer',
        padding: '8px 0 4px',
        width: '100%',
        userSelect: 'none',
      }}
    >
      <span style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '16px',
        height: '16px',
        border: '1px solid #2a2a3a',
        borderRadius: '3px',
        fontSize: '12px',
        lineHeight: 1,
        color: '#6060a0',
        flexShrink: 0,
      }}>
        {open ? '−' : '+'}
      </span>
      {label}
    </button>
  )
}

const inputStyle = {
  background: '#0f0f1a',
  border: '1px solid #1e1e30',
  borderRadius: '6px',
  color: '#e0e0f0',
  fontSize: '13px',
  padding: '8px 10px',
  width: '100%',
  outline: 'none',
  transition: 'border-color 0.15s',
  appearance: 'none',
  WebkitAppearance: 'none',
}

const labelStyle = {
  display: 'block',
  fontSize: '11px',
  fontWeight: 500,
  letterSpacing: '0.06em',
  textTransform: 'uppercase',
  color: '#6a6a8a',
  marginBottom: '4px',
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
      type="number"
      name={name}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      step={step}
      min={min}
      max={max}
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

export default function ProspectForm({ visible, playersReady, medians, onSubmit }) {
  const [open, setOpen] = useState({ combine: false, perGame: false, per36: false, advanced: false })
  const [vals, setVals] = useState({
    position: '', heightFt: '6', heightIn: '', weight: '', age: '',
    maxVertical: '', laneAgility: '', shuttle: '', sprint: '',
    wingspanFt: '', wingspanIn: '',
    pg_g: '', pg_mp: '', pg_fg_pct: '', pg_ft_pct: '',
    p36_pts: '', p36_reb: '', p36_ast: '', p36_blk: '', p36_stl: '', p36_to: '', p36_pf: '',
    ts_pct: '', usg_pct: '', three_pa_rate: '', fta_rate: '', proj_3p: '',
    ast_usg: '', ast_to: '', per: '', ows_40: '', dws_40: '', obpm: '', dbpm: '',
  })

  const set = (field) => (e) => setVals(v => ({ ...v, [field]: e.target.value }))
  const toggle = (section) => setOpen(o => ({ ...o, [section]: !o[section] }))

  const m = medians // shorthand
  const medH = m.height_inches ? fmtMedianHeight(m.height_inches) : ''

  const canSubmit = playersReady && vals.position && vals.heightFt && vals.weight && vals.age

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!canSubmit) return
    const heightInches = parseFloat(vals.heightFt) * 12 + parseFloat(vals.heightIn || 0)
    const wingspanInches = vals.wingspanFt
      ? parseFloat(vals.wingspanFt) * 12 + parseFloat(vals.wingspanIn || 0)
      : null

    const out = {
      position: vals.position,
      height_inches: heightInches,
      weight: parseFloat(vals.weight),
      age_at_draft: parseFloat(vals.age),
      ...(vals.maxVertical && { combine_max_vertical: parseFloat(vals.maxVertical) }),
      ...(vals.laneAgility && { combine_lane_agility: parseFloat(vals.laneAgility) }),
      ...(vals.shuttle && { combine_shuttle: parseFloat(vals.shuttle) }),
      ...(vals.sprint && { combine_three_qtr_sprint: parseFloat(vals.sprint) }),
      ...(wingspanInches && { combine_wingspan_inches: wingspanInches }),
      ...(vals.pg_g && { pg_g: parseFloat(vals.pg_g) }),
      ...(vals.pg_mp && { pg_mp: parseFloat(vals.pg_mp) }),
      ...(vals.pg_fg_pct && { pg_fg_pct: parseFloat(vals.pg_fg_pct) }),
      ...(vals.pg_ft_pct && { pg_ft_pct: parseFloat(vals.pg_ft_pct) }),
      ...(vals.p36_pts && { p36_pts: parseFloat(vals.p36_pts) }),
      ...(vals.p36_reb && { p36_reb: parseFloat(vals.p36_reb) }),
      ...(vals.p36_ast && { p36_ast: parseFloat(vals.p36_ast) }),
      ...(vals.p36_blk && { p36_blk: parseFloat(vals.p36_blk) }),
      ...(vals.p36_stl && { p36_stl: parseFloat(vals.p36_stl) }),
      ...(vals.p36_to && { p36_to: parseFloat(vals.p36_to) }),
      ...(vals.p36_pf && { p36_pf: parseFloat(vals.p36_pf) }),
      ...(vals.ts_pct && { adv1_ts_pct: parseFloat(vals.ts_pct) }),
      ...(vals.usg_pct && { adv1_usg_pct: parseFloat(vals.usg_pct) }),
      ...(vals.three_pa_rate && { adv1_3pa_rate: parseFloat(vals.three_pa_rate) }),
      ...(vals.fta_rate && { adv1_fta_rate: parseFloat(vals.fta_rate) }),
      ...(vals.proj_3p && { adv1_proj_nba_3p: parseFloat(vals.proj_3p) }),
      ...(vals.ast_usg && { adv1_ast_usg: parseFloat(vals.ast_usg) }),
      ...(vals.ast_to && { adv1_ast_to: parseFloat(vals.ast_to) }),
      ...(vals.per && { adv2_per: parseFloat(vals.per) }),
      ...(vals.ows_40 && { adv2_ows_40: parseFloat(vals.ows_40) }),
      ...(vals.dws_40 && { adv2_dws_40: parseFloat(vals.dws_40) }),
      ...(vals.obpm && { adv2_obpm: parseFloat(vals.obpm) }),
      ...(vals.dbpm && { adv2_dbpm: parseFloat(vals.dbpm) }),
    }
    onSubmit(out)
  }

  return (
    <div style={{
      position: 'absolute',
      inset: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#0a0a0f',
      zIndex: 10,
      overflow: 'auto',
      opacity: visible ? 1 : 0,
      pointerEvents: visible ? 'auto' : 'none',
      transition: 'opacity 0.3s ease',
    }}>
      <form
        onSubmit={handleSubmit}
        style={{
          width: '100%',
          maxWidth: '480px',
          margin: '40px 20px',
          padding: '32px',
          background: '#0d0d18',
          border: '1px solid #1a1a28',
          borderRadius: '12px',
        }}
      >
        {/* Header */}
        <div style={{ marginBottom: '28px' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#4a4a70', marginBottom: '8px' }}>
            NBA Draft Comps
          </div>
          <h1 style={{ margin: 0, fontSize: '22px', fontWeight: 600, color: '#d0d0e8', letterSpacing: '-0.02em', lineHeight: 1.2 }}>
            Enter a prospect profile
          </h1>
          <p style={{ margin: '6px 0 0', fontSize: '13px', color: '#4a4a68', lineHeight: 1.5 }}>
            Find your nearest historical draft comps in the 3D draft universe.
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

          {/* Required section */}
          <div style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: '#3a3a58', paddingBottom: '4px', borderBottom: '1px solid #141420' }}>
            Required
          </div>

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
                <NumericInput
                  value={vals.heightFt}
                  onChange={set('heightFt')}
                  placeholder="6"
                  min={5}
                  max={8}
                  step={1}
                />
                <span style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', color: '#4a4a68', fontSize: '12px', pointerEvents: 'none' }}>ft</span>
              </div>
              <div style={{ position: 'relative' }}>
                <NumericInput
                  value={vals.heightIn}
                  onChange={set('heightIn')}
                  placeholder={medH ? medH.split('′')[1].replace('″', '').trim() : ''}
                  step={0.25}
                  min={0}
                  max={11.75}
                />
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
                <Field label="Games">
                  <NumericInput value={vals.pg_g} onChange={set('pg_g')} placeholder={m.pg_g ? String(Math.round(m.pg_g)) : ''} />
                </Field>
                <Field label="Minutes">
                  <NumericInput value={vals.pg_mp} onChange={set('pg_mp')} placeholder={m.pg_mp?.toFixed(1) ?? ''} step={0.1} />
                </Field>
                <Field label="FG%">
                  <NumericInput value={vals.pg_fg_pct} onChange={set('pg_fg_pct')} placeholder={m.pg_fg_pct?.toFixed(3) ?? ''} step={0.001} min={0} max={1} />
                </Field>
                <Field label="FT%">
                  <NumericInput value={vals.pg_ft_pct} onChange={set('pg_ft_pct')} placeholder={m.pg_ft_pct?.toFixed(3) ?? ''} step={0.001} min={0} max={1} />
                </Field>
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
                  ['STL', 'p36_stl', 'p36_stl'], ['TO', 'p36_to', 'p36_to'],
                  ['PF', 'p36_pf', 'p36_pf'],
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
                  ['TS%', 'ts_pct', 'adv1_ts_pct', 0.001],
                  ['USG%', 'usg_pct', 'adv1_usg_pct', 0.001],
                  ['3PA Rate', 'three_pa_rate', 'adv1_3pa_rate', 0.001],
                  ['FTA Rate', 'fta_rate', 'adv1_fta_rate', 0.001],
                  ['Proj NBA 3P%', 'proj_3p', 'adv1_proj_nba_3p', 0.001],
                  ['AST/USG', 'ast_usg', 'adv1_ast_usg', 0.01],
                  ['AST/TO', 'ast_to', 'adv1_ast_to', 0.01],
                  ['PER', 'per', 'adv2_per', 0.1],
                  ['OWS/40', 'ows_40', 'adv2_ows_40', 0.01],
                  ['DWS/40', 'dws_40', 'adv2_dws_40', 0.01],
                  ['OBPM', 'obpm', 'adv2_obpm', 0.1],
                  ['DBPM', 'dbpm', 'adv2_dbpm', 0.1],
                ].map(([lbl, field, mkey, step]) => (
                  <Field key={field} label={lbl}>
                    <NumericInput value={vals[field]} onChange={set(field)} placeholder={m[mkey]?.toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : 1) ?? ''} step={step} />
                  </Field>
                ))}
              </div>
            )}
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={!canSubmit}
            style={{
              marginTop: '8px',
              width: '100%',
              padding: '12px',
              background: canSubmit ? '#1a1a38' : '#0f0f1e',
              border: `1px solid ${canSubmit ? '#3a3a6a' : '#1a1a28'}`,
              borderRadius: '8px',
              color: canSubmit ? '#c0c0e8' : '#3a3a50',
              fontSize: '13px',
              fontWeight: 500,
              letterSpacing: '0.04em',
              cursor: canSubmit ? 'pointer' : 'default',
              transition: 'background 0.15s, border-color 0.15s, color 0.15s',
            }}
            onMouseEnter={e => { if (canSubmit) e.target.style.background = '#20204a' }}
            onMouseLeave={e => { if (canSubmit) e.target.style.background = '#1a1a38' }}
          >
            {playersReady ? 'Enter the Draft Universe' : 'Preparing universe...'}
          </button>
        </div>
      </form>
    </div>
  )
}
