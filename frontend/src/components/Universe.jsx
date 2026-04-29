import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import gsap from 'gsap'

// ─── Pixel helpers ────────────────────────────────────────────────────────────

function px(ctx, x, y, color) {
  if (x < 0 || x >= ctx.canvas.width || y < 0 || y >= ctx.canvas.height) return
  ctx.fillStyle = color
  ctx.fillRect(x, y, 1, 1)
}

function rasterStar(ctx, mid, armLen, diagLen, fill, outlineColor, brightColor) {
  const S = ctx.canvas.width
  const litPixels = new Set()

  for (let d = 1; d <= armLen; d++) {
    const w = Math.max(1, 3 - Math.floor(d * 3 / armLen))
    const half = Math.floor(w / 2)
    for (let o = -half; o <= half; o++) {
      litPixels.add(`${mid + d},${mid + o}`)
      litPixels.add(`${mid - d},${mid + o}`)
      litPixels.add(`${mid + o},${mid + d}`)
      litPixels.add(`${mid + o},${mid - d}`)
    }
  }

  for (let d = 1; d <= diagLen; d++) {
    const w = Math.max(1, 3 - Math.floor(d * 3 / diagLen))
    const half = Math.floor(w / 2)
    for (let o = -half; o <= half; o++) {
      litPixels.add(`${mid + d + o},${mid - d + o}`)
      litPixels.add(`${mid - d + o},${mid - d - o}`)
      litPixels.add(`${mid + d + o},${mid + d - o}`)
      litPixels.add(`${mid - d + o},${mid + d + o}`)
    }
  }

  for (let dx = -2; dx <= 2; dx++)
    for (let dy = -2; dy <= 2; dy++)
      litPixels.add(`${mid + dx},${mid + dy}`)

  const dirs = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]
  litPixels.forEach(key => {
    const [x, y] = key.split(',').map(Number)
    dirs.forEach(([dx, dy]) => {
      const nx = x + dx, ny = y + dy
      if (!litPixels.has(`${nx},${ny}`)) px(ctx, nx, ny, outlineColor)
    })
  })
  litPixels.forEach(key => {
    const [x, y] = key.split(',').map(Number)
    px(ctx, x, y, fill)
  })
  px(ctx, mid - 1, mid - 1, brightColor)
  px(ctx, mid,     mid - 1, brightColor)
  px(ctx, mid - 1, mid,     brightColor)
  px(ctx, mid,     mid,     brightColor)
}

// ─── Texture factories ────────────────────────────────────────────────────────

const ARCADE_PALETTES = [
  { fill: '#7ecfff', outline: '#004a8f', bright: '#dff4ff' },
  { fill: '#3a8fff', outline: '#001f6e', bright: '#c0ddff' },
  { fill: '#eaf6ff', outline: '#5599cc', bright: '#ffffff' },
  { fill: '#00e5ff', outline: '#006080', bright: '#afffff' },
  { fill: '#2255cc', outline: '#000e44', bright: '#88aaff' },
]

function makeArcadeStarTexture(variant) {
  const S = 32, mid = S / 2
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = S
  const ctx = canvas.getContext('2d')
  ctx.imageSmoothingEnabled = false
  const { fill, outline, bright } = ARCADE_PALETTES[variant % 5]
  rasterStar(ctx, mid, 7, 0, fill, outline, bright)
  return new THREE.CanvasTexture(canvas)
}

function makeProspect8StarTexture() {
  const S = 32, mid = S / 2
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = S
  const ctx = canvas.getContext('2d')
  ctx.imageSmoothingEnabled = false
  rasterStar(ctx, mid, 8, 5, '#ffdd00', '#883300', '#ffffff')
  return new THREE.CanvasTexture(canvas)
}

function makeCompArcadeStarTexture() {
  const S = 32, mid = S / 2
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = S
  const ctx = canvas.getContext('2d')
  ctx.imageSmoothingEnabled = false
  rasterStar(ctx, mid, 7, 0, '#ff7700', '#ffffff', '#ffffff')
  return new THREE.CanvasTexture(canvas)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const toKey = (raw) => {
  const noSchool = raw.includes('|') ? raw.split('|')[0].trim() : raw.trim()
  const noPosition = noSchool.replace(/\s+[A-Z]{1,2}(\/[A-Z]{1,2})*$/, '').trim()
  return noPosition.normalize('NFD').replace(/[̀-ͯ]/g, '')
}

const fmtStat = v => (v != null && v !== '') ? (+v).toFixed(1) : '—'
const fmtHt   = h => h ? `${Math.floor(h / 12)}' ${Math.round(h % 12)}"` : '—'

function parseScout(text) {
  const animal    = text.match(/ANIMAL:\s*(.+?)(\n|$)/i)?.[1]?.trim()   ?? null
  const archetype = text.match(/ARCHETYPE:\s*(.+?)(\n|$)/i)?.[1]?.trim() ?? null
  const writeup   = text.match(/WRITEUP:\s*([\s\S]*)/i)?.[1]?.trim()    ?? null
  return { animal, archetype, writeup }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Universe({ players, visible, prospectData }) {
  const mountRef             = useRef(null)
  const sceneRef             = useRef(null)
  const meanRef              = useRef({ x: 0, y: 0, z: 0 })
  const spritesMapRef        = useRef(new Map())
  const animatingRef         = useRef(false)
  const composerRef          = useRef(null)
  const prospectRingRef      = useRef(null)
  const prospectSpriteRef    = useRef(null)
  const clockRef             = useRef(new THREE.Clock())
  const blinkStarsRef        = useRef([])        // { sprite, phase, period, lo, hi, opacityMul }
  const bgStarsRef           = useRef(null)
  const compSpritesRef       = useRef([])
  const savedNavRef          = useRef(null)
  const arrangementLockedRef = useRef(false)

  const keysRef = useRef(new Set())
  const navRef  = useRef({
    active: false, yaw: 0, pitch: 0,
    pos: new THREE.Vector3(0, 0, 300),
    dragging: false, dragX: 0, dragY: 0, scrollVel: 0,
  })

  const [compData, setCompData]                   = useState(null)
  const [prospectCoords, setProspectCoords]       = useState(null)
  const [arrangementActive, setArrangementActive] = useState(false)
  const [overlayFadeIn,     setOverlayFadeIn]     = useState(false)
  const [compLabelData,     setCompLabelData]     = useState([])
  const [prospectBubble,    setProspectBubble]    = useState(null)
  const [scoutingText,      setScoutingText]      = useState('')
  const [scoutingLoading,   setScoutingLoading]   = useState(false)

  // Derive parsed scouting data every render (cheap pure fn)
  const scout = parseScout(scoutingText)

  // ── Arcade font injection ─────────────────────────────────────────────────
  useEffect(() => {
    const link = document.createElement('link')
    link.rel  = 'stylesheet'
    link.href = 'https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap'
    document.head.appendChild(link)
    return () => { if (document.head.contains(link)) document.head.removeChild(link) }
  }, [])

  // ── Continue to explore ───────────────────────────────────────────────────
  function handleContinueExplore() {
    const saved = savedNavRef.current
    if (!saved) return

    // Fade out overlay then unmount
    setOverlayFadeIn(false)
    setTimeout(() => {
      setArrangementActive(false)
      setScoutingText('')
    }, 650)

    // Restore field-star opacity via opacityMul
    blinkStarsRef.current.forEach(entry => {
      if (!saved.compSprites.includes(entry.sprite)) {
        gsap.to(entry, { opacityMul: 1, duration: 0.8, ease: 'power2.out' })
      }
    })

    // Restore background points
    if (bgStarsRef.current) {
      gsap.to(bgStarsRef.current.material, { opacity: 0.45, duration: 0.8 })
    }

    // Glide comp sprites back to original UMAP positions
    saved.compSprites.forEach((sprite, i) => {
      const orig = saved.compOrigPositions[i]
      if (orig) gsap.to(sprite.position, { x: orig.x, y: orig.y, z: orig.z, duration: 0.8, ease: 'power2.inOut' })
    })

    // Restore WASD nav from snapshot
    const nav       = navRef.current
    nav.pos.copy(saved.pos)
    nav.yaw         = saved.yaw
    nav.pitch       = saved.pitch
    nav.active      = true
    arrangementLockedRef.current = false
    if (mountRef.current) mountRef.current.style.cursor = 'crosshair'
  }

  // ── Gemini streaming ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!arrangementActive || !prospectData) return

    setScoutingText('')
    setScoutingLoading(true)
    let cancelled = false

    fetch('http://localhost:8000/similarity/blurb', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prospectData),
    })
      .then(async res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const reader  = res.body.getReader()
        const decoder = new TextDecoder()
        while (true) {
          if (cancelled) break
          const { done, value } = await reader.read()
          if (done) break
          setScoutingText(prev => prev + decoder.decode(value, { stream: true }))
        }
        if (!cancelled) setScoutingLoading(false)
      })
      .catch(err => {
        if (cancelled) return
        console.error('Gemini blurb failed:', err)
        setScoutingText(`[GEMINI_ERROR: ${err.message}]`)
        setScoutingLoading(false)
      })

    return () => { cancelled = true }
  }, [arrangementActive, prospectData])

  // ── Scene setup ──────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current
    if (!mount || !players?.length) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)

    const camera = new THREE.PerspectiveCamera(70, mount.clientWidth / mount.clientHeight, 0.1, 10000)
    camera.position.set(0, 0, 300)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping      = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.1
    mount.appendChild(renderer.domElement)

    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(mount.clientWidth, mount.clientHeight),
      1.2, 0.3, 0.55,
    )
    composer.addPass(bloomPass)
    composer.addPass(new OutputPass())
    composerRef.current = composer

    const starTextures = Array.from({ length: 5 }, (_, i) => makeArcadeStarTexture(i))

    const xs = players.map(p => p.umap_x)
    const ys = players.map(p => p.umap_y)
    const zs = players.map(p => p.umap_z)
    const meanX = xs.reduce((a, b) => a + b, 0) / xs.length
    const meanY = ys.reduce((a, b) => a + b, 0) / ys.length
    const meanZ = zs.reduce((a, b) => a + b, 0) / zs.length
    meanRef.current = { x: meanX, y: meanY, z: meanZ }

    const spritesMap = new Map()
    const blinkStars = []

    players.forEach(player => {
      const variant = Math.floor(Math.random() * 5)
      const mat = new THREE.SpriteMaterial({
        map: starTextures[variant], transparent: true,
        depthWrite: false, blending: THREE.AdditiveBlending,
      })
      const sprite = new THREE.Sprite(mat)
      sprite.position.set(
        (player.umap_x - meanX) * 100,
        (player.umap_y - meanY) * 100,
        (player.umap_z - meanZ) * 150,
      )
      const base = Math.random() < 0.8 ? 7 + Math.random() * 4 : 11 + Math.random() * 3
      sprite.scale.set(base, base, 1)
      sprite.userData = { name: player.name }
      scene.add(sprite)
      spritesMap.set(toKey(player.name), sprite)
      blinkStars.push({
        sprite,
        phase:      Math.random() * Math.PI * 2,
        period:     1.8 + Math.random() * 3.2,
        lo:         0.55 + Math.random() * 0.15,
        hi:         0.85 + Math.random() * 0.15,
        opacityMul: 1,   // ← tweened during arrangement
      })
    })
    spritesMapRef.current = spritesMap
    blinkStarsRef.current = blinkStars

    // Background stars
    const bgCount = 3000
    const bgPos   = new Float32Array(bgCount * 3)
    for (let i = 0; i < bgCount; i++) {
      bgPos[i*3]   = (Math.random() - 0.5) * 5000
      bgPos[i*3+1] = (Math.random() - 0.5) * 5000
      bgPos[i*3+2] = (Math.random() - 0.5) * 5000
    }
    const bgGeo = new THREE.BufferGeometry()
    bgGeo.setAttribute('position', new THREE.BufferAttribute(bgPos, 3))
    const bgPoints = new THREE.Points(bgGeo, new THREE.PointsMaterial({
      size: 0.9, color: 0xaaccff, transparent: true, opacity: 0.45,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true,
    }))
    scene.add(bgPoints)
    bgStarsRef.current = bgPoints

    // Input
    function onKeyDown(e) {
      const k = e.key.toLowerCase()
      if (['w','s','a','d'].includes(k)) e.preventDefault()
      keysRef.current.add(k)
    }
    function onKeyUp(e) { keysRef.current.delete(e.key.toLowerCase()) }
    function onMouseDown(e) {
      if (!navRef.current.active) return
      navRef.current.dragging = true
      navRef.current.dragX = e.clientX
      navRef.current.dragY = e.clientY
      mount.style.cursor = 'grabbing'
    }
    function onMouseMove(e) {
      const nav = navRef.current
      if (!nav.active || !nav.dragging) return
      const dx = e.clientX - nav.dragX, dy = e.clientY - nav.dragY
      nav.yaw   -= dx * 0.0025
      nav.pitch -= dy * 0.0025
      nav.pitch  = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, nav.pitch))
      nav.dragX  = e.clientX; nav.dragY = e.clientY
    }
    function onMouseUp() {
      navRef.current.dragging = false
      if (navRef.current.active) mount.style.cursor = 'crosshair'
    }
    function onWheel(e) {
      e.preventDefault()
      if (!navRef.current.active) return
      navRef.current.scrollVel = Math.max(-25, Math.min(25, navRef.current.scrollVel + e.deltaY * -0.12))
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    mount.addEventListener('mousedown', onMouseDown)
    mount.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    mount.addEventListener('wheel', onWheel, { passive: false })

    function onResize() {
      const w = mount.clientWidth, h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
      composer.setSize(w, h)
      bloomPass.resolution.set(w, h)
    }
    window.addEventListener('resize', onResize)

    // Animation loop
    let rafId
    function animate() {
      rafId = requestAnimationFrame(animate)
      const elapsed = clockRef.current.getElapsedTime()
      const nav     = navRef.current
      const keys    = keysRef.current

      if (animatingRef.current) {
        // GSAP owns camera
      } else if (nav.active) {
        camera.rotation.set(nav.pitch, nav.yaw, 0, 'YXZ')
        const speed = 0.8
        const fwd   = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion)
        const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion)
        if (keys.has('w')) nav.pos.addScaledVector(fwd,    speed)
        if (keys.has('s')) nav.pos.addScaledVector(fwd,   -speed)
        if (keys.has('a')) nav.pos.addScaledVector(right, -speed)
        if (keys.has('d')) nav.pos.addScaledVector(right,  speed)
        nav.scrollVel *= 0.88
        if (Math.abs(nav.scrollVel) > 0.05) nav.pos.addScaledVector(fwd, nav.scrollVel)
        camera.position.copy(nav.pos)
      } else if (!arrangementLockedRef.current) {
        camera.position.set(
          Math.sin(elapsed * 0.038) * 130,
          Math.cos(elapsed * 0.026) * 65,
          280 + Math.sin(elapsed * 0.055) * 110,
        )
        camera.lookAt(
          Math.sin(elapsed * 0.022 + 1.1) * 55,
          Math.cos(elapsed * 0.017) * 35, 0,
        )
      }

      // Blink — opacity multiplied by opacityMul (GSAP tweens this during arrangement)
      blinkStarsRef.current.forEach(entry => {
        const { sprite, phase, period, lo, hi } = entry
        const mul = entry.opacityMul ?? 1
        const t = ((elapsed + phase) % period) / period
        sprite.material.opacity = (t < 0.70 ? hi : lo * 0.4) * mul
      })

      // Prospect: spin + scale snap
      if (prospectSpriteRef.current && (nav.active || arrangementLockedRef.current)) {
        prospectSpriteRef.current.material.rotation += 0.008
        const s = Math.floor(elapsed / 0.5) % 2 === 0 ? 26 : 32
        prospectSpriteRef.current.scale.set(s, s, 1)
      }

      if (prospectRingRef.current) {
        prospectRingRef.current.rotation.z += 0.004
        prospectRingRef.current.rotation.x = Math.sin(elapsed * 0.3) * 0.15
      }

      composer.render()
    }
    animate()

    sceneRef.current = { scene, camera, renderer }

    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      mount.removeEventListener('mousedown', onMouseDown)
      mount.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      mount.removeEventListener('wheel', onWheel)
      window.removeEventListener('resize', onResize)
      scene.traverse(obj => {
        obj.geometry?.dispose()
        if (obj.material)
          (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => m.dispose())
      })
      starTextures.forEach(t => t.dispose())
      composer.dispose()
      renderer.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [players])

  // ── Prospect + similarity ─────────────────────────────────────────────────
  useEffect(() => {
    if (!prospectData || !sceneRef.current) return
    if (animatingRef.current) return
    animatingRef.current = true

    const { scene, camera } = sceneRef.current
    const mount = mountRef.current

    fetch('http://localhost:8000/similarity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prospectData),
    })
      .then(r => r.json())
      .then(data => {
        const { prospect, comps } = data
        setCompData(comps)
        setProspectCoords(prospect)

        const { x: meanX, y: meanY, z: meanZ } = meanRef.current
        const ppx = (prospect.umap_x - meanX) * 100
        const ppy = (prospect.umap_y - meanY) * 100
        const ppz = (prospect.umap_z - meanZ) * 150
        const prospectPos = new THREE.Vector3(ppx, ppy, ppz)

        // Prospect 8-point gold star
        const prospectSprite = new THREE.Sprite(new THREE.SpriteMaterial({
          map: makeProspect8StarTexture(), transparent: true,
          depthWrite: false, blending: THREE.AdditiveBlending,
        }))
        prospectSprite.position.copy(prospectPos)
        prospectSprite.scale.set(26, 26, 1)
        scene.add(prospectSprite)
        prospectSpriteRef.current = prospectSprite

        // Gold particle ring
        const ringCount = 160
        const ringPos = new Float32Array(ringCount * 3)
        const ringCols = new Float32Array(ringCount * 3)
        for (let i = 0; i < ringCount; i++) {
          const angle = (i / ringCount) * Math.PI * 2 + (Math.random() - 0.5) * 0.4
          const r = 18 + Math.random() * 10
          ringPos[i*3] = Math.cos(angle) * r; ringPos[i*3+1] = Math.sin(angle) * r; ringPos[i*3+2] = (Math.random() - 0.5) * 4
          ringCols[i*3] = 1; ringCols[i*3+1] = 0.75 + Math.random() * 0.25; ringCols[i*3+2] = 0
        }
        const ringGeo = new THREE.BufferGeometry()
        ringGeo.setAttribute('position', new THREE.BufferAttribute(ringPos, 3))
        ringGeo.setAttribute('color',    new THREE.BufferAttribute(ringCols, 3))
        const ring = new THREE.Points(ringGeo, new THREE.PointsMaterial({
          size: 2.5, vertexColors: true, transparent: true,
          depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true, opacity: 0.85,
        }))
        ring.position.copy(prospectPos)
        scene.add(ring)
        prospectRingRef.current = ring

        // Fly-in curve
        const compSprites = comps.map(c => spritesMapRef.current.get(toKey(c.name))).filter(Boolean)
        const allPos      = [prospectPos, ...compSprites.map(s => s.position)]
        const centroid    = allPos.reduce((a, p) => a.add(p.clone()), new THREE.Vector3()).divideScalar(allPos.length)
        const camDir      = centroid.clone().sub(prospectPos)
        if (camDir.length() < 0.001) camDir.set(0, 0, 1)
        camDir.normalize()
        const maxSpread      = Math.max(60, ...allPos.map(p => p.distanceTo(centroid)))
        const finalCameraPos = centroid.clone().addScaledVector(camDir, maxSpread * 2.2 + 80)
        const finalTarget    = centroid.clone()

        const p0    = camera.position.clone()
        const p1    = new THREE.Vector3().lerpVectors(p0, centroid, 0.6).add(new THREE.Vector3(60, 40, 100))
        const p2    = centroid.clone().add(new THREE.Vector3(30, 20, 160))
        const curve = new THREE.CatmullRomCurve3([p0, p1, p2, finalCameraPos])
        const tObj  = { t: 0 }

        gsap.to(tObj, {
          t: 1, duration: 4.5, ease: 'power1.inOut',
          onUpdate:  () => { camera.position.copy(curve.getPoint(tObj.t)); camera.lookAt(finalTarget) },
          onComplete: () => {
            camera.position.copy(finalCameraPos)
            camera.lookAt(finalTarget)
            const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')
            const nav   = navRef.current
            nav.yaw = euler.y; nav.pitch = euler.x
            nav.pos.copy(finalCameraPos)
            nav.active = true
            if (mount) mount.style.cursor = 'crosshair'

            // Comp markers
            const compTex = makeCompArcadeStarTexture()
            compSprites.forEach(sprite => {
              sprite.material.map = compTex
              sprite.material.color.setHex(0xffffff)
              sprite.material.needsUpdate = true
              gsap.to(sprite.scale, { x: 18, y: 18, duration: 0.35, ease: 'power2.out' })
              const entry = blinkStarsRef.current.find(b => b.sprite === sprite)
              if (entry) { entry.period = 0.45; entry.lo = 0; entry.hi = 1; entry.phase = 0 }
            })
            compSpritesRef.current = compSprites
            animatingRef.current   = false

            // Snapshot for "continue to explore"
            savedNavRef.current = {
              pos:               nav.pos.clone(),
              yaw:               nav.yaw,
              pitch:             nav.pitch,
              compSprites,
              compOrigPositions: compSprites.map(s => s.position.clone()),
            }

            // ── Arrangement after 3 s ──────────────────────────────────
            setTimeout(() => {
              if (!sceneRef.current) return
              nav.active = false
              arrangementLockedRef.current = true

              // Fade all field stars (not comps) via opacityMul
              blinkStarsRef.current.forEach(entry => {
                if (!compSprites.includes(entry.sprite))
                  gsap.to(entry, { opacityMul: 0, duration: 1.0, ease: 'power2.out' })
              })
              if (bgStarsRef.current)
                gsap.to(bgStarsRef.current.material, { opacity: 0.04, duration: 1.0 })

              // New camera: D units from prospect along current forward
              const D   = 120
              const fwd = new THREE.Vector3()
              camera.getWorldDirection(fwd)
              const newCamPos = prospectPos.clone().sub(fwd.clone().multiplyScalar(D))
              const camUp     = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion)
              const camRight  = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion)
              const fovRad    = camera.fov * Math.PI / 180
              const halfH     = D * Math.tan(fovRad / 2)
              const halfW     = halfH * (mount.clientWidth / mount.clientHeight)

              const compTargets = [
                prospectPos.clone().addScaledVector(camUp,    halfH * 0.55),
                prospectPos.clone().addScaledVector(camRight, -halfW * 0.42).addScaledVector(camUp, -halfH * 0.38),
                prospectPos.clone().addScaledVector(camRight,  halfW * 0.42).addScaledVector(camUp, -halfH * 0.38),
              ]

              gsap.to(camera.position, {
                x: newCamPos.x, y: newCamPos.y, z: newCamPos.z,
                duration: 1.4, ease: 'power2.inOut',
                onUpdate:  () => camera.lookAt(prospectPos),
                onComplete: () => camera.lookAt(prospectPos),
              })

              compSprites.forEach((sprite, i) => {
                const t = compTargets[i]; if (!t) return
                gsap.to(sprite.position, { x: t.x, y: t.y, z: t.z, duration: 1.4, ease: 'power2.inOut', delay: 0.1 })
              })

              // Show overlay after animation
              gsap.delayedCall(1.8, () => {
                const w = mount.clientWidth, h = mount.clientHeight
                const labels = compSprites.map((sprite, i) => {
                  const target = compTargets[i] ?? sprite.position
                  const vec    = target.clone().project(camera)
                  return {
                    name:    toKey(comps[i]?.name ?? ''),
                    screenX: ((vec.x + 1) / 2) * w,
                    screenY: ((-vec.y + 1) / 2) * h,
                  }
                })
                setCompLabelData(labels)
                setProspectBubble({
                  position: prospectData?.position,
                  height:   prospectData?.height_inches,
                  weight:   prospectData?.weight,
                  age:      prospectData?.age_at_draft,
                  pts:      prospectData?.p36_pts,
                  reb:      prospectData?.p36_reb,
                  ast:      prospectData?.p36_ast,
                })
                setArrangementActive(true)
                requestAnimationFrame(() => setOverlayFadeIn(true))
              })
            }, 3000)
          },
        })
      })
      .catch(err => { console.error('Similarity fetch failed:', err); animatingRef.current = false })
  }, [prospectData])

  // ── Render ────────────────────────────────────────────────────────────────
  const FONT = '"Press Start 2P", monospace'

  return (
    <>
      <div
        ref={mountRef}
        style={{
          position: 'absolute', inset: 0,
          opacity: visible ? 1 : 0,
          transition: 'opacity 0.3s ease',
          pointerEvents: visible ? 'auto' : 'none',
        }}
      />

      {arrangementActive && visible && (
        <div style={{
          position: 'absolute', inset: 0, zIndex: 10,
          opacity: overlayFadeIn ? 1 : 0,
          transition: 'opacity 0.6s ease',
          pointerEvents: 'none',
        }}>

          {/* ── Comp name labels ── */}
          {compLabelData.map((label, i) => (
            <div key={i} style={{
              position: 'absolute',
              left: label.screenX,
              top:  label.screenY + 24,
              transform: 'translate(-50%, 0)',
              fontFamily: FONT,
              fontSize: '8px',
              color: '#ff7700',
              textShadow: '0 0 12px rgba(255,119,0,0.9), 2px 2px 0 #000',
              whiteSpace: 'nowrap',
              letterSpacing: '0.5px',
            }}>
              {label.name}
            </div>
          ))}

          {/* ── Prospect bubble ── */}
          {prospectBubble && (
            <div style={{
              position: 'absolute', left: '50%', top: '50%',
              transform: 'translate(-50%, 52px)',
              background: 'rgba(0, 4, 18, 0.96)',
              border: '2px solid #3a8fff',
              boxShadow: '4px 4px 0 #001040, 0 0 28px rgba(58,143,255,0.3)',
              padding: '16px 20px',
              minWidth: '240px',
              fontFamily: FONT,
              lineHeight: 2.2,
              userSelect: 'none',
            }}>
              <div style={{ color: '#ffdd00', fontSize: '10px', marginBottom: '14px', letterSpacing: '1px' }}>
                ★&nbsp;PROSPECT&nbsp;★
              </div>

              <div style={{ fontSize: '7px', color: '#7aadff' }}>
                {prospectBubble.position && <div>POS&nbsp;&nbsp;{prospectBubble.position}</div>}
                {prospectBubble.height   && <div>HT&nbsp;&nbsp;&nbsp;{fmtHt(prospectBubble.height)}</div>}
                {prospectBubble.weight   && <div>WT&nbsp;&nbsp;&nbsp;{prospectBubble.weight} lbs</div>}
                {prospectBubble.age      && <div>AGE&nbsp;&nbsp;{fmtStat(prospectBubble.age)}</div>}
              </div>

              {(prospectBubble.pts || prospectBubble.reb || prospectBubble.ast) && (
                <div style={{ borderTop: '1px solid #112244', marginTop: '10px', paddingTop: '10px', fontSize: '7px', color: '#aaccff' }}>
                  {prospectBubble.pts && <div>PTS/36&nbsp;&nbsp;{fmtStat(prospectBubble.pts)}</div>}
                  {prospectBubble.reb && <div>REB/36&nbsp;&nbsp;{fmtStat(prospectBubble.reb)}</div>}
                  {prospectBubble.ast && <div>AST/36&nbsp;&nbsp;{fmtStat(prospectBubble.ast)}</div>}
                </div>
              )}

              {/* Scouting report — streams from Gemini */}
              <div style={{ borderTop: '1px solid #112244', marginTop: '10px', paddingTop: '10px' }}>
                {scout.animal && (
                  <div style={{ color: '#ffdd00', fontSize: '9px', marginBottom: '6px', letterSpacing: '1px' }}>
                    {scout.animal}
                  </div>
                )}
                {scout.archetype && (
                  <div style={{ color: '#3a8fff', fontSize: '7px', marginBottom: '10px' }}>
                    {scout.archetype}
                  </div>
                )}
                {scout.writeup ? (
                  <div style={{ color: '#aaccff', fontSize: '6px', lineHeight: 2.0, maxWidth: '260px' }}>
                    {scout.writeup}
                  </div>
                ) : (
                  <div style={{ color: '#334466', fontSize: '6px', letterSpacing: '2px' }}>
                    SCOUTING REPORT
                    <div style={{ color: '#445577', marginTop: '6px' }}>
                      {scoutingLoading
                        ? '■ ■ ■'
                        : scoutingText
                          ? <span style={{ color: '#ff4444', fontSize: '5px', wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>{scoutingText}</span>
                          : 'SIGNAL LOST.'}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Continue to explore button ── */}
          <button
            onClick={handleContinueExplore}
            style={{
              position: 'absolute', right: '32px', top: '50%',
              transform: 'translateY(-50%)',
              pointerEvents: 'auto',
              background: 'rgba(0, 4, 18, 0.96)',
              border: '2px solid #3a8fff',
              boxShadow: '4px 4px 0 #001040, 0 0 16px rgba(58,143,255,0.25)',
              color: '#3a8fff',
              fontFamily: FONT,
              fontSize: '7px',
              padding: '18px 14px',
              cursor: 'pointer',
              letterSpacing: '1px',
              lineHeight: 2.4,
              writingMode: 'vertical-rl',
              textOrientation: 'mixed',
              transition: 'color 0.15s, box-shadow 0.15s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.color = '#ffffff'
              e.currentTarget.style.boxShadow = '4px 4px 0 #001040, 0 0 24px rgba(58,143,255,0.6)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.color = '#3a8fff'
              e.currentTarget.style.boxShadow = '4px 4px 0 #001040, 0 0 16px rgba(58,143,255,0.25)'
            }}
          >
            EXPLORE GALAXY ▶
          </button>

        </div>
      )}
    </>
  )
}
