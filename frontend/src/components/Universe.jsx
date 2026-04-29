import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import gsap from 'gsap'

// ─── GLSL shaders ─────────────────────────────────────────────────────────────

const VERT = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const FRAG_BASE = /* glsl */`
uniform float uTime;
varying vec2 vUv;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
    f.y
  );
}
float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * noise(p);
    p = p * 2.1 + vec2(100.0);
    a *= 0.5;
  }
  return v;
}
`

// Gold / amber corona — prospect
const FRAG_PROSPECT = FRAG_BASE + /* glsl */`
void main() {
  vec2 uv = vUv * 2.0 - 1.0;
  float dist = length(uv);
  if (dist > 1.0) discard;

  float angle = atan(uv.y, uv.x);
  float turb  = fbm(vec2(dist * 3.5 + uTime * 0.18, angle / 3.14159 * 3.0 + uTime * 0.12));

  float modDist = clamp(dist + (turb - 0.5) * 0.42, 0.0, 1.5);

  float corona = pow(max(0.0, 1.0 - modDist * 1.25), 2.2);
  float core   = pow(max(0.0, 1.0 - dist * 8.0), 3.0);

  float t = clamp(dist + (turb - 0.5) * 0.3, 0.0, 1.0);
  vec3 col = mix(vec3(1.0, 1.0, 0.94), vec3(1.0, 0.75, 0.08), smoothstep(0.0, 0.45, t));
  col       = mix(col, vec3(0.9, 0.22, 0.0), smoothstep(0.35, 0.88, t));

  float alpha = (corona * 0.92 + core) * smoothstep(1.0, 0.72, dist);
  gl_FragColor = vec4(col, clamp(alpha, 0.0, 1.0));
}
`

// Cyan / teal corona — comps
const FRAG_COMP = FRAG_BASE + /* glsl */`
void main() {
  vec2 uv = vUv * 2.0 - 1.0;
  float dist = length(uv);
  if (dist > 1.0) discard;

  float angle = atan(uv.y, uv.x);
  float turb  = fbm(vec2(dist * 3.5 + uTime * 0.15, angle / 3.14159 * 3.0 + uTime * 0.10));

  float modDist = clamp(dist + (turb - 0.5) * 0.38, 0.0, 1.5);

  float corona = pow(max(0.0, 1.0 - modDist * 1.3), 2.2);
  float core   = pow(max(0.0, 1.0 - dist * 8.0), 3.0);

  float t = clamp(dist + (turb - 0.5) * 0.28, 0.0, 1.0);
  vec3 col = mix(vec3(0.9, 1.0, 1.0), vec3(0.0, 1.0, 0.85), smoothstep(0.0, 0.45, t));
  col       = mix(col, vec3(0.0, 0.55, 0.52), smoothstep(0.35, 0.88, t));

  float alpha = (corona * 0.72 + core * 0.85) * smoothstep(1.0, 0.72, dist);
  gl_FragColor = vec4(col, clamp(alpha, 0.0, 1.0));
}
`

// ─── Texture factories ────────────────────────────────────────────────────────

function makePlayerStarTexture(variant) {
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const cx = size / 2, cy = size / 2

  const cores  = [
    'rgba(220,235,255,1)',
    'rgba(242,246,255,1)',
    'rgba(255,250,235,1)',
    'rgba(255,235,190,1)',
    'rgba(185,215,255,1)',
  ]
  const mids   = [
    'rgba(160,200,255,0.45)',
    'rgba(200,215,255,0.40)',
    'rgba(255,225,185,0.40)',
    'rgba(255,200,120,0.40)',
    'rgba(140,190,255,0.45)',
  ]
  const spikes = ['200,220,255','210,225,255','255,235,200','255,210,150','170,205,255']
  const v = variant % 5

  const corona = ctx.createRadialGradient(cx, cy, 0, cx, cy, cx)
  corona.addColorStop(0,    cores[v])
  corona.addColorStop(0.08, cores[v])
  corona.addColorStop(0.28, mids[v])
  corona.addColorStop(1,    'rgba(0,0,0,0)')
  ctx.fillStyle = corona
  ctx.fillRect(0, 0, size, size)

  const spikeLen = cx * 0.88
  ;[
    [cx - spikeLen, cy, cx + spikeLen, cy],
    [cx, cy - spikeLen, cx, cy + spikeLen],
  ].forEach(([x1, y1, x2, y2]) => {
    const g = ctx.createLinearGradient(x1, y1, x2, y2)
    g.addColorStop(0,    `rgba(${spikes[v]},0)`)
    g.addColorStop(0.38, `rgba(${spikes[v]},0.18)`)
    g.addColorStop(0.5,  `rgba(${spikes[v]},0.28)`)
    g.addColorStop(0.62, `rgba(${spikes[v]},0.18)`)
    g.addColorStop(1,    `rgba(${spikes[v]},0)`)
    ctx.strokeStyle = g
    ctx.lineWidth = 1.2
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()
  })

  const coreR = size * 0.055
  const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR * 2.5)
  coreGrad.addColorStop(0,    'rgba(255,255,255,1)')
  coreGrad.addColorStop(0.35, cores[v])
  coreGrad.addColorStop(1,    'rgba(0,0,0,0)')
  ctx.fillStyle = coreGrad
  ctx.beginPath()
  ctx.arc(cx, cy, coreR * 2.5, 0, Math.PI * 2)
  ctx.fill()

  return new THREE.CanvasTexture(canvas)
}

function makeProspectCoreTexture() {
  const size = 256
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const cx = size / 2, cy = size / 2

  const corona = ctx.createRadialGradient(cx, cy, 0, cx, cy, cx)
  corona.addColorStop(0,    'rgba(255,248,140,1)')
  corona.addColorStop(0.12, 'rgba(255,220,60,0.9)')
  corona.addColorStop(0.35, 'rgba(255,165,20,0.4)')
  corona.addColorStop(0.65, 'rgba(255,100,5,0.12)')
  corona.addColorStop(1,    'rgba(0,0,0,0)')
  ctx.fillStyle = corona
  ctx.fillRect(0, 0, size, size)

  const spikeLen = cx * 0.92
  ctx.lineWidth = 1.8
  ;[0, Math.PI/6, Math.PI/3, Math.PI/2, 2*Math.PI/3, 5*Math.PI/6].forEach(angle => {
    const x1 = cx + Math.cos(angle + Math.PI) * spikeLen
    const y1 = cy + Math.sin(angle + Math.PI) * spikeLen
    const x2 = cx + Math.cos(angle) * spikeLen
    const y2 = cy + Math.sin(angle) * spikeLen
    const g = ctx.createLinearGradient(x1, y1, x2, y2)
    g.addColorStop(0,    'rgba(255,215,60,0)')
    g.addColorStop(0.38, 'rgba(255,240,120,0.40)')
    g.addColorStop(0.5,  'rgba(255,255,200,0.60)')
    g.addColorStop(0.62, 'rgba(255,240,120,0.40)')
    g.addColorStop(1,    'rgba(255,215,60,0)')
    ctx.strokeStyle = g
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()
  })

  const coreR = size * 0.07
  const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR * 3)
  coreGrad.addColorStop(0,    'rgba(255,255,248,1)')
  coreGrad.addColorStop(0.25, 'rgba(255,250,180,0.95)')
  coreGrad.addColorStop(0.6,  'rgba(255,220,60,0.5)')
  coreGrad.addColorStop(1,    'rgba(0,0,0,0)')
  ctx.fillStyle = coreGrad
  ctx.beginPath()
  ctx.arc(cx, cy, coreR * 3, 0, Math.PI * 2)
  ctx.fill()

  return new THREE.CanvasTexture(canvas)
}

function makeNebulaTexture() {
  const size = 1024
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')

  ctx.fillStyle = '#000000'
  ctx.fillRect(0, 0, size, size)

  ;[
    { x: 0.22, y: 0.30, r: 0.38, rgb: '26,10,58',  a: 0.05 },
    { x: 0.72, y: 0.20, r: 0.30, rgb: '4,26,20',   a: 0.04 },
    { x: 0.50, y: 0.72, r: 0.42, rgb: '4,13,42',   a: 0.06 },
    { x: 0.85, y: 0.58, r: 0.26, rgb: '30,8,52',   a: 0.04 },
    { x: 0.10, y: 0.82, r: 0.32, rgb: '6,22,44',   a: 0.04 },
    { x: 0.58, y: 0.40, r: 0.22, rgb: '18,5,38',   a: 0.03 },
    { x: 0.32, y: 0.62, r: 0.28, rgb: '5,28,18',   a: 0.03 },
    { x: 0.78, y: 0.88, r: 0.24, rgb: '22,7,48',   a: 0.04 },
  ].forEach(({ x, y, r, rgb, a }) => {
    const gx = x * size, gy = y * size, radius = r * size
    const grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, radius)
    grad.addColorStop(0,   `rgba(${rgb},${a})`)
    grad.addColorStop(0.5, `rgba(${rgb},${a * 0.4})`)
    grad.addColorStop(1,   'rgba(0,0,0,0)')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, size, size)
  })

  return new THREE.CanvasTexture(canvas)
}

// ─── Name key ─────────────────────────────────────────────────────────────────

const toKey = (raw) => {
  const noSchool = raw.includes('|') ? raw.split('|')[0].trim() : raw.trim()
  const noPosition = noSchool.replace(/\s+[A-Z]{1,2}(\/[A-Z]{1,2})*$/, '').trim()
  return noPosition.normalize('NFD').replace(/[̀-ͯ]/g, '')
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Universe({ players, visible, prospectData }) {
  const mountRef          = useRef(null)
  const sceneRef          = useRef(null)
  const meanRef           = useRef({ x: 0, y: 0, z: 0 })
  const spritesMapRef     = useRef(new Map())
  const animatingRef      = useRef(false)
  const composerRef       = useRef(null)
  const prospectRingRef   = useRef(null)
  const prospectCoronaRef = useRef(null)
  const compCoronasRef    = useRef([])
  const coronaShadersRef  = useRef([])
  const clockRef          = useRef(new THREE.Clock())

  const keysRef = useRef(new Set())
  const navRef  = useRef({
    active:    false,
    yaw:       0,
    pitch:     0,
    pos:       new THREE.Vector3(0, 0, 300),
    dragging:  false,
    dragX:     0,
    dragY:     0,
    scrollVel: 0,
  })

  const [compData, setCompData]             = useState(null)
  const [prospectCoords, setProspectCoords] = useState(null)

  // ── Scene setup ──────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current
    if (!mount || !players?.length) return

    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x000000, 0.00022)

    const nebulaTex = makeNebulaTexture()
    scene.add(new THREE.Mesh(
      new THREE.SphereGeometry(4000, 32, 32),
      new THREE.MeshBasicMaterial({ map: nebulaTex, side: THREE.BackSide, depthWrite: false, fog: false }),
    ))

    const camera = new THREE.PerspectiveCamera(70, mount.clientWidth / mount.clientHeight, 0.1, 10000)
    camera.position.set(0, 0, 300)
    camera.lookAt(0, 0, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 0.9
    mount.appendChild(renderer.domElement)

    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(mount.clientWidth, mount.clientHeight),
      0.9, 0.6, 0.0,
    )
    composer.addPass(bloomPass)
    composer.addPass(new OutputPass())
    composerRef.current = composer

    const starTextures = Array.from({ length: 5 }, (_, i) => makePlayerStarTexture(i))

    const xs = players.map(p => p.umap_x)
    const ys = players.map(p => p.umap_y)
    const zs = players.map(p => p.umap_z)
    const meanX = xs.reduce((a, b) => a + b, 0) / xs.length
    const meanY = ys.reduce((a, b) => a + b, 0) / ys.length
    const meanZ = zs.reduce((a, b) => a + b, 0) / zs.length
    meanRef.current = { x: meanX, y: meanY, z: meanZ }

    const spritesMap = new Map()
    players.forEach(player => {
      const variant = Math.floor(Math.random() * 5)
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: starTextures[variant],
        color: 0xffffff,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }))
      sprite.position.set(
        (player.umap_x - meanX) * 100,
        (player.umap_y - meanY) * 100,
        (player.umap_z - meanZ) * 150,
      )
      const scale = Math.random() < 0.8 ? 8 + Math.random() * 5 : 13 + Math.random() * 4
      sprite.scale.set(scale, scale, 1)
      sprite.userData = { name: player.name }
      scene.add(sprite)
      spritesMap.set(toKey(player.name), sprite)
    })
    spritesMapRef.current = spritesMap

    const starCount = 4000
    const starPos  = new Float32Array(starCount * 3)
    const starCols = new Float32Array(starCount * 3)
    const ambientPalette = [
      [0.45, 0.45, 0.45],
      [0.35, 0.40, 0.50],
      [0.45, 0.43, 0.38],
      [0.30, 0.38, 0.50],
    ]
    for (let i = 0; i < starCount; i++) {
      starPos[i*3]   = (Math.random() - 0.5) * 4000
      starPos[i*3+1] = (Math.random() - 0.5) * 4000
      starPos[i*3+2] = (Math.random() - 0.5) * 4000
      const c = ambientPalette[Math.floor(Math.random() * ambientPalette.length)]
      starCols[i*3] = c[0]; starCols[i*3+1] = c[1]; starCols[i*3+2] = c[2]
    }
    const starGeo = new THREE.BufferGeometry()
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3))
    starGeo.setAttribute('color',    new THREE.BufferAttribute(starCols, 3))
    scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({
      size: 1.2,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    })))

    // ── Input handlers ────────────────────────────────────────────────────
    function onKeyDown(e) {
      const k = e.key.toLowerCase()
      if (['w','s','a','d'].includes(k)) e.preventDefault()
      keysRef.current.add(k)
    }
    function onKeyUp(e) {
      keysRef.current.delete(e.key.toLowerCase())
    }
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
      const dx = e.clientX - nav.dragX
      const dy = e.clientY - nav.dragY
      nav.yaw   -= dx * 0.0025
      nav.pitch -= dy * 0.0025
      nav.pitch  = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, nav.pitch))
      nav.dragX  = e.clientX
      nav.dragY  = e.clientY
    }
    function onMouseUp() {
      navRef.current.dragging = false
      if (navRef.current.active) mount.style.cursor = 'crosshair'
    }
    function onWheel(e) {
      e.preventDefault()
      if (!navRef.current.active) return
      navRef.current.scrollVel = Math.max(
        -25, Math.min(25, navRef.current.scrollVel + e.deltaY * -0.12),
      )
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

    // ── Animation loop ────────────────────────────────────────────────────
    let rafId
    function animate() {
      rafId = requestAnimationFrame(animate)

      const elapsed = clockRef.current.getElapsedTime()
      const nav     = navRef.current
      const keys    = keysRef.current

      if (animatingRef.current) {
        // GSAP owns the camera
      } else if (nav.active) {
        // ── WASD + mouse-drag navigation ───────────────────────────────
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
      } else {
        // ── Idle cinematic drift ───────────────────────────────────────
        camera.position.set(
          Math.sin(elapsed * 0.038) * 130,
          Math.cos(elapsed * 0.026) * 65,
          280 + Math.sin(elapsed * 0.055) * 110,
        )
        camera.lookAt(
          Math.sin(elapsed * 0.022 + 1.1) * 55,
          Math.cos(elapsed * 0.017) * 35,
          0,
        )
      }

      // Rotate prospect particle ring
      if (prospectRingRef.current) {
        prospectRingRef.current.rotation.z += 0.004
        prospectRingRef.current.rotation.x = Math.sin(elapsed * 0.3) * 0.15
      }

      // Billboard corona planes to camera
      if (prospectCoronaRef.current) {
        prospectCoronaRef.current.quaternion.copy(camera.quaternion)
      }
      compCoronasRef.current.forEach(c => c.quaternion.copy(camera.quaternion))

      // Advance all corona shader clocks
      coronaShadersRef.current.forEach(mat => {
        mat.uniforms.uTime.value = elapsed
      })

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
        if (obj.material) {
          ;(Array.isArray(obj.material) ? obj.material : [obj.material]).forEach(m => m.dispose())
        }
      })
      starTextures.forEach(t => t.dispose())
      nebulaTex.dispose()
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
        const px = (prospect.umap_x - meanX) * 100
        const py = (prospect.umap_y - meanY) * 100
        const pz = (prospect.umap_z - meanZ) * 150
        const prospectPos = new THREE.Vector3(px, py, pz)

        // ── Prospect core sprite ──────────────────────────────────────
        const prospectCoreTex = makeProspectCoreTexture()
        const prospectSprite = new THREE.Sprite(new THREE.SpriteMaterial({
          map: prospectCoreTex, transparent: true,
          depthWrite: false, blending: THREE.AdditiveBlending,
        }))
        prospectSprite.position.copy(prospectPos)
        prospectSprite.scale.set(20, 20, 1)
        scene.add(prospectSprite)

        // ── Prospect GLSL corona ──────────────────────────────────────
        const prospectCoronaMat = new THREE.ShaderMaterial({
          vertexShader: VERT,
          fragmentShader: FRAG_PROSPECT,
          uniforms: { uTime: { value: 0 } },
          transparent: true,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        })
        const prospectCorona = new THREE.Mesh(
          new THREE.PlaneGeometry(70, 70),
          prospectCoronaMat,
        )
        prospectCorona.position.copy(prospectPos)
        scene.add(prospectCorona)
        prospectCoronaRef.current = prospectCorona
        coronaShadersRef.current.push(prospectCoronaMat)

        // ── Prospect particle ring ────────────────────────────────────
        const ringCount = 160
        const ringPos  = new Float32Array(ringCount * 3)
        const ringCols = new Float32Array(ringCount * 3)
        for (let i = 0; i < ringCount; i++) {
          const angle = (i / ringCount) * Math.PI * 2 + (Math.random() - 0.5) * 0.4
          const r = 16 + Math.random() * 10
          ringPos[i*3]   = Math.cos(angle) * r
          ringPos[i*3+1] = Math.sin(angle) * r
          ringPos[i*3+2] = (Math.random() - 0.5) * 5
          ringCols[i*3]   = 1.0
          ringCols[i*3+1] = 0.4 + Math.random() * 0.45
          ringCols[i*3+2] = Math.random() * 0.1
        }
        const ringGeo = new THREE.BufferGeometry()
        ringGeo.setAttribute('position', new THREE.BufferAttribute(ringPos, 3))
        ringGeo.setAttribute('color',    new THREE.BufferAttribute(ringCols, 3))
        const ring = new THREE.Points(ringGeo, new THREE.PointsMaterial({
          size: 3, vertexColors: true, transparent: true,
          depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true, opacity: 0.9,
        }))
        ring.position.copy(prospectPos)
        scene.add(ring)
        prospectRingRef.current = ring

        // ── Camera fly-in setup ───────────────────────────────────────
        const compSprites = comps
          .map(c => spritesMapRef.current.get(toKey(c.name)))
          .filter(Boolean)

        const allPositions = [prospectPos, ...compSprites.map(s => s.position)]
        const centroid = allPositions
          .reduce((acc, p) => acc.add(p.clone()), new THREE.Vector3())
          .divideScalar(allPositions.length)
        const camDir = centroid.clone().sub(prospectPos)
        if (camDir.length() < 0.001) camDir.set(0, 0, 1)
        camDir.normalize()
        const maxSpread  = Math.max(60, ...allPositions.map(p => p.distanceTo(centroid)))
        const finalCameraPos = centroid.clone().addScaledVector(camDir, maxSpread * 2.2 + 80)
        const finalTarget    = centroid.clone()

        const p0 = camera.position.clone()
        const p1 = new THREE.Vector3().lerpVectors(p0, centroid, 0.6).add(new THREE.Vector3(60, 40, 100))
        const p2 = centroid.clone().add(new THREE.Vector3(30, 20, 160))
        const curve = new THREE.CatmullRomCurve3([p0, p1, p2, finalCameraPos])

        const tObj = { t: 0 }
        gsap.to(tObj, {
          t: 1,
          duration: 4.5,
          ease: 'power1.inOut',
          onUpdate: () => {
            camera.position.copy(curve.getPoint(tObj.t))
            camera.lookAt(finalTarget)
          },
          onComplete: () => {
            camera.position.copy(finalCameraPos)
            camera.lookAt(finalTarget)
            const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')
            const nav   = navRef.current
            nav.yaw     = euler.y
            nav.pitch   = euler.x
            nav.pos.copy(finalCameraPos)
            nav.active  = true
            if (mount) mount.style.cursor = 'crosshair'

            // ── Comp markers ──────────────────────────────────────────
            compSprites.forEach((sprite, i) => {
              sprite.material.color.setHex(0x00ffcc)
              gsap.to(sprite.scale, { x: 16, y: 16, duration: 0.4, ease: 'power2.out' })

              const compCoronaMat = new THREE.ShaderMaterial({
                vertexShader: VERT,
                fragmentShader: FRAG_COMP,
                uniforms: { uTime: { value: 0 } },
                transparent: true,
                depthWrite: false,
                blending: THREE.AdditiveBlending,
              })
              const compCorona = new THREE.Mesh(
                new THREE.PlaneGeometry(44, 44),
                compCoronaMat,
              )
              compCorona.position.copy(sprite.position)
              scene.add(compCorona)
              compCoronasRef.current.push(compCorona)
              coronaShadersRef.current.push(compCoronaMat)
            })

            animatingRef.current = false
          },
        })
      })
      .catch(err => {
        console.error('Similarity fetch failed:', err)
        animatingRef.current = false
      })
  }, [prospectData])

  return (
    <div
      ref={mountRef}
      style={{
        position: 'absolute',
        inset: 0,
        opacity: visible ? 1 : 0,
        transition: 'opacity 0.3s ease',
        pointerEvents: visible ? 'auto' : 'none',
      }}
    />
  )
}
