import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import gsap from 'gsap'

function makeGlowTexture() {
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  grad.addColorStop(0, 'rgba(180, 210, 255, 1)')
  grad.addColorStop(0.3, 'rgba(150, 180, 255, 0.6)')
  grad.addColorStop(1, 'rgba(100, 140, 255, 0)')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, size, size)
  return new THREE.CanvasTexture(canvas)
}

function makeStarTexture() {
  const size = 16
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  grad.addColorStop(0, 'rgba(255, 255, 255, 1)')
  grad.addColorStop(1, 'rgba(255, 255, 255, 0)')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, size, size)
  return new THREE.CanvasTexture(canvas)
}

const toKey = (raw) => {
  const noSchool = raw.includes('|') ? raw.split('|')[0].trim() : raw.trim()
  const noPosition = noSchool.replace(/\s+[A-Z]{1,2}(\/[A-Z]{1,2})*$/, '').trim()
  return noPosition.normalize('NFD').replace(/[̀-ͯ]/g, '')
}

export default function Universe({ players, visible, prospectData }) {
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const meanRef = useRef({ x: 0, y: 0, z: 0 })
  const spritesMapRef = useRef(new Map())
  const animatingRef = useRef(false)
  const glowTexRef = useRef(null)
  const [compData, setCompData] = useState(null)
  const [prospectCoords, setProspectCoords] = useState(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount || !players?.length) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x000000)

    const camera = new THREE.PerspectiveCamera(60, mount.clientWidth / mount.clientHeight, 0.1, 10000)
    camera.position.set(0, 0, 1800)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 0, 0)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.update()

    const glowTex = makeGlowTexture()
    glowTexRef.current = glowTex

    const spriteMaterial = new THREE.SpriteMaterial({
      map: glowTex,
      color: 0xaaccff,
      transparent: true,
      depthWrite: false,
    })

    const xs = players.map(p => p.umap_x)
    const ys = players.map(p => p.umap_y)
    const zs = players.map(p => p.umap_z)
    const meanX = xs.reduce((a, b) => a + b, 0) / xs.length
    const meanY = ys.reduce((a, b) => a + b, 0) / ys.length
    const meanZ = zs.reduce((a, b) => a + b, 0) / zs.length

    meanRef.current = { x: meanX, y: meanY, z: meanZ }

    const spritesMap = new Map()
    players.forEach(player => {
      const sprite = new THREE.Sprite(spriteMaterial.clone())
      sprite.position.set(
        (player.umap_x - meanX) * 100,
        (player.umap_y - meanY) * 100,
        (player.umap_z - meanZ) * 100,
      )
      sprite.scale.set(10, 10, 1)
      sprite.userData = { name: player.name }
      scene.add(sprite)
      spritesMap.set(toKey(player.name), sprite)
    })
    spritesMapRef.current = spritesMap
    console.log('Sprite map keys (first 5):', [...spritesMap.keys()].slice(0, 5))

    const starCount = 4000
    const starPositions = new Float32Array(starCount * 3)
    for (let i = 0; i < starCount; i++) {
      starPositions[i * 3]     = (Math.random() - 0.5) * 4000
      starPositions[i * 3 + 1] = (Math.random() - 0.5) * 4000
      starPositions[i * 3 + 2] = (Math.random() - 0.5) * 4000
    }
    const starGeo = new THREE.BufferGeometry()
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3))
    const starMat = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 1.5,
      map: makeStarTexture(),
      transparent: true,
      depthWrite: false,
      sizeAttenuation: true,
    })
    scene.add(new THREE.Points(starGeo, starMat))

    function onResize() {
      const w = mount.clientWidth
      const h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    let rafId
    function animate() {
      rafId = requestAnimationFrame(animate)
      if (!animatingRef.current) controls.update()
      renderer.render(scene, camera)
    }
    animate()

    sceneRef.current = { scene, camera, renderer, controls }

    return () => {
      cancelAnimationFrame(rafId)
      window.removeEventListener('resize', onResize)
      controls.dispose()
      renderer.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [players])

  useEffect(() => {
    if (!prospectData || !sceneRef.current) return
    if (animatingRef.current) return
    animatingRef.current = true

    const { scene, camera, controls } = sceneRef.current
    controls.enabled = false

    fetch('http://localhost:8000/similarity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prospectData),
    })
      .then(r => r.json())
      .then(data => {
        const { prospect, comps } = data

        console.log('Comp names from API:', comps.map(c => c.name))
        comps.forEach(c => console.log(`  ${c.name}: distance ${c.distance?.toFixed(4)}`))

        setCompData(comps)
        setProspectCoords(prospect)

        const { x: meanX, y: meanY, z: meanZ } = meanRef.current
        const px = (prospect.umap_x - meanX) * 100
        const py = (prospect.umap_y - meanY) * 100
        const pz = (prospect.umap_z - meanZ) * 100
        const prospectPos = new THREE.Vector3(px, py, pz)

        const prospectSprite = new THREE.Sprite(new THREE.SpriteMaterial({
          map: glowTexRef.current,
          color: 0xffd700,
          transparent: true,
          depthWrite: false,
        }))
        prospectSprite.position.copy(prospectPos)
        prospectSprite.scale.set(16, 16, 1)
        scene.add(prospectSprite)

        // STEP 1 — compute all positions before animation starts
        comps.forEach(c => {
          const key = toKey(c.name)
          const found = spritesMapRef.current.get(key)
          console.log(`Lookup: '${c.name}' → key='${key}' → found=${!!found}`)
        })
        const compSprites = comps.map(c => spritesMapRef.current.get(toKey(c.name))).filter(Boolean)
        const allPositions = [prospectPos, ...compSprites.map(s => s.position)]
        const centroid = allPositions
          .reduce((acc, p) => acc.add(p), new THREE.Vector3())
          .divideScalar(allPositions.length)
        const finalCameraPos = centroid.clone().add(new THREE.Vector3(0, 0, 120))
        const finalTarget = centroid.clone()

        // STEP 2 — build curve with finalCameraPos as last point
        const p0 = camera.position.clone()
        const p1 = new THREE.Vector3().lerpVectors(p0, centroid, 0.6)
          .add(new THREE.Vector3(60, 40, 100))
        const p2 = centroid.clone().add(new THREE.Vector3(30, 20, 160))
        const p3 = finalCameraPos

        const curve = new THREE.CatmullRomCurve3([p0, p1, p2, p3])

        const tObj = { t: 0 }
        gsap.to(tObj, {
          t: 1,
          duration: 4.5,
          ease: 'power1.inOut',
          onUpdate: () => {
            camera.position.copy(curve.getPoint(tObj.t))
            camera.lookAt(finalTarget)
          },
          // STEP 3 — no camera.position touch, camera naturally ends at p3=finalCameraPos
          onComplete: () => {
            controls.target.copy(finalTarget)
            requestAnimationFrame(() => { controls.enabled = true })

            compSprites.forEach(sprite => {
              sprite.material.color.setHex(0xffcc88)
              sprite.material.opacity = 1.0
              gsap.to(sprite.scale, { x: 14, y: 14, duration: 0.3, ease: 'power2.out' })
            })

            animatingRef.current = false
          },
        })
      })
      .catch(err => {
        console.error('Similarity fetch failed:', err)
        controls.enabled = true
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
