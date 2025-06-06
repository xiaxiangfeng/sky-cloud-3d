# threejs-sky-cloud

[![npm version](https://img.shields.io/npm/v/threejs-sky-cloud.svg)](https://www.npmjs.com/package/threejs-sky-cloud)

![SkyCloudMesh Example](./example.png)

## Example
See the [example](https://xiaxiangfeng.github.io/sky-cloud-3d/threejs-sky-clouds.html) for a full integration with Three.js and GUI controls.

A physically-based sky and volumetric cloud Mesh/Material for [Three.js](https://threejs.org/), suitable for realistic outdoor scenes, games, and simulations. This library provides a ready-to-use, highly customizable sky and cloud dome mesh and material.

## Features
- Realistic sky gradient and sun disk
- Volumetric, animated clouds with wind and lighting
- Fully configurable via parameters (cloud density, height, thickness, wind, etc.)
- One-line integration as a Mesh (SkyCloudMesh)
- TypeScript compatible (ESM)

## Usage

```js
import * as THREE from 'three';
import { SkyCloudMesh } from './dist/SkyCloudMesh.min.js';

// 1. Create the sky and cloud Mesh
const skyMesh = new SkyCloudMesh({
  cloudCoverage: 0.7,
  cloudHeight: 600.0,
  cloudThickness: 45.0,
  windSpeedX: 5.0,
  windSpeedZ: 3.0,
  maxCloudDistance: 10000.0,
  perlinTextureUrl: './perlin256.png', // Required for clouds
  radius: 20000,
  widthSegments: 64,
  heightSegments: 32
});
scene.add(skyMesh);

// 2. In your animation loop, update sun direction and time
function animate() {
  requestAnimationFrame(animate);
  const elapsedTime = performance.now() * 0.001;
  // Calculate sunDirection (THREE.Vector3) as needed
  skyMesh.updateSun(sunDirection);
  skyMesh.updateTime(elapsedTime);
  renderer.render(scene, camera);
}
animate();
```

## API

### `SkyCloudMesh(params)`
A ready-to-use Mesh for sky and clouds. All parameters are optional:
- `cloudCoverage` (number, default: 0.65)
- `cloudHeight` (number, default: 600.0)
- `cloudThickness` (number, default: 45.0)
- `cloudAbsorption` (number, default: 1.03)
- `windSpeedX` (number, default: 5.0)
- `windSpeedZ` (number, default: 3.0)
- `maxCloudDistance` (number, default: 10000.0)
- `perlinTextureUrl` (string, required for clouds)
- `radius` (number, default: 20000)
- `widthSegments` (number, default: 64)
- `heightSegments` (number, default: 32)

#### Methods
- `updateSun(sunDirection: THREE.Vector3)` — Update the sun direction for lighting.
- `updateTime(elapsedTime: number)` — Update the animation time.

### Advanced: Material API
If you want to use the material directly:
- `createSkyMaterial(params)` — Returns a ShaderMaterial.
- `setPerlinNoiseTexture(material, texture)` — Assign a Perlin noise texture.
- `createSkyMaterialWithTexture(params)` — Returns a Promise<ShaderMaterial> with texture auto-loaded.

## Perlin Noise Texture
You must provide a seamless Perlin noise texture (e.g. `perlin256.png`). You can generate one or use a public domain texture. The texture should be grayscale, 256x256 or larger, and set to `THREE.RepeatWrapping`.

## License

This project is licensed for non-commercial use only. See the LICENSE file for details.
