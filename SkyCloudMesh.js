import {
    BackSide,
    ClampToEdgeWrapping,
    Data3DTexture,
    DataTexture,
    LinearFilter,
    Mesh,
    NodeMaterial,
    NoColorSpace,
    RedFormat,
    RepeatWrapping,
    RGBAFormat,
    SphereGeometry,
    TextureLoader,
    UnsignedByteType,
    Vector2,
    Vector3,
} from 'three/webgpu';

import {
    Break,
    Fn,
    If,
    Loop,
    abs,
    cameraPosition,
    clamp,
    dot,
    exp,
    fract as tslFract,
    float,
    mat3,
    max,
    min,
    mix,
    normalize,
    positionWorld,
    pow,
    sin as tslSin,
    smoothstep,
    texture,
    texture3D,
    uniform,
    vec3,
    vec4,
} from 'three/tsl';

const FALLBACK_NOISE_SIZE = 256;
const FALLBACK_VOLUME_NOISE_SIZE = 64;
const FALLBACK_OCTAVES = 5;
const FALLBACK_BASE_FREQUENCY = 4;
const FALLBACK_PERSISTENCE = 0.55;
const HASH_SCALE = 43758.5453123;
const TWO_PI = Math.PI * 2.0;
const BASE_CLOUD_STEPS = 22;
const MAX_CLOUD_STEPS = 56;
const LIGHT_STEPS = 5;

const DEFAULT_PARAMS = {
    cloudAbsorption: 1.03,
    cloudCoverage: 0.65,
    cloudHeight: 600.0,
    cloudThickness: 45.0,
    maxCloudDistance: 10000.0,
    windSpeedX: 5.0,
    windSpeedZ: 3.0,
};

const PARAM_TO_NODE = {
    cloudAbsorption: 'uCloudAbsorption',
    cloudCoverage: 'uCloudCoverage',
    cloudHeight: 'uCloudHeight',
    cloudThickness: 'uCloudThickness',
    maxCloudDistance: 'uMaxCloudDistance',
    windSpeedX: 'uWindSpeedX',
    windSpeedZ: 'uWindSpeedZ',
};

function fract(value) {
    return value - Math.floor(value);
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function smoothstepScalar(edge0, edge1, value) {
    const t = Math.max(0, Math.min(1, (value - edge0) / Math.max(edge1 - edge0, 1e-6)));
    return t * t * (3 - 2 * t);
}

function fade(t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

function clampByte(value) {
    return Math.max(0, Math.min(255, Math.round(value)));
}

function hash2D(x, y, seed) {
    return fract(Math.sin(x * 127.1 + y * 311.7 + seed * 74.7) * HASH_SCALE);
}

function createGradientGrid(period, seed) {
    const gradients = new Float32Array(period * period * 2);

    for (let y = 0; y < period; y++) {
        for (let x = 0; x < period; x++) {
            const angle = TWO_PI * hash2D(x, y, seed);
            const index = (y * period + x) * 2;
            gradients[index] = Math.cos(angle);
            gradients[index + 1] = Math.sin(angle);
        }
    }

    return gradients;
}

function dotGradient(gradients, period, ix, iy, x, y) {
    const wrappedX = ((ix % period) + period) % period;
    const wrappedY = ((iy % period) + period) % period;
    const index = (wrappedY * period + wrappedX) * 2;
    const dx = x - ix;
    const dy = y - iy;

    return dx * gradients[index] + dy * gradients[index + 1];
}

function sampleTileablePerlinNoise(x, y, period, gradients) {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = x0 + 1;
    const y1 = y0 + 1;

    const sx = fade(x - x0);
    const sy = fade(y - y0);

    const n00 = dotGradient(gradients, period, x0, y0, x, y);
    const n10 = dotGradient(gradients, period, x1, y0, x, y);
    const n01 = dotGradient(gradients, period, x0, y1, x, y);
    const n11 = dotGradient(gradients, period, x1, y1, x, y);

    const ix0 = lerp(n00, n10, sx);
    const ix1 = lerp(n01, n11, sx);

    return lerp(ix0, ix1, sy) * 0.5 + 0.5;
}

function createTileableNoiseData(size = FALLBACK_NOISE_SIZE) {
    const data = new Uint8Array(size * size);
    const octaveConfigs = [];
    let totalAmplitude = 0.0;

    for (let octave = 0; octave < FALLBACK_OCTAVES; octave++) {
        const period = FALLBACK_BASE_FREQUENCY << octave;
        const amplitude = Math.pow(FALLBACK_PERSISTENCE, octave);
        octaveConfigs.push({
            amplitude,
            gradients: createGradientGrid(period, 17 + octave * 31),
            period,
        });
        totalAmplitude += amplitude;
    }

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            let noiseValue = 0.0;

            for (const octave of octaveConfigs) {
                const sampleX = (x / size) * octave.period;
                const sampleY = (y / size) * octave.period;
                noiseValue += octave.amplitude * sampleTileablePerlinNoise(
                    sampleX,
                    sampleY,
                    octave.period,
                    octave.gradients,
                );
            }

            noiseValue /= totalAmplitude;
            noiseValue = smoothstepScalar(0.18, 0.88, noiseValue);
            data[y * size + x] = clampByte(noiseValue * 255);
        }
    }

    return data;
}

function createGradientGrid3D(period, seed) {
    const gradients = new Float32Array(period * period * period * 3);

    for (let z = 0; z < period; z++) {
        for (let y = 0; y < period; y++) {
            for (let x = 0; x < period; x++) {
                const angleA = TWO_PI * hash2D(x + z * 17, y + seed * 13, seed);
                const angleB = TWO_PI * hash2D(y + x * 11, z + seed * 7, seed + 29);
                const sinB = Math.sin(angleB);
                const index = ((z * period * period) + (y * period) + x) * 3;

                gradients[index] = Math.cos(angleA) * sinB;
                gradients[index + 1] = Math.sin(angleA) * sinB;
                gradients[index + 2] = Math.cos(angleB);
            }
        }
    }

    return gradients;
}

function dotGradient3D(gradients, period, ix, iy, iz, x, y, z) {
    const wrappedX = ((ix % period) + period) % period;
    const wrappedY = ((iy % period) + period) % period;
    const wrappedZ = ((iz % period) + period) % period;
    const index = ((wrappedZ * period * period) + (wrappedY * period) + wrappedX) * 3;
    const dx = x - ix;
    const dy = y - iy;
    const dz = z - iz;

    return (
        dx * gradients[index] +
        dy * gradients[index + 1] +
        dz * gradients[index + 2]
    );
}

function sampleTileablePerlinNoise3D(x, y, z, period, gradients) {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const z0 = Math.floor(z);
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const z1 = z0 + 1;

    const sx = fade(x - x0);
    const sy = fade(y - y0);
    const sz = fade(z - z0);

    const n000 = dotGradient3D(gradients, period, x0, y0, z0, x, y, z);
    const n100 = dotGradient3D(gradients, period, x1, y0, z0, x, y, z);
    const n010 = dotGradient3D(gradients, period, x0, y1, z0, x, y, z);
    const n110 = dotGradient3D(gradients, period, x1, y1, z0, x, y, z);
    const n001 = dotGradient3D(gradients, period, x0, y0, z1, x, y, z);
    const n101 = dotGradient3D(gradients, period, x1, y0, z1, x, y, z);
    const n011 = dotGradient3D(gradients, period, x0, y1, z1, x, y, z);
    const n111 = dotGradient3D(gradients, period, x1, y1, z1, x, y, z);

    const nx00 = lerp(n000, n100, sx);
    const nx10 = lerp(n010, n110, sx);
    const nx01 = lerp(n001, n101, sx);
    const nx11 = lerp(n011, n111, sx);
    const nxy0 = lerp(nx00, nx10, sy);
    const nxy1 = lerp(nx01, nx11, sy);

    return lerp(nxy0, nxy1, sz) * 0.5 + 0.5;
}

function createTileableVolumeNoiseData(size = FALLBACK_VOLUME_NOISE_SIZE) {
    const data = new Uint8Array(size * size * size);
    const octaveConfigs = [];
    let totalAmplitude = 0.0;

    for (let octave = 0; octave < FALLBACK_OCTAVES - 1; octave++) {
        const period = FALLBACK_BASE_FREQUENCY << octave;
        const amplitude = Math.pow(FALLBACK_PERSISTENCE, octave);
        octaveConfigs.push({
            amplitude,
            gradients: createGradientGrid3D(period, 23 + octave * 37),
            period,
        });
        totalAmplitude += amplitude;
    }

    for (let z = 0; z < size; z++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let noiseValue = 0.0;

                for (const octave of octaveConfigs) {
                    const sampleX = (x / size) * octave.period;
                    const sampleY = (y / size) * octave.period;
                    const sampleZ = (z / size) * octave.period;
                    noiseValue += octave.amplitude * sampleTileablePerlinNoise3D(
                        sampleX,
                        sampleY,
                        sampleZ,
                        octave.period,
                        octave.gradients,
                    );
                }

                noiseValue /= totalAmplitude;
                noiseValue = smoothstepScalar(0.16, 0.88, noiseValue);

                const index = (z * size * size) + (y * size) + x;
                data[index] = clampByte(noiseValue * 255);
            }
        }
    }

    return data;
}

function configureNoiseTexture(texture, managed = false) {
    if (!texture) {
        return texture;
    }

    texture.wrapS = RepeatWrapping;
    texture.wrapT = RepeatWrapping;
    texture.minFilter = LinearFilter;
    texture.magFilter = LinearFilter;
    texture.generateMipmaps = false;
    texture.flipY = false;
    texture.colorSpace = NoColorSpace;

    texture.userData = {
        ...texture.userData,
        isSkyCloudManagedNoiseTexture: managed,
    };
    texture.needsUpdate = true;

    return texture;
}

function configureVolumeNoiseTexture(texture, managed = false) {
    if (!texture) {
        return texture;
    }

    texture.wrapS = RepeatWrapping;
    texture.wrapT = RepeatWrapping;
    texture.wrapR = RepeatWrapping;
    texture.minFilter = LinearFilter;
    texture.magFilter = LinearFilter;
    texture.generateMipmaps = false;
    texture.flipY = false;
    texture.unpackAlignment = 1;
    texture.colorSpace = NoColorSpace;

    texture.userData = {
        ...texture.userData,
        isSkyCloudManagedNoiseTexture: managed,
    };
    texture.needsUpdate = true;

    return texture;
}

function createNoiseTextureFromData(data, size) {
    const rgba = new Uint8Array(size * size * 4);

    for (let i = 0; i < data.length; i++) {
        const offset = i * 4;
        rgba[offset] = data[i];
        rgba[offset + 1] = data[i];
        rgba[offset + 2] = data[i];
        rgba[offset + 3] = 255;
    }

    return configureNoiseTexture(new DataTexture(rgba, size, size, RGBAFormat), true);
}

function createFallbackPerlinTexture() {
    return createNoiseTextureFromData(createTileableNoiseData(FALLBACK_NOISE_SIZE), FALLBACK_NOISE_SIZE);
}

function createNeutralPerlinTexture() {
    const texture = new DataTexture(new Uint8Array([128, 128, 128, 255]), 1, 1, RGBAFormat);
    texture.wrapS = ClampToEdgeWrapping;
    texture.wrapT = ClampToEdgeWrapping;
    texture.minFilter = LinearFilter;
    texture.magFilter = LinearFilter;
    texture.generateMipmaps = false;
    texture.flipY = false;
    texture.colorSpace = NoColorSpace;
    texture.needsUpdate = true;
    return texture;
}

function createVolumeNoiseTextureFromData(data, size) {
    const texture = new Data3DTexture(data, size, size, size);
    texture.format = RedFormat;
    texture.type = UnsignedByteType;
    return configureVolumeNoiseTexture(texture, true);
}

function createFallbackVolumeNoiseTexture() {
    return createVolumeNoiseTextureFromData(createTileableVolumeNoiseData(FALLBACK_VOLUME_NOISE_SIZE), FALLBACK_VOLUME_NOISE_SIZE);
}

function normalizeParamValue(name, value) {
    if (!(name in DEFAULT_PARAMS)) {
        return value;
    }

    return Number.isFinite(value) ? value : DEFAULT_PARAMS[name];
}

function createNodeSet(options = {}) {
    const initialTexture2D = configureNoiseTexture(options.perlinTexture ?? createNeutralPerlinTexture(), !options.perlinTexture);
    const initialTexture3D = configureVolumeNoiseTexture(options.volumeNoiseTexture ?? createFallbackVolumeNoiseTexture(), !options.volumeNoiseTexture);
    const noiseMode = options.perlinTexture ? 0.0 : 1.0;

    return {
        t_PerlinNoise: texture(initialTexture2D),
        t_PerlinNoise3D: texture3D(initialTexture3D),
        uCloudAbsorption: uniform(normalizeParamValue('cloudAbsorption', options.cloudAbsorption ?? DEFAULT_PARAMS.cloudAbsorption)),
        uCloudCoverage: uniform(normalizeParamValue('cloudCoverage', options.cloudCoverage ?? DEFAULT_PARAMS.cloudCoverage)),
        uCloudHeight: uniform(normalizeParamValue('cloudHeight', options.cloudHeight ?? DEFAULT_PARAMS.cloudHeight)),
        uCloudThickness: uniform(normalizeParamValue('cloudThickness', options.cloudThickness ?? DEFAULT_PARAMS.cloudThickness)),
        uMaxCloudDistance: uniform(normalizeParamValue('maxCloudDistance', options.maxCloudDistance ?? DEFAULT_PARAMS.maxCloudDistance)),
        uNoiseMode: uniform(noiseMode),
        uResolution: uniform(new Vector2(options.width ?? 1920, options.height ?? 1080)),
        uSunDirection: uniform((options.sunDirection ?? new Vector3(0.5, 0.5, -0.5)).clone().normalize()),
        uTime: uniform(options.time ?? 0.0),
        uWindSpeedX: uniform(normalizeParamValue('windSpeedX', options.windSpeedX ?? DEFAULT_PARAMS.windSpeedX)),
        uWindSpeedZ: uniform(normalizeParamValue('windSpeedZ', options.windSpeedZ ?? DEFAULT_PARAMS.windSpeedZ)),
    };
}

function attachMaterialState(material, nodes) {
    material.userData = {
        ...material.userData,
        isSkyCloudMaterial: true,
        skyCloudNodes: nodes,
    };

    material.uniforms = nodes;
}

function getSkyCloudNodes(material) {
    return material?.userData?.skyCloudNodes ?? null;
}

function buildColorNode(nodes) {
    const getSkyColor = Fn(( [ rayDirImmutable ] ) => {
        const rayDir = vec3(rayDirImmutable);
        const sunAmount = max(0.0, dot(rayDir, nodes.uSunDirection));
        const skyGradient = pow(max(0.0, rayDir.y), 0.5);
        const sunHalo = pow(sunAmount, 48.0);
        const sunDisc = smoothstep(0.99985, 0.99997, sunAmount);

        const skyColor = mix(
            vec3(0.095, 0.19, 0.38),
            vec3(0.82, 0.72, 0.56),
            pow(sunAmount, 12.0),
        ).toVar();

        skyColor.assign(mix(vec3(0.68, 0.73, 0.79), skyColor, skyGradient));
        skyColor.addAssign(vec3(1.0, 0.82, 0.62).mul(sunHalo).mul(0.18));
        skyColor.addAssign(vec3(1.0, 0.8, 0.58).mul(sunDisc).mul(1.4));

        return skyColor;
    });

    const noiseTransform = mat3(
        0.0, 0.968, 0.726,
        -0.968, 0.4356, -0.5808,
        -0.726, -0.5808, 0.7744,
    );

    const noise3D = Fn(( [ pImmutable ] ) => {
        const p = vec3(pImmutable);
        const sample2D = nodes.t_PerlinNoise.sample(p.xz.mul(0.01)).x;
        const sample3D = nodes.t_PerlinNoise3D.sample(
            p.mul(vec3(0.0075, 0.005, 0.0075)),
        ).x;

        return mix(sample2D, sample3D, nodes.uNoiseMode);
    });

    const fbm = Fn(( [ pImmutable ] ) => {
        const p = vec3(pImmutable).toVar();
        const value = float(0.0).toVar();
        const mult = float(2.76434);

        value.addAssign(noise3D(p).mul(0.51749673));
        p.assign(noiseTransform.mul(p).mul(mult));

        value.addAssign(noise3D(p).mul(0.25584929));
        p.assign(noiseTransform.mul(p).mul(mult));

        value.addAssign(noise3D(p).mul(0.12527603));
        p.assign(noiseTransform.mul(p).mul(mult));

        value.addAssign(noise3D(p).mul(0.06255931));

        return value;
    });

    const cloudDensity = Fn(( [ posImmutable, offsetImmutable ] ) => {
        const pos = vec3(posImmutable);
        const offset = vec3(offsetImmutable);
        const density = fbm(pos.mul(0.0212242).add(offset)).toVar();
        const detail = fbm(
            pos.mul(0.048).add(offset.mul(1.73)).add(vec3(19.1, 0.0, 7.3)),
        ).toVar();
        const coverageCutoff = float(1.0).sub(nodes.uCloudCoverage);
        const layerHeight = max(nodes.uCloudThickness, 0.0001);
        const height = pos.y.sub(nodes.uCloudHeight);
        const heightWarp = detail.sub(0.5).mul(0.18).toVar();
        const height01 = clamp(height.div(layerHeight).add(heightWarp), 0.0, 1.0).toVar();
        const bottomFade = smoothstep(0.0, 0.32, height01).toVar();
        const topFade = float(1.0).sub(smoothstep(0.42, 1.0, height01)).toVar();
        const heightAttenuation = bottomFade.mul(topFade).toVar();
        const detailLift = detail.sub(0.5).mul(0.08);

        density.addAssign(detailLift);
        density.assign(smoothstep(coverageCutoff.sub(0.04), coverageCutoff.add(0.12), density));
        density.mulAssign(smoothstep(0.02, 0.65, density));
        density.mulAssign(heightAttenuation);

        return clamp(density, 0.0, 1.0);
    });

    const cloudLight = Fn(( [ posImmutable, dirStepImmutable, offsetImmutable ] ) => {
        const pos = vec3(posImmutable).toVar();
        const dirStep = vec3(dirStepImmutable);
        const offset = vec3(offsetImmutable);
        const transmittance = float(1.0).toVar();

        Loop(LIGHT_STEPS, () => {
            const density = cloudDensity(pos, offset);
            const stepTransmittance = exp(nodes.uCloudAbsorption.negate().mul(density));

            transmittance.mulAssign(stepTransmittance);
            pos.addAssign(dirStep);
        });

        return transmittance;
    });

    const renderClouds = Fn(( [ rayOriginImmutable, rayDirectionImmutable ] ) => {
        const rayOrigin = vec3(rayOriginImmutable);
        const rayDirection = vec3(rayDirectionImmutable);
        const color = vec3(0.0).toVar();
        const alpha = float(0.0).toVar();
        const epsilon = float(0.0001);

        If(abs(rayDirection.y).greaterThan(epsilon), () => {
            const cloudBase = nodes.uCloudHeight;
            const cloudTop = nodes.uCloudHeight.add(nodes.uCloudThickness);
            const tBase = cloudBase.sub(rayOrigin.y).div(rayDirection.y);
            const tTop = cloudTop.sub(rayOrigin.y).div(rayDirection.y);
            const tEntry = max(min(tBase, tTop), 0.0).toVar();
            const tExit = min(max(tBase, tTop), nodes.uMaxCloudDistance).toVar();

            If(tExit.greaterThan(tEntry), () => {
                const distanceFade = float(1.0).sub(smoothstep(nodes.uMaxCloudDistance.mul(0.6), nodes.uMaxCloudDistance, tEntry));
                const marchDistance = tExit.sub(tEntry);
                const angleFactor = float(1.0).sub(abs(rayDirection.y)).toVar();
                const targetSteps = clamp(
                    float(BASE_CLOUD_STEPS)
                        .add(angleFactor.mul(18.0))
                        .add(marchDistance.div(max(nodes.uCloudThickness, epsilon)).mul(2.0)),
                    float(BASE_CLOUD_STEPS),
                    float(MAX_CLOUD_STEPS),
                ).toVar();
                const marchStep = marchDistance.div(targetSteps).toVar();
                const dirStep = rayDirection.mul(marchStep);
                const lightStep = nodes.uSunDirection.mul(5.0);
                const windOffset = vec3(
                    nodes.uTime.mul(nodes.uWindSpeedX).negate(),
                    0.0,
                    nodes.uTime.mul(nodes.uWindSpeedZ).negate(),
                );
                const jitter = tslFract(
                    tslSin(
                        dot(
                            rayDirection.add(rayOrigin.mul(0.0001)),
                            vec3(12.9898, 78.233, 37.719),
                        ),
                    ).mul(43758.5453123),
                ).toVar();
                const pos = rayOrigin.add(
                    rayDirection.mul(tEntry.add(marchStep.mul(jitter.sub(0.5)))),
                ).toVar();
                const transmittance = float(1.0).toVar();
                const cloudColor = vec3(0.0).toVar();
                const cloudAlpha = float(0.0).toVar();
                const layerHeight = max(nodes.uCloudThickness, epsilon);

                Loop(MAX_CLOUD_STEPS, ( { i } ) => {
                    If(float(i).greaterThanEqual(targetSteps), () => {
                        Break();
                    });

                    const density = cloudDensity(pos, windOffset).toVar();

                    If(density.greaterThan(0.005), () => {
                        const stepTransmittance = exp(nodes.uCloudAbsorption.negate().mul(density).mul(marchStep)).toVar();
                        const directLight = cloudLight(pos, lightStep, windOffset);
                        const h = clamp(pos.y.sub(cloudBase).div(layerHeight), 0.0, 1.0).toVar();
                        const lightFactor = exp(h).div(1.95);
                        const sunContribution = pow(max(0.0, dot(rayDirection, nodes.uSunDirection)), 2.5);
                        const edgeColor = mix(
                            vec3(0.82, 0.84, 0.88),
                            vec3(1.0, 0.84, 0.62),
                            clamp(sunContribution.mul(0.7).add(directLight.mul(0.2)), 0.0, 1.0),
                        );
                        const shadeMix = clamp(directLight.mul(lightFactor).add(sunContribution.mul(0.18)), 0.0, 1.0);
                        const shadedColor = mix(
                            vec3(0.34, 0.36, 0.4),
                            edgeColor,
                            shadeMix,
                        );

                        transmittance.mulAssign(stepTransmittance);
                        cloudColor.addAssign(shadedColor.mul(transmittance).mul(density).mul(marchStep).mul(1.18));
                        cloudAlpha.addAssign(float(1.0).sub(stepTransmittance).mul(float(1.0).sub(cloudAlpha)));
                    });

                    pos.addAssign(dirStep);

                    If(transmittance.lessThan(0.01), () => {
                        Break();
                    });
                });

                color.assign(
                    cloudColor.mul(
                        mix(
                            vec3(0.4, 0.5, 0.6),
                            vec3(0.9, 0.7, 0.5),
                            pow(max(0.0, dot(rayDirection, nodes.uSunDirection)), 2.0).mul(0.5),
                        ),
                    ).mul(distanceFade),
                );
                alpha.assign(cloudAlpha.mul(distanceFade));
            });
        });

        return vec4(color, alpha);
    });

    return Fn(() => {
        const rayDirection = normalize(positionWorld.sub(cameraPosition)).toVar();
        const skyColor = getSkyColor(rayDirection).toVar();
        const clouds = renderClouds(cameraPosition, rayDirection).toVar();
        const finalColor = mix(skyColor, clouds.rgb, clouds.a).toVar();
        const horizonFade = pow(float(1.0).sub(max(0.0, rayDirection.y)), 5.0);

        finalColor.assign(mix(finalColor, vec3(0.65, 0.7, 0.75), horizonFade.mul(0.5)));

        return vec4(finalColor, 1.0);
    })();
}

function createSkyMaterial(options = {}) {
    const material = new NodeMaterial();
    const nodes = createNodeSet(options);

    material.name = 'SkyCloudNodeMaterial';
    material.side = BackSide;
    material.depthWrite = false;
    material.fog = false;
    material.toneMapped = true;
    material.colorNode = buildColorNode(nodes);

    attachMaterialState(material, nodes);

    return material;
}

function loadPerlinTexture(url, managed = false) {
    return new Promise((resolve, reject) => {
        new TextureLoader().load(
            url,
            (loadedTexture) => resolve(configureNoiseTexture(loadedTexture, managed)),
            undefined,
            reject,
        );
    });
}

function disposeManagedNoiseTexture(textureValue, nextTextureValue) {
    if (!textureValue || textureValue === nextTextureValue) {
        return;
    }

    if (textureValue.userData?.isSkyCloudManagedNoiseTexture) {
        textureValue.dispose();
    }
}

function disposeSkyCloudMaterialResources(material) {
    const nodes = getSkyCloudNodes(material);

    if (!nodes) {
        return;
    }

    disposeManagedNoiseTexture(nodes.t_PerlinNoise?.value);
    disposeManagedNoiseTexture(nodes.t_PerlinNoise3D?.value);
}

function setPerlinNoiseTexture(material, nextTexture, managed = false) {
    const nodes = getSkyCloudNodes(material);

    if (!nodes || !nextTexture) {
        return material;
    }

    const isVolumeTexture = nextTexture.isData3DTexture === true;
    const configuredTexture = isVolumeTexture
        ? configureVolumeNoiseTexture(nextTexture, managed)
        : configureNoiseTexture(nextTexture, managed);

    if (isVolumeTexture) {
        const previousTexture = nodes.t_PerlinNoise3D.value;

        nodes.t_PerlinNoise3D.value = configuredTexture;
        nodes.uNoiseMode.value = 1.0;
        material.needsUpdate = true;

        disposeManagedNoiseTexture(previousTexture, configuredTexture);

        return material;
    }

    const previousTexture = nodes.t_PerlinNoise.value;

    nodes.t_PerlinNoise.value = configuredTexture;
    nodes.uNoiseMode.value = 0.0;
    material.needsUpdate = true;

    disposeManagedNoiseTexture(previousTexture, configuredTexture);

    return material;
}

async function createSkyMaterialWithTexture(options = {}) {
    const material = createSkyMaterial(options);

    if (options.perlinTexture) {
        setPerlinNoiseTexture(material, options.perlinTexture);
        return material;
    }

    if (options.volumeNoiseTexture) {
        setPerlinNoiseTexture(material, options.volumeNoiseTexture);
        return material;
    }

    if (!options.perlinTextureUrl) {
        return material;
    }

    try {
        const loadedTexture = await loadPerlinTexture(options.perlinTextureUrl, true);
        setPerlinNoiseTexture(material, loadedTexture, true);
    } catch (error) {
        console.warn('SkyCloudMesh: failed to load perlin texture, keeping fallback texture.', error);
    }

    return material;
}

class SkyCloudMesh extends Mesh {
    constructor(options = {}) {
        const {
            radius = 20000,
            widthSegments = 64,
            heightSegments = 32,
            ...materialOptions
        } = options;

        const geometry = new SphereGeometry(radius, widthSegments, heightSegments);
        const material = createSkyMaterial(materialOptions);

        super(geometry, material);

        this.isSkyCloudMesh = true;
        this.frustumCulled = false;
        this.renderOrder = -1000;
        this._disposed = false;

        this.ready = Promise.resolve(this);

        if (materialOptions.perlinTexture) {
            setPerlinNoiseTexture(this.material, materialOptions.perlinTexture);
        } else if (materialOptions.perlinTextureUrl) {
            this.ready = loadPerlinTexture(materialOptions.perlinTextureUrl, true)
                .then((loadedTexture) => {
                    if (this._disposed || !this.material) {
                        disposeManagedNoiseTexture(loadedTexture);
                        return this;
                    }

                    if (this.material) {
                        setPerlinNoiseTexture(this.material, loadedTexture, true);
                    }

                    return this;
                })
                .catch((error) => {
                    console.warn('SkyCloudMesh: failed to load perlin texture, keeping fallback texture.', error);
                    return this;
                });
        }
    }

    updateSun(direction) {
        const nodes = getSkyCloudNodes(this.material);

        if (nodes?.uSunDirection && direction) {
            nodes.uSunDirection.value.copy(direction).normalize();
        }
    }

    updateTime(time) {
        const nodes = getSkyCloudNodes(this.material);

        if (nodes?.uTime) {
            nodes.uTime.value = time;
        }
    }

    updateResolution(width, height) {
        const nodes = getSkyCloudNodes(this.material);

        if (nodes?.uResolution) {
            nodes.uResolution.value.set(width, height);
        }
    }

    updateCamera(camera) {
        if (camera?.position) {
            this.position.copy(camera.position);
        }
    }

    setParameter(name, value) {
        const nodes = getSkyCloudNodes(this.material);

        if (!nodes) {
            return;
        }

        if (name === 'sunDirection' && value?.isVector3) {
            nodes.uSunDirection.value.copy(value).normalize();
            return;
        }

        if (name === 'perlinTexture' && value) {
            setPerlinNoiseTexture(this.material, value);
            return;
        }

        if (name === 'volumeNoiseTexture' && value) {
            setPerlinNoiseTexture(this.material, value);
            return;
        }

        if (name === 'resolution' && value?.isVector2) {
            nodes.uResolution.value.copy(value);
            return;
        }

        const nodeName = PARAM_TO_NODE[name];

        if (nodeName && nodes[nodeName]) {
            nodes[nodeName].value = normalizeParamValue(name, value);
        }
    }

    dispose() {
        if (this._disposed) {
            return;
        }

        this._disposed = true;

        disposeSkyCloudMaterialResources(this.material);
        this.geometry?.dispose?.();
        this.material?.dispose?.();
    }
}

export {
    SkyCloudMesh,
    createSkyMaterial,
    createSkyMaterialWithTexture,
    setPerlinNoiseTexture,
};
