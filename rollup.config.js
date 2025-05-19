import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';

export default {
  input: 'SkyCloudMesh.js',
  output: {
    file: 'dist/SkyCloudMesh.min.js',
    format: 'esm',
  },
  plugins: [
    resolve(),
    commonjs(),
    terser()
  ],
  external: ['three']
};
