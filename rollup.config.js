import nodeResolve from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';
import babel from 'rollup-plugin-babel';
import replace from 'rollup-plugin-replace';
import uglify from 'rollup-plugin-uglify';

const env = process.env.NODE_ENV;
const config = {
  input: 'src/index.js',
  external: ['mathjs'],
  output: {
    globals: {
      mathjs: 'math'
    }
  },
  plugins: [
    commonjs({
      include: 'node_modules/**',
      namedExports: {
        'mathjs': [
          'exp', 'sqrt', 'sum', 'multiply', 'dotDivide', 'square', 'subtract', 'erf', 'lup', 'lusolve', 'transpose', 'map'
        ]
      }
    })
  ]
};

if (env === 'es' || env === 'cjs') {
  config.output = { 
    ...config.output,
    format: env, 
    indent: false 
  };
  config.plugins.push(
    babel({
      plugins: ['external-helpers'],
      runtimeHelpers: true
    })
  );
}

if (env === 'development' || env === 'production') {
  config.output = { 
    ...config.output,
    format: 'umd',
    name: 'hpjs',
    indent: false
  };
  config.plugins.push(
    nodeResolve({
      jsnext: true,
      preferBuiltins: true
    }),
    babel({
      exclude: 'node_modules/**',
      plugins: ['external-helpers'],
      runtimeHelpers: true
    }),
    replace({
      'process.env.NODE_ENV': JSON.stringify(env)
    })
  );
}

if (env === 'production') {
  config.plugins.push(
    uglify({
      compress: {
        pure_getters: true,
        unsafe: true,
        unsafe_comps: true,
        warnings: false
      }
    })
  );
}

export default config;
