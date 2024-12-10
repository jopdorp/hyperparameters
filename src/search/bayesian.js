import * as math from 'mathjs';
import cho from 'cholesky';
import BaseSpace from '../base/base';
import RandomState from '../utils/RandomState';

// Implement the surrogate model using Gaussian Process Regression
class SurrogateModel {
  constructor() {
    this.X = [];
    this.y = [];
    this.noise = 1e-6;
    this.lengthScale = 1.0;
  }

  kernel(X1, X2) {
    // Ensure inputs are 2D arrays
    const x1Matrix = Array.isArray(X1[0]) ? X1 : X1.map(x => (Array.isArray(x) ? x : [x]));
    const x2Matrix = Array.isArray(X2[0]) ? X2 : X2.map(x => (Array.isArray(x) ? x : [x]));

    return x1Matrix.map(x1 =>
      x2Matrix.map((x2) => {
        // Ensure arrays have same dimensions before subtracting
        const x1Arr = Array.isArray(x1) ? x1 : [x1];
        const x2Arr = Array.isArray(x2) ? x2 : [x2];

        if (x1Arr.length !== x2Arr.length) {
          const maxLen = Math.max(x1Arr.length, x2Arr.length);
          while (x1Arr.length < maxLen) x1Arr.push(0);
          while (x2Arr.length < maxLen) x2Arr.push(0);
        }

        const diff = math.subtract(math.matrix(x1Arr), math.matrix(x2Arr));
        const squaredDist = math.sum(math.square(diff));
        return Math.exp((-0.5 * squaredDist) / (this.lengthScale ** 2));
      }));
  }

  fit(X, y) {
    if (!X.length || !y.length) return;

    this.X = X;
    this.y = Array.isArray(y) ? y : [y];

    const K = this.kernel(X, X);
    const kY = math.add(K, math.multiply(this.noise, math.identity(K.length)));

    try {
      // Use the standalone cholesky package
      this.L = cho(kY);
      // Convert L to the format expected by subsequent operations
      this.L = this.L.map((row) => {
        const fullRow = new Array(kY.length).fill(0);
        row.forEach((val, j) => {
          fullRow[j] = val;
        });
        return fullRow;
      });
      const yMatrix = math.matrix(this.y);
      this.alpha = math.lusolve(math.matrix(this.L), math.transpose(yMatrix));
    } catch (e) {
      // Fallback to identity matrix if decomposition fails
      this.L = math.identity(K.length);
      this.alpha = math.matrix(this.y);
    }
  }

  predict(XNew) {
    const kS = this.kernel(XNew, this.X);
    const kSS = this.kernel(XNew, XNew);
    const mu = kS.map(kSRow => math.multiply(kSRow, this.alpha));
    const v = math.lusolve(this.L, math.transpose(kS));
    const cov = math.subtract(kSS, math.multiply(math.transpose(v), v));
    const stdv = cov.map((row, i) => Math.sqrt(Math.max(row[i], 1e-9)));
    return { mean: mu.map(m => m[0]), stdv };
  }
}

// Implement the Expected Improvement acquisition function
function expectedImprovement(mu, sigma, yBest) {
  const epsilon = 1e-9;
  return mu.map((m, i) => {
    const s = sigma[i] + epsilon;
    const z = (yBest - m) / s;
    const phi = math.exp(-0.5 * (z * z)) / math.sqrt(2 * Math.PI);
    const Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2)));
    return s * ((z * Phi) + phi);
  });
}

class BayesianSearch extends BaseSpace {
  constructor() {
    super();
    this.surrogateModel = new SurrogateModel();
    this.observations = [];
    this.categoricalMaps = new Map(); // Store mappings for categorical variables
  }

  // Add all parameter handlers to match RandomSearch
  choice = (params, rng) => {
    const { options } = params;
    const idx = rng.randrange(0, options.length, 1);
    const option = options[idx];
    return this.eval(option, { rng });
  };

  randint = (params, rng) => rng.randrange(0, params.upper, 1);

  loguniform = (params, rng) => {
    const { low, high } = params;
    return Math.exp(rng.uniform(low, high));
  };

  qloguniform = (params, rng) => {
    const { low, high, q } = params;
    return Math.round(Math.exp(rng.uniform(low, high)) / q) * q;
  };

  normal = (params, rng) => {
    const { mu, sigma } = params;
    return rng.gauss(mu, sigma);
  };

  qnormal = (params, rng) => {
    const { mu, sigma, q } = params;
    return Math.round(rng.gauss(mu, sigma) / q) * q;
  };

  lognormal = (params, rng) => {
    const { mu, sigma } = params;
    return Math.exp(rng.gauss(mu, sigma));
  };

  qlognormal = (params, rng) => {
    const { mu, sigma, q } = params;
    return Math.round(Math.exp(rng.gauss(mu, sigma)) / q) * q;
  };

  // Update the surrogate model with new observations
  updateSurrogateModel() {
    const X = this.observations.map(obs => this.flattenParams(obs.params));
    const y = this.observations.map(obs => obs.result.loss || obs.result.accuracy);
    if (X.length > 0) {
      this.surrogateModel.fit(X, y);
    }
  }

  // Generate candidate points from the search space
  generateCandidates(expr, rng, numCandidates = 10) {
    const candidates = [];
    const flatCandidates = [];
    for (let i = 0; i < numCandidates; i += 1) {
      const candidate = super.eval(expr, { rng });
      candidates.push(candidate);
      flatCandidates.push(this.flattenParams(candidate));
    }
    return { candidates, flatCandidates };
  }

  // Helper method to encode categorical values
  encodeCategorical(value) {
    if (typeof value === 'string') {
      if (!this.categoricalMaps.has(value)) {
        this.categoricalMaps.set(value, this.categoricalMaps.size);
      }
      return this.categoricalMaps.get(value);
    }
    if (typeof value === 'boolean') {
      return value ? 1 : 0;
    }
    return value;
  }

  // Helper method to flatten parameter objects
  flattenParams(params) {
    if (Array.isArray(params)) {
      return params.flatMap(p => this.flattenParams(p));
    }
    if (typeof params === 'object' && params !== null) {
      return Object.values(params).flatMap(p => this.flattenParams(p));
    }
    return [this.encodeCategorical(params)];
  }

  // Propose the next set of parameters to evaluate
  propose(expr, rng) {
    const { candidates, flatCandidates } = this.generateCandidates(expr, rng, 100);
    const { mean, stdv } = this.surrogateModel.predict(flatCandidates);

    const yBest = Math.min(...this.observations.map(obs => obs.result.loss || Infinity));
    const eiValues = expectedImprovement(mean, stdv, yBest);

    const maxIndex = eiValues.indexOf(Math.max(...eiValues));
    return candidates[maxIndex]; // Return the original structured candidate
  }

  // Evaluate the expression using Bayesian optimization
  eval(expr, { rng }) {
    if (this.observations.length > 0) {
      return this.propose(expr, rng);
    }
    // Random sample for the initial observation
    return super.eval(expr, { rng });
  }

  // Add parameter handlers similar to RandomSearch
  uniform = (params, rng) => {
    const { low, high } = params;
    return rng.uniform(low, high);
  };

  quniform = (params, rng) => {
    const { low, high, q } = params;
    return Math.round(rng.uniform(low, high) / q) * q;
  };

  // Add other parameter handlers (normal, lognormal, etc.) as needed
}

export const bayesianSample = (space, params = {}) => {
  const bs = new BayesianSearch();
  const args = bs.eval(space, params);
  if (Object.keys(args).length === 1) {
    const results = Object.keys(args).map(key => args[key]);
    return results.length === 1 ? results[0] : results;
  }
  return args;
};

export const bayesianSearch = (newIds, domain, trials, seed) => {
  const rng = new RandomState(seed);
  let rval = [];
  const bs = new BayesianSearch();

  // Update observations from past trials
  bs.observations = trials.trials.map(trial => ({
    params: trial.args,
    result: trial.result,
  }));
  if (bs.observations.length > 0) {
    bs.updateSurrogateModel();
  }

  newIds.forEach((newId) => {
    const paramsEval = bs.eval(domain.expr, { rng });
    const result = domain.newResult();
    rval = [...rval, ...trials.newTrialDocs([newId], [result], [paramsEval])];
  });
  return rval;
};
