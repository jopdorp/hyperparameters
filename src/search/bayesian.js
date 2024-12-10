import * as math from 'mathjs';
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

  kernel(x1Array, x2Array) {
    const squaredDist = x1Array.map(x1 =>
      x2Array.map(x2 =>
        math.sum(math.dotDivide(
          math.square(math.subtract(x1, x2)),
          math.multiply(2, math.square(this.lengthScale))
        ))));
    return math.map(squaredDist, x => Math.exp(-x));
  }

  fit(X, y) {
    this.X = X;
    this.y = y;
    this.K = this.kernel(X, X);
    for (let i = 0; i < this.K.length; i += 1) {
      this.K[i][i] += this.noise;
    }
    this.L = math.lup(this.K);
  }

  predict(xNew) {
    if (this.X.length === 0) {
      return {
        mean: xNew.map(() => 0),
        variance: xNew.map(() => 1)
      };
    }

    const Ks = this.kernel(xNew, this.X);
    const Kss = this.kernel(xNew, xNew);

    const mean = math.multiply(Ks, math.lusolve(this.L, this.y));

    const v = math.lusolve(this.L, math.transpose(Ks));
    const variance = Kss.map((kss, i) => kss[i] - math.sum(math.multiply(Ks[i], v[i])));

    return { mean, variance };
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
    const X = this.observations.map(obs => Object.values(obs.params));
    const y = this.observations.map(obs => obs.result.loss || obs.result.accuracy);
    this.surrogateModel.fit(X, y);
  }

  // Generate candidate points from the search space
  generateCandidates(expr, rng, numCandidates = 10) {
    const candidates = [];
    for (let i = 0; i < numCandidates; i += 1) {
      const candidate = super.eval(expr, { rng });
      candidates.push(Object.values(candidate));
    }
    return candidates;
  }

  // Propose the next set of parameters to evaluate
  propose(expr, rng) {
    const candidates = this.generateCandidates(expr, rng);
    const { mean, variance } = this.surrogateModel.predict(candidates);

    const yBest = Math.min(...this.observations.map(obs => obs.result.loss || Infinity));
    const acquisitionValues = expectedImprovement(mean, variance, yBest);

    const maxIndex = acquisitionValues.indexOf(Math.max(...acquisitionValues));
    const bestCandidate = candidates[maxIndex];
    const paramKeys = Object.keys(super.eval(expr, { rng }));
    return paramKeys.reduce((acc, key, idx) => ({
      ...acc,
      [key]: bestCandidate[idx]
    }), {});
  }

  // Evaluate the expression using Bayesian optimization
  eval(expr, { rng }) {
    return this.propose(expr, rng);
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
  bs.updateSurrogateModel();

  newIds.forEach((newId) => {
    const paramsEval = bs.eval(domain.expr, { rng });
    const result = domain.newResult();
    rval = [...rval, ...trials.newTrialDocs([newId], [result], [paramsEval])];
  });
  return rval;
};
