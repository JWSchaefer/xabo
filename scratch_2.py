"""Sanity check for Prior-wraps-value architecture.

Tests three cases:
1. Fixed prior (standard usage)
2. Learnable hyperparameters (empirical Bayes)
3. Metaprior (hierarchical Bayes)
4. Model.log_prior() with hyperprior evaluation
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from xabo.core import Model
from xabo.core.prior import (
    HalfNormal,
    LogNormalLearnable,
    NoPrior,
    Normal,
    NormalLearnable,
)
from xabo.core.spec import Spec
from xabo.core.transform import Log

# =============================================================================
# Case 1: Fixed prior (standard usage)
# =============================================================================
print('=' * 60)
print('Case 1: Fixed prior')
print('=' * 60)


class FixedPriorSpec(Spec):
    ell: NoPrior[float]


fixed_spec = FixedPriorSpec(ell=NoPrior(value=1.0))
fixed_params = fixed_spec.init_params()
fixed_state = fixed_spec.init_state()

print(f'Spec: {fixed_spec}')
print(f'Params: {fixed_params}')
print(f'Params.ell.value: {fixed_params.ell.value}')
print(f'State: {fixed_state}')

# Test the prior directly
ell_prior = fixed_spec.ell
prior_params = ell_prior.init_params()
prior_state = ell_prior.init_state()
print(f'\nPrior: {ell_prior}')
print(f'Prior.Params: {prior_params}')
print(f'Prior.Params.value: {prior_params.value}')


# Evaluate log prob
x = jnp.array(1.5)
lp = ell_prior.log_prob(x, prior_params, prior_state)
print(f'\nlog_prob({x}) = {lp}')

# Sample
key = jr.PRNGKey(0)
sample = ell_prior.sample(key, prior_params, prior_state, shape=(3,))
print(f'sample(shape=(3,)) = {sample}')

# Tree leaves - should contain just the value
print(f'\ntree_leaves(params): {jax.tree.leaves(fixed_params)}')

# Transforms - LogNormal uses Log transform
transforms = fixed_spec.get_transforms()
print(f'Transforms: {transforms}')


# =============================================================================
# Case 2: Learnable hyperparameters (empirical Bayes)
# =============================================================================
print('\n' + '=' * 60)
print('Case 2: Learnable hyperparameters (empirical Bayes)')
print('=' * 60)


class LearnablePriorSpec(Spec):
    ell: LogNormalLearnable[float, Normal[float], HalfNormal[float]]


learnable_spec = LearnablePriorSpec(
    ell=LogNormalLearnable(
        value=1.0,
        mu=Normal(value=0.0, loc=0.0, scale=10.0),
        sigma=HalfNormal(value=1.0, scale=1.0),
    )
)
learnable_params = learnable_spec.init_params()

print(f'Spec: {learnable_spec}')
print(f'Params: {learnable_params}')
print(f'Params.ell.value: {learnable_params.ell.value}')
print(f'Params.ell.mu.value: {learnable_params.ell.mu.value}')
print(f'Params.ell.sigma.value: {learnable_params.ell.sigma.value}')

# Tree leaves - should contain value, mu.value, sigma.value
leaves = jax.tree.leaves(learnable_params)
print(f'\ntree_leaves(params): {leaves}')
print(f'Number of leaves: {len(leaves)}')

# Evaluate log prob
x = jnp.array(1.5)
ell_state = learnable_spec.ell.init_state()
lp = learnable_spec.ell.log_prob(x, learnable_params.ell, ell_state)
print(f'\nlog_prob({x}) = {lp}')

# Sample
sample = learnable_spec.ell.sample(
    key, learnable_params.ell, ell_state, shape=(3,)
)
print(f'sample(shape=(3,)) = {sample}')


# =============================================================================
# Case 3: Metaprior (hierarchical Bayes)
# =============================================================================
print('\n' + '=' * 60)
print('Case 3: Metaprior (hierarchical Bayes)')
print('=' * 60)


class MetaPriorSpec(Spec):
    ell: LogNormalLearnable[
        float,
        NormalLearnable[float, Normal[float], HalfNormal[float]],
        HalfNormal[float],
    ]


meta_spec = MetaPriorSpec(
    ell=LogNormalLearnable(
        value=1.0,
        mu=NormalLearnable(
            value=0.0,
            loc=Normal(value=0.0, loc=0.0, scale=100.0),
            scale=HalfNormal(value=1.0, scale=10.0),
        ),
        sigma=HalfNormal(value=1.0, scale=1.0),
    )
)
meta_params = meta_spec.init_params()

print(f'Params.ell.value: {meta_params.ell.value}')
print(f'Params.ell.mu.value: {meta_params.ell.mu.value}')
print(f'Params.ell.mu.loc.value: {meta_params.ell.mu.loc.value}')
print(f'Params.ell.mu.scale.value: {meta_params.ell.mu.scale.value}')
print(f'Params.ell.sigma.value: {meta_params.ell.sigma.value}')

# Tree leaves - should contain ALL learnable values in the hierarchy
leaves = jax.tree.leaves(meta_params)
print(f'\ntree_leaves(params): {leaves}')
print(
    f'Number of leaves: {len(leaves)} (expected 5: value, mu.value, mu.loc.value, mu.scale.value, sigma.value)'
)


# =============================================================================
# Case 4: Model.log_prior() with hyperprior evaluation
# =============================================================================
print('\n' + '=' * 60)
print('Case 4: Model.log_prior() with hyperprior evaluation')
print('=' * 60)

# Create a model with learnable hyperparameters
model = Model.from_spec(learnable_spec)

print(f'Model.params: {model.params}')
print(f'Model.params.ell.value: {model.params.ell.value}')
print(f'Model.params.ell.mu.value: {model.params.ell.mu.value}')
print(f'Model.params.ell.sigma.value: {model.params.ell.sigma.value}')

# Get unconstrained params
unconstrained = model.to_unconstrained()
print(f'\nUnconstrained params: {unconstrained}')

# Evaluate log prior (includes hyperpriors!)
log_prior_value = model.log_prior(unconstrained)
print(f'\nlog_prior (with hyperpriors): {log_prior_value}')

# Manual verification
constrained_ell = model.params.ell.value
print('\nManual verification:')
print(f'  constrained ell = {constrained_ell}')

# 1. LogNormalLearnable.log_prob(ell) using mu and sigma from params
ell_prior = learnable_spec.ell
ell_log_prob = ell_prior.log_prob(
    jnp.asarray(constrained_ell), learnable_params.ell, ell_prior.init_state()
)
print(f'  LogNormalLearnable.log_prob(ell) = {ell_log_prob}')

# 2. Hyperprior on mu: Normal.log_prob(mu)
mu_prior = learnable_spec.ell.mu
mu_val = learnable_params.ell.mu.value
mu_params = mu_prior.init_params()
mu_state = mu_prior.init_state()
mu_log_prob = mu_prior.log_prob(jnp.asarray(mu_val), mu_params, mu_state)
print(f'  Normal.log_prob(mu={mu_val}) = {mu_log_prob}')

# 3. Hyperprior on sigma: HalfNormal.log_prob(sigma)
sigma_prior = learnable_spec.ell.sigma
sigma_val = learnable_params.ell.sigma.value
sigma_params = sigma_prior.init_params()
sigma_state = sigma_prior.init_state()
sigma_log_prob = sigma_prior.log_prob(
    jnp.asarray(sigma_val), sigma_params, sigma_state
)
print(f'  HalfNormal.log_prob(sigma={sigma_val}) = {sigma_log_prob}')

# 4. Jacobian corrections
log_transform = Log()
jacobian_ell = log_transform.log_det_jacobian(unconstrained['ell']['value'])
jacobian_sigma = log_transform.log_det_jacobian(
    unconstrained['ell']['sigma']['value']
)
print(f'  Jacobian (ell Log transform) = {jacobian_ell}')
print(f'  Jacobian (sigma Log transform) = {jacobian_sigma}')

manual_total = (
    ell_log_prob + mu_log_prob + sigma_log_prob + jacobian_ell + jacobian_sigma
)
print(f'\n  Manual total = {manual_total}')
print(f'  Model.log_prior = {log_prior_value}')
print(f'  Match: {jnp.allclose(manual_total, log_prior_value)}')


# =============================================================================
# Case 5: NoPrior (parameter without distribution)
# =============================================================================
print('\n' + '=' * 60)
print('Case 5: NoPrior (parameter without distribution)')
print('=' * 60)


class NoPriorSpec(Spec):
    x: NoPrior[float]


no_prior_spec = NoPriorSpec(x=NoPrior(value=42.0))
no_prior_params = no_prior_spec.init_params()

print(f'Params.x.value: {no_prior_params.x.value}')
print(f'tree_leaves: {jax.tree.leaves(no_prior_params)}')
print(
    f'log_prob: {no_prior_spec.x.log_prob(jnp.asarray(no_prior_params.x.value), no_prior_params.x, no_prior_spec.x.init_state())}'
)


print('\n' + '=' * 60)
print('All cases completed successfully!')
print('=' * 60)
