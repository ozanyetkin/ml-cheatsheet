import numpy as np

# Generate 1000 random samples from a normal distribution with mean 0 and standard deviation 1
normal_samples = np.random.normal(0, 1, 1000)

# Generate 1000 random samples from a uniform distribution between 0 and 1
uniform_samples = np.random.uniform(0, 1, 1000)

# Generate 1000 random samples from a binomial distribution with 10 trials and success probability 0.5
binomial_samples = np.random.binomial(10, 0.5, 1000)

# Generate 1000 random samples from a Poisson distribution with lambda 5
poisson_samples = np.random.poisson(5, 1000)

# Generate 1000 random samples from an exponential distribution with scale parameter 2
exponential_samples = np.random.exponential(2, 1000)
