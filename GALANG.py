#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num=500)

uniform_dist = sts.uniform.pdf(mu) + 1
uniform_dist = uniform_dist / uniform_dist.sum()

beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.2)
beta_dist = beta_dist / beta_dist.sum()

plt.plot(mu, uniform_dist, label='Uniform Dist')
plt.plot(mu, beta_dist, label='Beta Dist')
plt.xlabel("Value of μ in meters")
plt.ylabel("Probability density")
plt.legend()
plt.show()


# In[2]:


def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)
    return likelihood_out / likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title("Likelihood of μ given observation 1.7m")
plt.ylabel("Probability density / Likelihood")
plt.xlabel("Value of μ")
plt.show()


# In[3]:


unnormalized_posterior = likelihood_out * uniform_dist

plt.plot(mu, unnormalized_posterior)
plt.xlabel("μ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()


# In[4]:


import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

mu = np.linspace(1.65, 1.8, num=500)

uniform_dist = sts.uniform.pdf(mu) + 1
uniform_dist = uniform_dist / uniform_dist.sum()

beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.2)
beta_dist = beta_dist / beta_dist.sum()

plt.plot(mu, uniform_dist, label='Uniform Dist')
plt.plot(mu, beta_dist, label='Beta Dist')
plt.xlabel("Value of μ in meters")
plt.ylabel("Probability density")
plt.legend()
plt.show()

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)
    return likelihood_out / likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title("Likelihood of μ given observation 1.7m")
plt.ylabel("Probability density / Likelihood")
plt.xlabel("Value of μ")
plt.show()

unnormalized_posterior = likelihood_out * uniform_dist

plt.plot(mu, unnormalized_posterior)
plt.xlabel("μ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()


# In[ ]:




