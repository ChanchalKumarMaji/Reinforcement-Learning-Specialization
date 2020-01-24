"""An abstract class that specifies the optimizer API for Agent.
"""

from abc import ABCMeta, abstractmethod

class BaseOptimizer:
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def optimizer_init(self, optimizer_info):
		"""Setup for the optimizer."""

	@abstractmethod
	def update_weights(self, weights, g):
		"""
        Given weights and update g, return updated weights
        """
