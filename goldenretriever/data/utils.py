from typing import Any, Iterable, List, Optional, Union

import numpy as np


class NegativeSampler:
    def __init__(
        self, num_elements: int, probabilities: Optional[Union[List, np.ndarray]] = None
    ):
        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)

        if probabilities is None:
            # probabilities should sum to 1
            probabilities = np.random.random(num_elements)
            probabilities /= np.sum(probabilities)
        self.probabilities = probabilities

    def __call__(
        self,
        sample_size: int,
        num_samples: int = 1,
        probabilities: np.array = None,
        exclude: List[int] = None,
    ) -> np.array:
        """
        Fast sampling of `sample_size` elements from `num_elements` elements.
        The sampling is done by randomly shifting the probabilities and then
        finding the smallest of the negative numbers. This is much faster than
        sampling from a multinomial distribution.

        Args:
            sample_size (`int`):
                number of elements to sample
            num_samples (`int`, optional):
                number of samples to draw. Defaults to 1.
            probabilities (`np.array`, optional):
                probabilities of each element. Defaults to None.
            exclude (`List[int]`, optional):
                indices of elements to exclude. Defaults to None.

        Returns:
            `np.array`: array of sampled indices
        """
        if probabilities is None:
            probabilities = self.probabilities

        if exclude is not None:
            probabilities[exclude] = 0
            # re-normalize?
            # probabilities /= np.sum(probabilities)

        # replicate probabilities as many times as `num_samples`
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        # get random shifting numbers & scale them correctly
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - replicated_probabilities
        sampled_indices = np.argpartition(shifted_probabilities, sample_size, axis=1)[
            :, :sample_size
        ]
        return sampled_indices


def batch_generator(samples: Iterable[Any], batch_size: int) -> Iterable[Any]:
    """
    Generate batches from samples.

    Args:
        samples (`Iterable[Any]`): Iterable of samples.
        batch_size (`int`): Batch size.

    Returns:
        `Iterable[Any]`: Iterable of batches.
    """
    batch = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []

    # leftover batch
    if len(batch) > 0:
        yield batch
