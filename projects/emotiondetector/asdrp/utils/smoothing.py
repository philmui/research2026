"""Temporal smoothing filters for reducing noise in time-series data.

This module provides various smoothing filters for temporal data, including
moving average filters, exponential moving average, and Kalman filters.
These are useful for smoothing facial landmarks and emotion predictions over time.
"""

from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt


class MovingAverageFilter:
    """Moving average filter for temporal smoothing.

    Applies a simple moving average (arithmetic mean) over a sliding window
    of recent values. This filter gives equal weight to all values in the window.

    Attributes:
        window_size: Number of values to include in the moving average.
        values: Queue storing the recent values.

    Example:
        >>> filter = MovingAverageFilter(window_size=3)
        >>> print(filter.update(1.0))
        1.0
        >>> print(filter.update(2.0))
        1.5
        >>> print(filter.update(3.0))
        2.0
        >>> print(filter.update(4.0))
        3.0
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialize the moving average filter.

        Args:
            window_size: Number of values to include in the moving average.
                        Must be at least 1.

        Raises:
            ValueError: If window_size is less than 1.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}")

        self.window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        """Update the filter with a new value and return the smoothed result.

        Args:
            value: New value to add to the filter.

        Returns:
            Smoothed value (mean of values in the window).
        """
        self.values.append(value)
        return float(np.mean(self.values))

    def update_array(self, value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Update the filter with a new array value and return the smoothed result.

        This method handles multi-dimensional arrays by storing them in a list
        and computing element-wise means.

        Args:
            value: New array value to add to the filter.

        Returns:
            Smoothed array (mean of arrays in the window).
        """
        if not hasattr(self, "_array_values"):
            self._array_values: deque[npt.NDArray[np.float32]] = deque(maxlen=self.window_size)

        self._array_values.append(value.copy())
        return np.mean(self._array_values, axis=0).astype(np.float32)

    def reset(self) -> None:
        """Reset the filter by clearing all stored values."""
        self.values.clear()
        if hasattr(self, "_array_values"):
            self._array_values.clear()

    @property
    def is_full(self) -> bool:
        """Check if the filter window is full."""
        return len(self.values) == self.window_size

    @property
    def current_value(self) -> Optional[float]:
        """Get the most recently added value, or None if empty."""
        return self.values[-1] if self.values else None

    def __repr__(self) -> str:
        """String representation of the filter."""
        return f"MovingAverageFilter(window_size={self.window_size}, filled={len(self.values)})"


class ExponentialMovingAverageFilter:
    """Exponential moving average (EMA) filter for temporal smoothing.

    Applies exponential weighting where recent values have more influence than
    older values. The smoothing parameter alpha controls the decay rate.

    The formula is: EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}

    Attributes:
        alpha: Smoothing factor between 0 and 1. Higher values give more weight
              to recent observations. alpha=1 means no smoothing (use raw value),
              alpha=0 means complete smoothing (never update).
        current_ema: Current exponential moving average value.

    Example:
        >>> filter = ExponentialMovingAverageFilter(alpha=0.3)
        >>> print(filter.update(1.0))
        1.0
        >>> print(f"{filter.update(2.0):.2f}")
        1.30
        >>> print(f"{filter.update(3.0):.2f}")
        1.81
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Initialize the exponential moving average filter.

        Args:
            alpha: Smoothing factor between 0 and 1. Higher values (e.g., 0.7)
                  respond quickly to changes, lower values (e.g., 0.1) provide
                  more smoothing.

        Raises:
            ValueError: If alpha is not in range [0, 1].
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

        self.alpha = alpha
        self.current_ema: Optional[float] = None

    def update(self, value: float) -> float:
        """Update the filter with a new value and return the smoothed result.

        Args:
            value: New value to add to the filter.

        Returns:
            Smoothed value using exponential moving average.
        """
        if self.current_ema is None:
            # First value initializes the EMA
            self.current_ema = value
        else:
            # Apply exponential moving average formula
            self.current_ema = self.alpha * value + (1 - self.alpha) * self.current_ema

        return self.current_ema

    def update_array(self, value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Update the filter with a new array value and return the smoothed result.

        Args:
            value: New array value to add to the filter.

        Returns:
            Smoothed array using exponential moving average.
        """
        if self.current_ema is None:
            # First value initializes the EMA
            self.current_ema = value.copy()
        else:
            # Apply exponential moving average formula element-wise
            self.current_ema = self.alpha * value + (1 - self.alpha) * self.current_ema

        return self.current_ema.astype(np.float32)

    def reset(self) -> None:
        """Reset the filter by clearing the stored EMA value."""
        self.current_ema = None

    @property
    def is_initialized(self) -> bool:
        """Check if the filter has been initialized with at least one value."""
        return self.current_ema is not None

    def __repr__(self) -> str:
        """String representation of the filter."""
        return f"ExponentialMovingAverageFilter(alpha={self.alpha}, initialized={self.is_initialized})"


class KalmanFilter:
    """Simple 1D Kalman filter for temporal smoothing.

    A Kalman filter is an optimal estimator that uses a series of measurements
    observed over time to produce estimates that tend to be more accurate than
    those based on a single measurement alone.

    This implementation is a simple 1D constant-position Kalman filter suitable
    for smoothing time-series data like landmark coordinates or emotion scores.

    Attributes:
        process_variance: Process noise variance (Q). Models uncertainty in the
                         system dynamics. Higher values allow faster tracking.
        measurement_variance: Measurement noise variance (R). Models uncertainty
                             in the measurements. Higher values trust measurements less.
        estimate: Current state estimate.
        error_covariance: Current error covariance estimate.

    Example:
        >>> filter = KalmanFilter(process_variance=0.01, measurement_variance=0.1)
        >>> print(filter.update(1.0))
        1.0
        >>> print(f"{filter.update(1.5):.3f}")
        1.050
        >>> print(f"{filter.update(2.0):.3f}")
        1.571
    """

    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 0.1,
        initial_estimate: float = 0.0,
        initial_error_covariance: float = 1.0,
    ) -> None:
        """Initialize the Kalman filter.

        Args:
            process_variance: Process noise variance (Q). Higher values allow
                            the filter to adapt more quickly to changes.
                            Typical range: 0.001 to 0.1.
            measurement_variance: Measurement noise variance (R). Higher values
                                 indicate less trust in measurements and more
                                 smoothing. Typical range: 0.01 to 1.0.
            initial_estimate: Initial state estimate. Default is 0.0.
            initial_error_covariance: Initial error covariance. Default is 1.0.

        Raises:
            ValueError: If variances are not positive.
        """
        if process_variance <= 0:
            raise ValueError(f"process_variance must be positive, got {process_variance}")
        if measurement_variance <= 0:
            raise ValueError(f"measurement_variance must be positive, got {measurement_variance}")

        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error_covariance = initial_error_covariance
        self._is_initialized = False

    def update(self, measurement: float) -> float:
        """Update the filter with a new measurement and return the filtered estimate.

        Args:
            measurement: New measurement value.

        Returns:
            Filtered estimate after incorporating the measurement.
        """
        if not self._is_initialized:
            # Initialize with first measurement
            self.estimate = measurement
            self._is_initialized = True
            return self.estimate

        # Prediction step
        # (For constant-position model, prediction is just the previous estimate)
        predicted_estimate = self.estimate
        predicted_error_covariance = self.error_covariance + self.process_variance

        # Update step
        # Calculate Kalman gain
        kalman_gain = predicted_error_covariance / (
            predicted_error_covariance + self.measurement_variance
        )

        # Update estimate with measurement
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)

        # Update error covariance
        self.error_covariance = (1 - kalman_gain) * predicted_error_covariance

        return self.estimate

    def update_array(self, measurement: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Update the filter with a new array measurement.

        This method applies the Kalman filter independently to each element
        of the input array.

        Args:
            measurement: New measurement array.

        Returns:
            Filtered estimate array after incorporating the measurement.
        """
        if not hasattr(self, "_array_estimate"):
            # Initialize array states on first call
            self._array_estimate = measurement.copy()
            self._array_error_covariance = np.ones_like(measurement)
            self._array_is_initialized = False

        if not self._array_is_initialized:
            # Initialize with first measurement
            self._array_estimate = measurement.copy()
            self._array_is_initialized = True
            return self._array_estimate

        # Prediction step
        predicted_estimate = self._array_estimate
        predicted_error_covariance = self._array_error_covariance + self.process_variance

        # Update step
        # Calculate Kalman gain
        kalman_gain = predicted_error_covariance / (
            predicted_error_covariance + self.measurement_variance
        )

        # Update estimate with measurement
        self._array_estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)

        # Update error covariance
        self._array_error_covariance = (1 - kalman_gain) * predicted_error_covariance

        return self._array_estimate.astype(np.float32)

    def reset(self) -> None:
        """Reset the filter to its initial state."""
        self.estimate = 0.0
        self.error_covariance = 1.0
        self._is_initialized = False
        if hasattr(self, "_array_estimate"):
            delattr(self, "_array_estimate")
            delattr(self, "_array_error_covariance")
            delattr(self, "_array_is_initialized")

    @property
    def is_initialized(self) -> bool:
        """Check if the filter has been initialized with at least one measurement."""
        return self._is_initialized

    def __repr__(self) -> str:
        """String representation of the filter."""
        return (
            f"KalmanFilter(process_variance={self.process_variance}, "
            f"measurement_variance={self.measurement_variance}, "
            f"initialized={self.is_initialized})"
        )


class MedianFilter:
    """Median filter for temporal smoothing.

    Applies a median filter over a sliding window of recent values. This filter
    is particularly effective at removing outliers and spike noise while
    preserving edges better than moving average filters.

    Attributes:
        window_size: Number of values to include in the median calculation.
        values: Queue storing the recent values.

    Example:
        >>> filter = MedianFilter(window_size=3)
        >>> print(filter.update(1.0))
        1.0
        >>> print(filter.update(5.0))  # Potential outlier
        3.0
        >>> print(filter.update(2.0))
        2.0
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialize the median filter.

        Args:
            window_size: Number of values to include in the median calculation.
                        Must be at least 1. Odd values are recommended.

        Raises:
            ValueError: If window_size is less than 1.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}")

        self.window_size = window_size
        self.values: deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        """Update the filter with a new value and return the smoothed result.

        Args:
            value: New value to add to the filter.

        Returns:
            Smoothed value (median of values in the window).
        """
        self.values.append(value)
        return float(np.median(self.values))

    def update_array(self, value: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Update the filter with a new array value and return the smoothed result.

        Args:
            value: New array value to add to the filter.

        Returns:
            Smoothed array (element-wise median of arrays in the window).
        """
        if not hasattr(self, "_array_values"):
            self._array_values: deque[npt.NDArray[np.float32]] = deque(maxlen=self.window_size)

        self._array_values.append(value.copy())
        return np.median(self._array_values, axis=0).astype(np.float32)

    def reset(self) -> None:
        """Reset the filter by clearing all stored values."""
        self.values.clear()
        if hasattr(self, "_array_values"):
            self._array_values.clear()

    @property
    def is_full(self) -> bool:
        """Check if the filter window is full."""
        return len(self.values) == self.window_size

    def __repr__(self) -> str:
        """String representation of the filter."""
        return f"MedianFilter(window_size={self.window_size}, filled={len(self.values)})"
