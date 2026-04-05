"""HIM+HER training loop.

This module will implement the main training loop for the HIM+HER agent.
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
