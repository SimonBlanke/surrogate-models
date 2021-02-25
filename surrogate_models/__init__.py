# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

__version__ = "0.0.1"
__license__ = "MIT"


from .surrogate_models import GPySurrogateModel, GPflowSurrogateModel


__all__ = [
    "GPySurrogateModel",
    "GPflowSurrogateModel",
]
