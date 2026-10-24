"""Errors that may be shown to users without exposing provider internals."""


class ScoutError(Exception):
    """Base class for actionable Nemotron Scout errors."""


class ConfigurationError(ScoutError):
    """Missing or invalid configuration."""


class ModelError(ScoutError):
    """A model request or response failed; message is safe to display."""


class SecurityError(ScoutError):
    """Unsafe or unresolvable destination."""


class FetchError(ScoutError):
    """A page could not be downloaded safely."""


class BrowserError(ScoutError):
    """A browser could not safely render the page."""


class DiscoveryError(ScoutError):
    """A request could not be resolved to a usable public page."""


class RobotsError(FetchError):
    """Robots rules deny access or could not be verified."""
