"""Network-isolation guard ensuring local inference and storage endpoints stay on-premise."""

from urllib.parse import urlparse


LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class LocalEndpointViolation(ValueError):
    """Raised when an inference or storage endpoint points outside the local host."""


def is_local_host(hostname: str | None) -> bool:
    """Return True only when hostname resolves explicitly to the loopback interface."""
    if not hostname:
        return False
    return hostname.strip().lower() in LOCAL_HOSTS


def validate_local_endpoint(url: str, service_name: str = "local service") -> str:
    """
    Validate that an HTTP endpoint targets localhost/127.0.0.1/::1.

    Raises LocalEndpointViolation if an external or non-HTTP URL is configured.
    """
    if not url or not isinstance(url, str):
        raise LocalEndpointViolation(f"Missing endpoint URL for {service_name}.")

    parsed = urlparse(url.strip())

    if parsed.scheme not in {"http", "https"}:
        raise LocalEndpointViolation(
            f"Unsupported URL scheme for {service_name}: {parsed.scheme!r}."
        )

    if not is_local_host(parsed.hostname):
        raise LocalEndpointViolation(
            f"External network endpoint blocked for {service_name}: {parsed.hostname!r}. "
            "Only localhost, 127.0.0.1, or ::1 are permitted."
        )

    return url.strip()
