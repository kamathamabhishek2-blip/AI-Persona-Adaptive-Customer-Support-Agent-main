# API Authentication Guide

## Overview

Our SaaS platform uses secure API authentication to protect access to endpoints and resources.

Authentication is required for all API requests. Clients must provide a valid API key or bearer token in the request header.

---

## Supported Authentication Methods

* API Key Authentication
* Bearer Token Authentication
* OAuth 2.0 (Enterprise Plans)

---

## Common Errors

### 401 Unauthorized

Occurs when authentication fails.

Possible reasons:

* Invalid API key
* Expired token
* Missing authorization header

### 403 Forbidden

Occurs when authentication is valid but access is restricted.

Possible reasons:

* Insufficient permissions
* Restricted endpoint access

---

## Correct Header Format

Authorization: Bearer YOUR_API_KEY

---

## Troubleshooting Steps

1. Verify API key in dashboard.
2. Ensure token is active.
3. Check authorization header formatting.
4. Confirm endpoint permissions.

---

## Best Practices

* Rotate API keys regularly.
* Avoid sharing API credentials.
* Store credentials securely.

If the issue persists, contact technical support with request logs and error details.
