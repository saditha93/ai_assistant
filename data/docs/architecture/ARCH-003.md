---
doc_id: ARCH-003
title: Mobile banking platform architecture
document_type: architecture
department: digital
access_level: internal
created_date: 2025-08-22
owner: Malith Jayawardena
tags: [mobile-banking, architecture, digital]
---

## Overview
The CrestLine mobile banking app is served by mobile-banking-api, which handles authentication, account overview, transfers, and card management within the app. It integrates with core-banking (ARCH-002) for account data and with payment-orchestrator (ARCH-001) for card-based transfers initiated from the app.

## Authentication
Authentication uses a token-based scheme with biometric unlock on supported devices. Tokens are signed using a rotating key set managed through the key management service described in ARCH-004 (restricted). A legacy key format remains supported for accounts created before the 2023 authentication migration; a gap in test coverage for this legacy format caused INC-2026-002.

## Client Architecture
The app itself is a thin client; business logic lives server-side in mobile-banking-api. This design allows most fixes to be deployed without requiring an app store release, though authentication and session handling changes still carry elevated risk given their blast radius, as seen in INC-2026-002.

## Deployment Practices
Following INC-2026-002, all authentication-path deployments require a canary rollout of at least 5% of traffic for a minimum of 30 minutes before full rollout, with automated rollback on elevated login failure rate. This practice is now the standard referenced in the digital engineering team's deployment checklist.

## Dependencies
mobile-banking-api depends on core-banking for balance and account data, payment-orchestrator for card transfers, and event-bus for push notification triggers. It does not depend on internet-banking or share its authentication path, which is why the INC-2026-002 outage did not affect internet banking customers.

## Related Documents
See ARCH-002 for core banking integration details, ARCH-004 for key management, and INC-2026-002 for the incident that shaped current deployment safeguards.
