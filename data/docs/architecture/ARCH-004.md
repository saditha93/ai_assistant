---
doc_id: ARCH-004
title: Key management and HSM design
document_type: architecture
department: security
access_level: restricted
created_date: 2025-09-10
owner: Fathima Rizwan
tags: [security, key-management, hsm, restricted]
---

## Overview
This document describes the design of Crestline's key management service and its use of hardware security modules (HSMs) for card tokenisation, TLS certificate private keys, and mobile authentication token signing. Distribution of this document is restricted to the security team and named architecture reviewers; see POL-002 for the definition of the restricted classification.

## HSM Deployment
Two HSM clusters are deployed in an active-active configuration across the Colombo primary data centre and the Kandy secondary site. All key material used for card tokenisation and payment data encryption, referenced in ARCH-001, is generated and stored exclusively within the HSM boundary; keys are never exported in plaintext form under any circumstance.

## Key Types and Rotation
Card tokenisation keys rotate on a 12-month schedule. Mobile authentication signing keys (ARCH-003) rotate on a 90-day schedule with overlapping validity to support graceful client-side transition. TLS private keys for externally facing services are generated within the HSM boundary where the service supports it; the CrestPay API gateway's certificate, involved in INC-2026-005, is a case where the private key is HSM-backed but the expiry tracking process itself was not, which is the gap RB-003 now addresses.

## Access Control
Access to HSM management functions requires two-person authorisation for any key ceremony, whether generation, rotation, or destruction. All access is logged and reviewed monthly by the security team per POL-001. No individual, including HSM administrators, can single-handedly extract or use key material outside its defined operational boundary.

## Incident Coordination
Any suspected compromise of key material is treated as a minimum SEV1 event under POL-003 and requires immediate engagement of the CISO, Amali Fernando, regardless of time of day.

## Related Documents
See ARCH-001 for how tokenisation keys are used in the payments flow, ARCH-003 for mobile authentication key usage, and POL-001 and POL-002 for the governing security and data classification policies.
