---
doc_id: ARCH-001
title: Payments platform architecture
document_type: architecture
department: payments
access_level: internal
created_date: 2025-06-02
owner: Michael Chen
tags: [payments, architecture, gateway, orchestrator]
---

## Overview
The payments platform processes card authorisations, CrestPay Instant transfers (SPEC-001), and QR merchant payments (SPEC-002). The core components are the CrestPay API gateway, the payment orchestrator, the payments-ledger-db, the NorthGate Card Switch integration, and the internal event bus. This document is the primary architecture reference cited across incident reports, for example INC-2025-041, INC-2025-052, and INC-2026-021.

## Components
The CrestPay API gateway (crestpay-gateway) terminates external TLS connections and applies rate limiting before forwarding requests to the payment orchestrator. The payment orchestrator (payment-orchestrator) coordinates the authorisation flow: it validates the request, calls the fraud engine, calls NorthGate Card Switch for card-network authorisation, and writes the resulting transaction record to payments-ledger-db. All state changes are also published to the event-bus for downstream consumers such as notifications and reconciliation.

## Data Flow
A typical card transaction flows: client to crestpay-gateway, gateway to payment-orchestrator, orchestrator to fraud-engine as a synchronous check, orchestrator to northgate-card-switch for external authorisation, orchestrator writes the result to payments-ledger-db, and orchestrator publishes an event to event-bus. The end-to-end latency target is under 2 seconds at the 95th percentile.

## Dependencies and Failure Modes
payments-ledger-db is a shared dependency for read and write paths; connection pool exhaustion here has caused multiple incidents (INC-2025-019, INC-2025-041, INC-2026-013), see RB-002. northgate-card-switch is an external dependency outside our control; timeout handling without backoff caused INC-2025-052 and was mitigated with a circuit breaker following MTG-2026-009, see RB-004. The CrestPay gateway's TLS certificate lifecycle caused INC-2026-005, addressed in RB-003.

## Capacity and Scaling
Baseline design capacity is 800 transactions per second sustained, with bursts to 2,000 TPS supported for up to 15 minutes. Known peak periods (salary days, festive seasons) can exceed 3x average daily volume; capacity planning for these periods is now a standing agenda item at the payments weekly sync (MTG-2026-014).

## Security Considerations
Card data in transit is encrypted end-to-end and tokenised at the gateway; card numbers are not intended to appear in plaintext in application logs by design, though INC-2026-021 identified a logging gap where this control was bypassed in a debug log line. Key management for tokenisation is described in ARCH-004 (restricted). See POL-002 for data classification requirements applicable to payment data.
