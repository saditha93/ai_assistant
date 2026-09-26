---
doc_id: ARCH-002
title: Core banking integration architecture
document_type: architecture
department: platform
access_level: internal
created_date: 2025-07-18
owner: Suresh Kumar
tags: [core-banking, integration, architecture]
---

## Overview
Core banking (core-banking) is the system of record for customer accounts, balances, and ledger postings outside the payments-specific ledger. This document describes how digital and payments channels integrate with core banking, including the mobile banking platform (ARCH-003), the ATM network referenced in INC-2026-009, and internet banking.

## Integration Pattern
Channels do not connect to core banking directly. All requests route through a core banking integration layer that translates channel-specific requests into the core banking system's proprietary message format, a fixed-width batch and real-time hybrid protocol. Real-time balance enquiries and postings use a synchronous request-response pattern with a 5-second timeout; end-of-day batch settlement reconciles any discrepancies.

## Channels Served
The integration layer serves mobile-banking-api, internet-banking, atm-network, and payment-orchestrator for account-based transfers that do not go through the card network. Each channel has a dedicated connection pool to core banking to prevent one channel's load from starving another, a lesson drawn from the payments-ledger-db incidents described in ARCH-001.

## Availability and Resilience
Core banking runs on a mainframe-adjacent platform with a documented maintenance window every Sunday 01:00-03:00 IST, during which all real-time channels operate in a degraded read-only mode. The integration layer caches recent balance data for up to 10 minutes to support read-only mode gracefully.

## Known Constraints
Core banking transaction throughput is capped at approximately 600 TPS across all channels combined, lower than the payments platform's own 800 TPS design target. During shared peak periods such as salary days, core banking capacity, rather than the payments ledger, can become the binding constraint for account-based transfers; this was evaluated but ruled out as a contributing factor during the INC-2025-041 investigation.

## Future Work
A phased migration to a modern core banking API layer is planned for 2027, which would remove the fixed-width protocol dependency. This was discussed at a preliminary level in the architecture review board (MTG-2026-009) but has not yet been formally scoped.
