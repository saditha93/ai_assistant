---
doc_id: SPEC-001
title: CrestPay Instant transfers product specification
document_type: product_spec
department: payments
access_level: internal
created_date: 2025-05-01
owner: Sarah O'Connell
tags: [product, payments, instant-transfer]
---

## Overview
CrestPay Instant transfers allow customers to send funds between Crestline accounts and to other participating banks in under 10 seconds, available 24/7 including weekends and public holidays. The product is built on the payments platform described in ARCH-001, using payment-orchestrator and payments-ledger-db.

## Customer Experience
Customers initiate a transfer from the mobile app (ARCH-003) or internet banking, selecting a recipient by account number or registered mobile number. A confirmation screen displays the recipient name for verification before the customer confirms the transfer. Funds typically arrive in the recipient's account within 10 seconds; the app displays a real-time status indicator.

## Limits
Standard customers may transfer up to LKR 500,000 per transaction and LKR 2,000,000 per day. Premium tier customers have a per-transaction limit of LKR 2,000,000. Limits can be temporarily raised by contacting customer service, subject to additional verification.

## Reliability Targets
The product targets 99.9% availability and a 95th percentile completion time under 5 seconds. Peak period performance, such as salary days and festive seasons, has historically fallen short of this target due to payments-ledger-db capacity constraints; see INC-2025-041 and INC-2026-013 for the relevant incident history and RB-002 for the standing mitigation procedure.

## Fraud Controls
Every instant transfer is screened by the fraud-engine synchronously before funds move. Transfers flagged as high risk are held for manual review, which may delay the standard 10-second target; customers are notified of the hold via push notification.

## Roadmap
Planned enhancements for late 2026 include recipient nickname support and a scheduled or recurring transfer option. Both are pending architecture review board sign-off, following the same governance process used for the circuit breaker adoption discussed at MTG-2026-009.
