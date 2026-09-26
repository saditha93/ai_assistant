---
doc_id: SPEC-002
title: QR merchant payments product specification
document_type: product_spec
department: payments
access_level: internal
created_date: 2025-06-20
owner: Chandima Ratnayake
tags: [product, payments, qr, merchant]
---

## Overview
QR merchant payments allow customers to pay participating merchants by scanning a static or dynamic QR code from the CrestLine mobile app. The product shares the payment-orchestrator and payments-ledger-db infrastructure described in ARCH-001 with CrestPay Instant transfers (SPEC-001).

## Merchant Onboarding
Merchants apply through their relationship manager or a self-service portal. Onboarding includes settlement account setup and, for static QR codes, generation of a fixed merchant QR image; dynamic QR codes are generated per transaction with an embedded amount and expire after 5 minutes.

## Transaction Flow
The customer scans the QR code, confirms the amount and merchant name in-app, and authorises with biometric or PIN confirmation. The transaction routes through payment-orchestrator the same way as a card transaction, including the fraud-engine check, but settles directly to the merchant's account rather than through NorthGate Card Switch, so QR payments are not affected by NorthGate-related incidents such as INC-2025-052.

## Settlement
Merchant settlement occurs on a T+1 basis by default, with same-day settlement available to merchants on the premium merchant tier for an additional fee. Settlement relies on the same payments-ledger-db as other payment products and is therefore subject to the same capacity constraints described in RB-002.

## Fees
The standard merchant fee is 0.5% per transaction, lower than card-present fees, reflecting the lower cost of the QR rail compared to card network processing.

## Known Issues
QR payments experienced a minor capacity-related slowdown in INC-2025-028 during a merchant promotional campaign that drove unexpectedly high volume; this was resolved by scaling the orchestrator's QR-specific processing pool independently from card processing.

## Roadmap
Cross-border QR interoperability with select regional payment networks is under evaluation for 2027, pending regulatory approval and architecture review.
