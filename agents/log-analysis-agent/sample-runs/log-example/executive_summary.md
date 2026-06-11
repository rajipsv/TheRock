# Log Analysis Executive Summary

**Log:** `C:\Users\Rajeswari\.gemini\antigravity\scratch\TheRock\TheRock\agents\log-analysis-agent\tests\fixtures\example.log`
**Mode:** tool_only
**Preset:** custom
**Errors found:** 7

## Summary
Tool-only qualification pass found 7 highlighted error lines. Stats: path=C:\Users\Rajeswari\.gemini\antigravity\scratch\TheRock\TheRock\agents\log-analysis-agent\tests\fixtures\example.log; lines=21; bytes=1474; github_error=0; keyword_hits: ERROR=6, EXCEPTION=1, CRITICAL=1, FATAL=1, WARNING=2

## Top errors
- **Line 3** (HIGH): 2024-12-10 10:15:25 ERROR [Database] Connection failed: Connection timeout after 30s
  - Recommendation: Check connectivity, firewall, restart DB, increase pool size
- **Line 8** (HIGH): 2024-12-10 10:18:33 ERROR [PaymentService] Payment processing failed: InvalidCardException
  - Recommendation: Validate payment method, check gateway status, retry with idempotency
- **Line 11** (HIGH): Caused by: stripe.error.CardError: Your card was declined
  - Recommendation: Validate payment method, check gateway status, retry with idempotency
- **Line 15** (HIGH): 2024-12-10 10:21:30 ERROR [FileSystem] Failed to write file: /data/reports/daily.pdf - Disk quota exceeded
  - Recommendation: Free space, log rotation, archive data, increase quota
- **Line 16** (HIGH): 2024-12-10 10:22:00 FATAL [Application] Out of memory error - shutting down
  - Recommendation: Increase heap, fix leaks, paginate data, add memory monitoring
- ... and 2 more

## Knowledge base matches
- `2024-12-10 10:15:25 ERROR [Database] Connection failed: Connection timeout after` → Database connection timeout (Database, score=51.0)
- `2024-12-10 10:18:33 ERROR [PaymentService] Payment processing failed: InvalidCar` → Payment / card processing failure (Runtime, score=34.5)
- `Caused by: stripe.error.CardError: Your card was declined` → Payment / card processing failure (Runtime, score=28.0)
