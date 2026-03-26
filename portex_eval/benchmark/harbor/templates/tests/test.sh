#!/bin/bash

echo "=== Agent Submission ==="
cat /app/answer.txt 2>/dev/null || echo "(no submission found at /app/answer.txt)"
echo ""
echo "=== Grading ==="

python3 /tests/portex_grade.py

if [ $? -ne 0 ]; then
  echo '{"reward": 0.0}' > /logs/verifier/reward.json
fi
