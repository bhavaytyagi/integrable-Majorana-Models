#!/bin/sh
# One command that reproduces every check quoted in the paper.
# Requires Python 3.9+ and nothing else.
set -e
echo "Python: $(python3 --version)"
( cd paper_checks   && python3 check_paper_consistency.py )
( cd paper_checks   && python3 extension_test.py )
( cd paper_checks   && python3 duality_and_gf_test.py )
( cd algebra_checks && python3 commutator_tests.py )
( cd algebra_checks && python3 uniqueness_checks.py )
for n in 3 4 5; do
  ( cd HamFam_Resources/python && python3 "verify_n${n}_majorana.py" )
done
echo
echo "All checks passed. Note: solve_n4_compact_ansatz.py exits 1 BY DESIGN"
echo "and is not part of this suite; it is a negative control."
