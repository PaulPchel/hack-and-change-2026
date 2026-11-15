#!/bin/sh
# Убирает из проекта сломанные и лишние файлы. Запускать из корня репозитория: sh cleanup.sh
rm -f evaluate_dict.py make_ground_truth.py precompute.py submit.zip submission.csv
rm -rf submit_build __pycache__
git rm -r -q --cached --ignore-unmatch evaluate_dict.py make_ground_truth.py precompute.py \
  submit.zip submission.csv submit_build __pycache__ questions.csv ground_truth.csv 2>/dev/null
echo "Готово. Проверьте: git status"
