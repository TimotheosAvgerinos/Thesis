#!/usr/bin/env bash
set -e

# Optional (only if you’ve set STATIC_ROOT in settings)
python manage.py collectstatic --noinput || true

python manage.py migrate

python manage.py runserver 0.0.0.0:8000
