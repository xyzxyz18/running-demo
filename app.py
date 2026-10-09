"""Start the browser UI; also exposes app:app for Gunicorn."""
from pace.web import app, main


if __name__ == "__main__":
    main()
