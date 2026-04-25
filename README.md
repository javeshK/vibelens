# 🎵 VIbeLens

AI-powered music recommendation engine that analyzes images and suggests songs matching the mood, colors, and aesthetic of the photo.

## Features

- **Image Analysis**: Extracts 6 feature vectors (emotion, color psychology, scene context, face presence, aesthetic score, crop-aware analysis)
- **Smart Recommendations**: Scores songs based on mood (30%), context (20%), trends (25%), language preference (15%), and aesthetic match (10%)
- **Multi-Language Support**: English, Hindi, and Punjabi music libraries
- **YouTube Integration**: Fetch trending music via YouTube Data API

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python run.py
```

The app will start at `http://localhost:5000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend |
| POST | `/api/analyze` | Analyze image |
| POST | `/api/recommend` | Full pipeline (analyze + recommend) |
| GET | `/api/health` | Health check |
| GET | `/api/library/stats` | Song count per language |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 5000 | Server port |
| `HOST` | 0.0.0.0 | Server host |
| `DEBUG` | true | Debug mode |
| `YOUTUBE_API_KEY` | - | YouTube Data API key (for fetcher) |

## Project Structure

```
vibelens-ai/
├── run.py              # Entry point
├── requirements.txt    # Dependencies
├── backend/
│   ├── app.py          # Flask application
│   ├── analyzer.py     # Image analysis engine
│   ├── recommender.py  # Song recommendation engine
│   ├── youtube_fetcher.py # YouTube API fetcher
│   └── trends/         # Song data JSON files
└── frontend/
    ├── static/         # Static assets
    └── templates/      # HTML templates
```

## Fetching Trending Music

```bash
python -m backend.youtube_fetcher --language english --max 200
python -m backend.youtube_fetcher --language hindi --max 200
python -m backend.youtube_fetcher --language punjabi --max 200
```

Requires `YOUTUBE_API_KEY` environment variable.

## Deployement on Render

```
This was done as to showcase to my friends of what was made 
Here is the link that you can visit

https://soulmatch-xqan.onrender.com/

I will work on getting a good domain name in future :)
