# Movie Recommender Frontend

A beautiful, responsive frontend for the AI-powered movie recommendation system. Built with Next.js 15, TypeScript, and Tailwind CSS.

## Features

- **Cold Start Flow**: Rate movies to get personalized AI recommendations
- **Movie Discovery**: Browse popular movies, filter by genre
- **Search**: Find movies by title with instant results
- **Similar Movies**: Explore movies similar to ones you like
- **Dark Mode**: Automatic dark mode based on system preferences
- **Responsive Design**: Works great on mobile, tablet, and desktop

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **Icons**: Lucide React
- **Font**: Geist (via next/font)

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create environment file:
```bash
cp .env.example .env.local
```

3. Update the API URL in `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Development

Run the development server:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── src/
│   ├── app/              # Next.js App Router
│   │   ├── globals.css   # Global styles
│   │   ├── layout.tsx    # Root layout
│   │   └── page.tsx      # Main page
│   ├── components/       # React components
│   │   ├── MovieCard.tsx
│   │   ├── SearchBar.tsx
│   │   ├── StarRating.tsx
│   │   └── Skeleton.tsx
│   ├── lib/              # Utilities
│   │   ├── api.ts        # API client
│   │   └── utils.ts      # Helper functions
│   └── types/            # TypeScript types
│       └── api.ts        # API response types
├── .env.example          # Example environment variables
└── package.json
```

## API Integration

The frontend connects to the FastAPI backend for:
- `GET /api/v1/movies/popular` - Popular movies
- `GET /api/v1/movies/search` - Search movies
- `POST /api/v1/cold-start/rate` - Cold start recommendations
- `GET /api/v1/similar/{movie_id}` - Similar movies

If the API is unavailable, the frontend falls back to demo data for demonstration purposes.

## Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/YOUR_REPO&env=NEXT_PUBLIC_API_URL&envDescription=API%20URL%20for%20the%20recommendation%20backend)

1. Push to GitHub
2. Import to Vercel
3. Set `NEXT_PUBLIC_API_URL` environment variable
4. Deploy!

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |
