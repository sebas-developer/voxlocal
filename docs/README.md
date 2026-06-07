# VoxLocal Documentation

A documentation site for VoxLocal built with [Fumadocs](https://fumadocs.vercel.app/) and deployed to GitHub Pages.

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm start
```

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the `main` branch in the `docs/` directory.

### Manual Deployment

```bash
# Build
npm run build

# The static site is in the `out/` directory
```

## Structure

```
docs/
├── app/                    # Next.js app directory
├── content/docs/           # MDX documentation
│   ├── index.mdx          # Getting Started
│   ├── installation.mdx   # Installation guide
│   ├── quickstart.mdx     # Quick start guide
│   ├── concepts/          # Core concepts
│   ├── api/               # API reference
│   ├── guides/            # Tutorials
│   └── examples/          # Code examples
├── lib/                    # Source configuration
└── next.config.mjs        # Next.js configuration
```

## Writing Documentation

Documentation is written in MDX format. Each file has frontmatter:

```mdx
---
title: Page Title
description: Page description
---

Content goes here.
```

### Navigation

Edit `content/docs/meta.json` to control sidebar navigation.
