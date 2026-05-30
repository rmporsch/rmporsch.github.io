# Copilot instructions

## Build and validation

- Use `quarto render` from the repository root to build the full site. The generated output is written to `_site/`, which is gitignored.
- To validate a single page or post, render that file directly, for example: `quarto render posts/good-strategy-bad-strategy-review/index.qmd`.
- Use `quarto preview` when you need a live local preview while editing content.
- There is no repo-defined automated test suite or lint configuration. A clean `quarto render` is the main validation path for this repository.

## High-level architecture

- This repository is a Quarto website, configured in `_quarto.yml`. That file defines the site-level structure, navbar entries, HTML theme, and global stylesheet (`styles.css`).
- `index.qmd` is the homepage and uses Quarto `listing:` to automatically build the front page from content under `posts/`. The listing is sorted by descending date and exposes categories.
- `about.qmd` is a standalone page linked from the navbar rather than part of the post listing.
- `posts/_metadata.yml` applies shared defaults to every post in `posts/`, currently enabling frozen execution output and banner-style title blocks.
- Each post is organized as its own directory under `posts/<slug>/`, with an `index.qmd` file plus any local assets such as `images/` or a post-specific bibliography file.
- Publishing is CI-driven. `.github/workflows/publish.yml` deploys the site to GitHub Pages on pushes to `master` using Quarto’s publish action.

## Key conventions

- Add new articles as `posts/<slug>/index.qmd`, not as flat files in `posts/`, so they inherit `posts/_metadata.yml` and are picked up by the homepage listing.
- Keep post assets next to the post that uses them. Images live in the post directory, and bibliography files such as `references.bib` or `thesis.bib` are post-local.
- Use Quarto/Pandoc front matter on posts. Common fields in this repo are `title`, `description`, `author`, `date`, and `categories`, with optional fields like `image`, `image-alt`, `bibliography`, `toc`, and `abstract` when the post needs them.
- Citations are written with Pandoc citation syntax such as `[@rumelt2011]` and resolved through the bibliography declared in that post’s front matter.
- Do not treat `_site/` as source content. Edit the `.qmd`, bibliography, image, and stylesheet files instead.
- Keep every tracked `posts/*/index.qmd` renderable with valid metadata. The homepage listing scans that pattern directly, so empty placeholder post files produce render warnings and incomplete listing data.
