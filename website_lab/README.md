# AquaShield Research Lab Demo Site

A polished static multi-page website created for:
1. academic grading and presentation quality
2. web-scraping detection experiments

## Where to place it

Put this folder inside your main project, for example:

```text
web_scraping_detector_project/
└── website_lab/
```

You can rename this folder to `website_lab` if you want.

## Fastest way to run locally

From inside the website folder:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/index.html
```

## Best way for collecting logs

Serve this folder with **Nginx** so you can collect an access log.

Example Linux Nginx root:

```nginx
server {
    listen 80;
    server_name localhost;
    root /path/to/website_lab;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    access_log /var/log/nginx/access.log;
}
```

## Suggested placement in your project

```text
web_scraping_detector_project/
├── src/
├── data/
├── website_lab/
└── README.md
```

## Pages included

- Home
- Products
- 6 product detail pages
- Articles
- 3 article pages
- About
- FAQ
- Contact
- Search
- Hidden diagnostic page (can be used later as a honeypot)

## Notes

- All images are local SVG files.
- The site works offline.
- Search, filter, FAQ toggles, and contact form validation work in the browser.
- A hidden diagnostic page is included for future research use.
