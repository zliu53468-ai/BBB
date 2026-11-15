# Google indexing setup for bgsv1.xyz (YouTube thumbnail + BBB)

## Video strategy
- Featured YouTube link: https://www.youtube.com/watch?v=ns76j16hLTM
- Standard official thumbnail endpoint: https://i.ytimg.com/vi/ns76j16hLTM/hqdefault.jpg
- Public homepage: https://bgsv1.xyz/
- Indexable video information page: https://bgsv1.xyz/youtube-video.html
- XML sitemap: https://bgsv1.xyz/sitemap.xml
- robots.txt: https://bgsv1.xyz/robots.txt

The video may be age-restricted. The site shows a clickable thumbnail, not
a potentially broken YouTube iframe. YouTube sign-in and age verification
are respected. A thumbnail-only external link does not qualify as a Google
Video watch page, so do not add misleading VideoObject markup or claim that
Google video rich results are guaranteed.

## Submit to Google Search Console
1. Open https://search.google.com/search-console/welcome and add the
   bgsv1.xyz Domain property. Google will provide a TXT record.
2. In the domain DNS manager, paste the TXT record, save changes and
   return to Search Console to click Verify. DNS updates can take time.
   You may use a URL-prefix property with another supported verification method.
3. Under Sitemaps, submit https://bgsv1.xyz/sitemap.xml
   (or sitemap.xml when the field already includes the domain).
4. Use URL Inspection on https://bgsv1.xyz/ and on
   https://bgsv1.xyz/youtube-video.html, run Test Live URL,
   then Request Indexing for indexable pages.
5. Monitor Pages / Page indexing and Performance / Search results.
   Compare queries, impressions, average position, clicks and CTR.

## Search discovery best practices
- Keep unique, truthful titles and descriptions on each crawlable page.
- Publish useful articles explaining baccarat B/P/T outcomes, reading the big
  road, limitations of probability estimates and how BBB works for visitors.
- Explain that slot games and 賽特 are different from baccarat and not
  predicted by the BBB model. Avoid keyword stuffing or guaranteed outcomes.
- Add relevant internal links from the homepage to each new page.
- Maintain visible about/contact information and disclosures as appropriate.
- Keep the site mobile-friendly and fix crawl errors and blocked pages.
- Respect gambling advertising rules and local law. Avoid paid fake traffic,
  link spam and misleading claims.

Crawling, indexing and ranking are controlled by Google. A sitemap submission
does not guarantee indexing or top rankings. This site owner must complete
Search Console verification; it has not been linked to this work.
