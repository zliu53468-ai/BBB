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

## SEO/GEO topic cluster update (2026-11-16)

Indexable page map:
- https://bgsv1.xyz/ — main 百家樂AI預測程式 and 莊家剋星 brand
- https://bgsv1.xyz/baccarat-ai-prediction.html — how the AI prediction analysis works
- https://bgsv1.xyz/baccarat-ai-winrate.html — how the prediction accuracy denominator works
- https://bgsv1.xyz/baccarat-road-guide.html — baccarat big road, derived roads and rules
- https://bgsv1.xyz/baccarat-kelly-guide.html — Kelly criterion and bankroll education
- https://bgsv1.xyz/youtube-video.html — public YouTube thumbnail and description

Each page has its own title, description, canonical, OG fields and visible original topic-matched content. Guide pages also have WebPage and BreadcrumbList schema. The homepage uses WebSite structured data to state the brand. Link between relevant pages naturally; never create misleading hidden keyword pages or fake performance metrics.

Search snippets cannot be dictated. Google can use meta descriptions or generate snippets from visible text depending on query. The homepage adds data-nosnippet to the original technical prediction app and lengthy feedback/details, so those regions should not be used as snippets. This is not source-code secrecy: publicly served JavaScript remains readable.

Messaging previews on LINE, WhatsApp or Telegram often cache old OpenGraph tags. The revised og:title, og:description, og:image and og:site_name identify the brand without internal model versions. Cached previews may not update immediately and are different from a Google search result.

Google AI Overviews and AI Mode use normal crawl/index and SEO best practices, not a magical separate GEO keyword list or llms.txt requirement. Accurate specific answers, honest first-party product details, structured navigation and original useful content are more valuable than keyword stuffing.

Search Console operations that still require website-owner access:
1. Verify the domain property bgsv1.xyz in https://search.google.com/search-console/welcome using a DNS TXT record.
2. Submit https://bgsv1.xyz/sitemap.xml.
3. Inspect the homepage and new guides; use Test Live URL and Request Indexing where suitable.
4. Review indexing, impressions, CTR, average query position, real search queries and mobile usability weekly.
5. Improve pages based on real evidence, avoid fake link schemes/traffic and comply with relevant gambling laws.

Neither publishing sitemap/structured data nor any SEO/GEO service can guarantee indexing or a number-one ranking.

Official references:
https://developers.google.com/search/docs/appearance/snippet
https://developers.google.com/search/docs/appearance/site-names
https://developers.google.com/search/docs/appearance/ai-features
https://developers.google.com/search/docs/fundamentals/ai-optimization-guide
https://developers.google.com/search/docs/crawling-indexing/special-tags
