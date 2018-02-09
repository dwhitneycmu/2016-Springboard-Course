# **Project Overview:**
This is my culiminating capstone project when I enrolled in Springboard’s Data Science Intensive course during 2016. Springboard's course is pretty fantastic, as I was challenged not only to learn new analtyics, but to present my work with different audiences in mind. My project deliverables (see links below) involved writing a blog post summarizing my findings in a layman's format, presenting my project to fellow students, writing up a detailed Ipython notebook and capstone report for Data Science-minded readers that would like to see how I analyzed my data. I have also included all of the code used for analysis in this GitHub repository. 
<br />

# **Links:**
<br /> *<a href="https://www.springboard.com/blog/six-stocks-matter-nasdaq/">Blog post</a>
<br /> *<a href="https://www.youtube.com/watch?v=A-kx8BZkt-o">YouTube Presentation</a>
<br /> *<a href="https://github.com/dwhitneycmu/2016-Springboard-Course/blob/master/2016 Capstone Project/SlideShow/CapstoneSlideShow_Revision_NoTransitions.pptx">Presentation Slides</a>
<br /> *<a href="https://github.com/dwhitneycmu/2016-Springboard-Course/blob/master/2017 Blog Post/Springboard Blog Post.ipynb">Ipython notebook</a>
<br /> *<a href="https://github.com/dwhitneycmu/2016-Springboard-Course/tree/master/2016">Github link for code</a>
<br /> *<a href="https://github.com/dwhitneycmu/2016-Springboard-Course/blob/master/2016 Capstone Project/CapstoneReport.pdf">PDF of the project</a>

# **Project Overview:**
# For this capstone project, I had the pleasure to be mentored by Sameera Poduri, the head data-scientist at Amino Health. Early on, Sameera and I decided to try something not related to my expertise in systems neuroscience, and instead challenge myself with an unconnected topic in finance. Pretty quickily I was struck by a recent Wall Street journal article that reported just six giant technology/ pharmaceutical-related companies accounted for over half of the NASDAQ’s growth back in 2015: Facebook, Amazon, Apple, Netflix, Gilead, and Google [1]. Not surprisingly, these six high-performing stocks have received considerable attention from business analysts [1-3]. Jim Cramer from Mad Money even went so far to dub four of these fast-growing NASDAQ stocks (Facebook, Amazon, Netflix, and Google) with the acronym ‘FANG’ to emphasize the privileged position that these companies enjoy [3].

# While investors understandably are drawn to invest in popular stocks, I wondered how much growth in the NASDAQ Composite Index over the last six years was attributable just to these six stocks. Furthermore, as someone with a portion of my own 401k savings invested in these high-profile companies, I was motivated to develop the tools to examine whether stock growth in these companies is correlated with each other. While strong, consistent growth is always preferred, modern portfolio theory predicts increased risk associated with investments lacking diversification [4]. Thus, along with Sameera, we looked at some historical stock data to determine how risky these six companies might be as investments relative to other NASDAQ companies.

# **Data Summary:**
# 1.) For the last six years (2011-2017), FAANG stocks have shown extraordinary high growth rates as a group (9.39% per year, sample size of 6 stocks) relative to other Top-100 NASDAQ stocks (3.86% per year, sample size of 95 stocks) and the bulk of ordinary NASDAQ stocks (1.05% per year, sample size of 2,982 stocks).
# 2.) However, a closer inspection of individual NASDAQ stocks reveals that ~20% of ordinary NASDAQ stocks display growth rates comparable to the fast-growing FAANG stocks (4.14% per year, sample size of 555 stocks). These results suggest that recent growth in the stock market is not driven by a small handful of privileged stocks, but rather reflects a broader advance by American companies.
# 3.) Under a modern portfolio theory approach, an investment portfolio containing just FAANG stocks is a less risky investment than a portfolio including either all other Top-100 NASDAQ stocks or ordinary NASDAQ stocks.
# 4.) Nonetheless, day-to-day fluctuations in both FAANG and other Top-100 stocks equally predict NASDAQ movements (with a correlation of ~0.55). In contrast, day-to-day fluctuations in the bulk of ordinary NASDAQ stocks more weakly predict market swings (with a correlation of ~0.3). In a regression model, day-to-day fluctuations in either FAANG or other Top-100 NASDAQ stocks more strongly predicted the market by a 3x margin relative to ordinary NASDAQ stocks indicating that larger cap stocks overall better predict market movements (with a correlation of ~ 0.5).

# **References:**
# [1] D. Strumpf. Wall Street Journal. “The Only Six Stocks That Matter”. http://www.wsj.com/articles/the-only-six-stocks-that-matter-1437942926  (July 26, 2015).
# [2] A. Mirhaydari. CBS Moneywatch. “Uh oh — just 8 stocks are propping up the market”. http://www.cbsnews.com/news/uh-oh-just-8-stocks-are-propping-up-the-market/ (November 12, 2005)
# [3] C. Ciaccia. “What Are FANG Stocks and Why Does Jim Cramer Love Them?”. https://www.thestreet.com/story/13230576/1/what-are-fang-stocks-and-why-does-jim-cramer-love-them.html (July 24, 2015)
# [4] “Modern Portfolio Theory.” Investopedia.  Investopedia. Web. 21 April 2017. 
