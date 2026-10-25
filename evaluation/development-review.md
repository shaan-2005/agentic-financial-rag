# Development retrieval review

Reviewed the 12 failures in publisher-filtered dense retrieval and the remaining
hybrid failures. This is an assistant review, not independent human validation.
The benchmark JSON and its scoring rules were left unchanged for this experiment.
Do not interpret exact-excerpt recall as answer accuracy.

| Development cases | Observation in retrieved passages | Interpretation |
|---|---|---|
| gs02–gs05 | Broad summaries, unrelated growth history, or charts replace the annotated forecast-and-trend passages. Some summaries contain only the forecast. | Partial support is insufficient when the question requires both forecast and trend. Hybrid recovers all four annotated passages. |
| gs06 | Dense results are broad or about earnings. Hybrid retrieves an explicit 4.1% emerging-market GDP forecast on printed page 68. | The hybrid answer-supporting alternative is missed by the exact wording label. Its question asks the forecast, while the annotation also mentions trend. Retain the score for comparability; add this alternative only in a separately reviewed benchmark version. |
| jp05 | Dense results discuss bonds, disclosures, or correlation charts, but omit the annotated inflation-shock sentence. | Genuine evidence miss in the inspected results; both new strategies recover it. |
| jp06 | Dense finds page 47, but its chunk ends at “low correlations”; the annotated completion “to stocks and bonds” is outside that chunk. Charts are also retrieved, but their flattened text does not reliably establish the complete asset relationships. | Chunk-boundary/context problem plus conservative labels. Both new strategies recover the complete annotated passage. |
| im06 | Dense/hybrid retrieve euro-area interest-rate assumptions and adjacent growth commentary, rather than the required pair of GDP forecasts. | Topic and metric confusion. Reranking recovers the annotated pair. |
| cmp01 | Global retrieval favours publication details; hybrid finds the Goldman evidence but not the annotated IMF global forecast in the final six. | Source balancing helps coverage but does not guarantee relevance. |
| cmp02 | Growth summaries and historical figures replace the requested source-specific 2025 forecasts. | Both strategies still miss the annotated source pair. |
| cmp03 | Adjacent euro-area commentary and inflation charts crowd out one growth passage. | Hybrid/reranked search finds the IMF item but not the Goldman item within each source's three-slot quota. |
| cmp04 | Dense includes Goldman page 89 with further BOJ tightening expected in 2025, an alternative to the annotated page 56 wording. It does not establish both sources' positions. | Dense score understates Goldman support. Both new strategies recover both annotated items. |
| im01–im02 (new hybrid regressions) | Country-specific GDP discussions displace the annotated global annual/Q4-over-Q4 passage. | Hybrid also introduces regressions; it is not uniformly better. |

The review supports improving ranking and preserving surrounding context. It does
not justify accepting arbitrary matching numbers: publisher, geography, forecast
year, annual versus quarterly basis, and GDP versus earnings must all agree.
Short annotations such as im06 also need stronger subject/context validation in a
future benchmark version. Keep original and revised-label results separate.

Independent review of alternative evidence and a broader question set remain
necessary before making accuracy claims. The 16 reserved test queries were not run.
