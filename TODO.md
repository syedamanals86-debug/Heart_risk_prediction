# Hackathon readiness + UI enhancement checklist

## Plan
- Update **index.html**: add smoking status + exercise frequency + diet fields.
- Update **app.py**: compute ML score as before, then compute an adjusted score using rule-based deltas from new lifestyle inputs; update risk category/advice/avoid/actions to use adjusted score; store influence summary for UI + PDF.
- Update **result.html**: display ML score vs adjusted score + lifestyle influence summary.
- Update **style.css**: add minimal styling for new result sections.
- Smoke-test locally: run `python app.py` and submit the form; verify PDF download.

## Done
- Repo analysis complete
- (pending) Implement smoking/lifestyle UI + adjusted scoring



