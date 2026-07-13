.PHONY: test demo attack limit run capture

test:                 ## 36 unit tests, no API key, no network
	python -m unittest discover -s tests -v

demo:                 ## the whole loop, offline
	python -m order_agent.main --offline

attack:               ## the injection defence, offline
	python -m order_agent.main --offline --attack

limit:                ## the step limit firing, offline
	python -m order_agent.main --offline --runaway

run:                  ## the real thing (needs GOOGLE_API_KEY)
	python -m order_agent.main

capture:              ## live run(s) + paste the transcript into the README
	python capture_run.py
