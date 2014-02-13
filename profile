python -m yep python/tla_test.py
google-pprof --callgrind viscosaur.so entry.py.prof > output.callgrind
kcachegrind output.callgrind
