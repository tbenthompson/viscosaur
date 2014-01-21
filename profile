python -m yep entry.py
google-pprof --callgrind viscosaur.so entry.py.prof > output.callgrind
kcachegrind output.callgrind
