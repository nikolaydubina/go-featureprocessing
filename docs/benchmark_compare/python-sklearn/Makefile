install:
	pip3 install -r requirements.txt

clean:
	rm -rf macbook_2017

bench: install clean
	python3 bench.py --nsamples=1 --ntrials=10 --ntrialsgroup=100 >> macbook_2017
	python3 bench.py --nsamples=10 --ntrials=10 --ntrialsgroup=100 >> macbook_2017
	python3 bench.py --nsamples=100 --ntrials=10 --ntrialsgroup=100 >> macbook_2017
	python3 bench.py --nsamples=1000 --ntrials=10 --ntrialsgroup=100 >> macbook_2017
	python3 bench.py --nsamples=10000 --ntrials=10 --ntrialsgroup=10 >> macbook_2017
	python3 bench.py --nsamples=100000 --ntrials=10 --ntrialsgroup=10 >> macbook_2017
	python3 bench.py --nsamples=1000000 --ntrials=10 --ntrialsgroup=1 >> macbook_2017
	python3 bench.py --nsamples=5000000 --ntrials=10 --ntrialsgroup=1 >> macbook_2017
	python3 bench.py --nsamples=15000000 --ntrials=10 --ntrialsgroup=1 >> macbook_2017