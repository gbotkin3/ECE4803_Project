default:
	rm -rf results/figures
	rm -f ./results/Models_Performance
	mkdir results/figures
	cd code; python3 toplevel.py

