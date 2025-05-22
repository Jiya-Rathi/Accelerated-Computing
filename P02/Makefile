
run:
	@echo "Running experiments..."
	@echo "n, m, result" > results1.txt
	@for n in 2 4 8 16 32 64 128; do \
		for m in $$(seq 0 18); do \
			echo "Running ./P02_Parallel $$n $$m"; \
			./P02_Parallel $$n $$m >> results.csv || exit 1; \
		done; \
	done
	@echo "All experiments completed. Results saved to results.txt."
