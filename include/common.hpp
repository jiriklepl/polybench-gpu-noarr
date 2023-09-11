#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <memory>

class virtual_experiment {
protected:
	struct experiment_data {
		virtual ~experiment_data() = default;

		virtual void run() = 0;
		virtual void print_results(std::ostream& os) = 0;
	};

public:
	virtual_experiment() = default;
	virtual ~virtual_experiment() = default;

	void run() {
		data->run();
	}

	void print_results(std::ostream& os) {
		data->print_results(os);
	}

protected:
	std::unique_ptr<experiment_data> data;
};

std::unique_ptr<virtual_experiment> make_experiment(int argc, char *argv[]);

#endif // COMMON_HPP