
import sys

input_f = sys.argv[1]
output = sys.argv[2]

format_f = "utf-16"
try:
	with open(input_f, "r", encoding="utf-8") as f:
		for line in f:
			format_f = "utf-8"
		format_f = "utf-8"
except Exception as e:
	format_f = "utf-16"


next_line = 0

with open(input_f, "r", encoding=format_f) as infile, open(output, "w", encoding="utf-8") as outfile:
	for line in infile:
		if next_line == 1:
			outfile.write(line)
			next_line = 0
		keywords = {"[Average]", "Running model", "TTFT", "first_token_latency:", "rest_token_latency:", "TPOT:", "Throughput:", "First token latency", "Second token latency"}
		for keyword in keywords:
			if keyword in line:
				if keyword == "First token latency" or keyword == "Second token latency":
					next_line = 1
				outfile.write(line)

print("Filter completed")