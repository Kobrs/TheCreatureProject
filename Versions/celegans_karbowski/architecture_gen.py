data_file = "connections_motor_karbowski.txt"


with open(data_file, 'r') as f:
	# Skip header and empty line after it
	f.next()
	f.next()
	col_header = f.next()
	row_header = f.next()
	cells 

	for line in f:
		if line != '\n':
			vals = [float(v) for v in line.strip().split('\t')]
			print vals

		else:

