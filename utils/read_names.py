
def read_dino_csv(path):
	total_dataset = []
	f = open("dinosaurs.csv", "r")
	for x in f:
	  temp = ['<start>']+[*x.strip()]+['<end>']
	  temp_idx = []
	  for elm in temp:
	    idx = total_labels.index(elm)
	    temp_idx.append(idx)

	  temp_idx = temp_idx + [28]*(20-len(temp_idx))
	  total_dataset.append(temp_idx)

	return total_dataset