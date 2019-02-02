import matplotlib.pyplot as plt

def plot(data_df):
	data_df.plot(kind="scatter",x='X',y='Y',color="blue")
	plt.show()
	return