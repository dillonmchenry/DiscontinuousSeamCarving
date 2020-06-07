import numpy as np
import disc_video_carving


def carve_seams(frame):
	seams = [[] for i in range(frame.shape[1])]
	energies = np.zeros(frame.shape[1])

	min_energy = np.amin(frame[0, 0:2])
	y = np.where(frame[0, 0:2] == min_energy)[0][0]  # calculates seam for top left corner
	seams[0].append([0, y])
	seams[0].append([1, 0])
	energies[0] += min_energy + frame[1][0]

	for j in range(1, frame.shape[1] - 1):
		min_energy = np.amin(frame[0, j - 1:j + 2])
		y = j - 1 + np.where(frame[0, j - 1:j + 2] == min_energy)[0][
			0]  # calculates initial seams for the second row, from the first row
		seams[j].append([0, y])
		seams[j].append([1, j])
		energies[j] += min_energy + frame[1][j]

	min_energy = np.amin(frame[0, frame.shape[1] - 2:frame.shape[1]])
	y = frame.shape[1] - 2 + np.where(frame[0, frame.shape[1] - 2:frame.shape[1]] == min_energy)[0][
		0]  # calculates seam from top left
	seams[frame.shape[1] - 1].append([0, y])
	seams[frame.shape[1] - 1].append([1, frame.shape[1] - 1])
	energies[frame.shape[1] - 1] += min_energy + frame[1][frame.shape[1] - 1]

	for i in range(2, frame.shape[0]):  # loops through all subsequent rows
		new_seams = [[] for i in range(frame.shape[1])]
		new_energies = np.zeros(frame.shape[1])

		min_energy = np.amin(energies[0:2])
		y = np.where(energies[0:2] == min_energy)[0][0]  # calculates the new seam to the left
		new_seams[0] = seams[y].copy()
		new_seams[0].append([i, 0])
		new_energies[0] = min_energy + frame[i][0]

		for j in range(1, frame.shape[1] - 1):
			min_energy = np.amin(energies[j - 1:j + 2])
			y = j - 1 + np.where(energies[j - 1:j + 2] == min_energy)[0][0]  # calculates all middle seams
			new_seams[j] = seams[y].copy()
			new_seams[j].append([i, j])
			new_energies[j] = min_energy + frame[i][j]

		min_energy = np.amin(energies[frame.shape[1] - 2:frame.shape[1]])
		y = frame.shape[1] - 2 + np.where(energies[frame.shape[1] - 2:frame.shape[1]] == min_energy)[0][
			0]  # calculates last seam of a row
		new_seams[frame.shape[1] - 1] = seams[y].copy()
		new_seams[frame.shape[1] - 1].append([i, frame.shape[1] - 1])
		new_energies[frame.shape[1] - 1] = min_energy + frame[i][frame.shape[1] - 1]

		seams = new_seams.copy()
		energies = new_energies[:]

	return (seams, energies)

def get_n_seams(seams, energies, n):
	newSeams = []
	modifiedEnergies = energies.copy()
	newEnergies = []
	if (energies.shape[0] < n):
		raise Exception("Cannot get " + str(n) + " seams. Only " + energies.shape[0] + " exist")
	maxEnergy = np.max(energies)
	for i in range(0, n):
		minIndex = np.where(modifiedEnergies == np.min(modifiedEnergies))[0][0]
		modifiedEnergies[minIndex] = maxEnergy + 1
		newSeams.append(seams[minIndex])
		newEnergies.append(energies[minIndex])
	return (newSeams, newEnergies)

if __name__ == "__main__":
	frame = np.random.randint(low=1, high=100, size=30)
	frame = np.reshape(frame, (5,6))
	print(frame)
	seam, energies = carve_seams(frame)
	print(seam)
	mySeams = get_n_seams(seam, energies, 1)
	temporal_mapg = disc_video_carving.compute_temporal_coherence_cost(frame, mySeams[0][0])


	print(energies)