# Libraries

import argparse
import copy
import librosa
import numpy
import pandas
import plotly.express
import random
import scipy.fft

# Color Palette

colorPalette = ["#CFD8DC", "#F44336", "#9C27B0", "#3F51B5", "#03A9F4", "#009688", "#8BC34A", "#FFEB3B", "#FF9800", "#795548", "#607D8B", "#000000"]

# Musical Notes

musicalNotes = {
    "C": [16.35, 32.70, 65.41, 130.81, 261.63, 523.25, 1046.50, 2093.00],
    "C#": [17.32, 34.65, 69.30, 138.59, 277.18, 554.37, 1108.73, 2217.46],
    "D": [18.35, 36.71, 73.42, 146.83, 293.66, 587.33, 1174.66, 2349.32],
    "D#": [19.45, 38.89, 77.78, 155.56, 311.13, 622.25, 1244.51, 2489.02],
    "E": [20.60, 41.20, 82.41, 164.81, 329.63, 659.26, 1318.51, 2637.02],
    "F": [21.83, 43.65, 87.31, 174.61, 349.23, 698.46, 1396.91, 2793.83],
    "F#": [23.12, 46.25, 92.50, 185.00, 369.99, 739.99, 1479.98, 2959.96],
    "G": [24.50, 49.00, 98.00, 196.00, 392.00, 783.99, 1567.98, 3135.96],
    "G#": [25.96, 51.91, 103.83, 207.65, 415.30, 830.61, 1661.22, 3322.44],
    "A": [27.50, 55.00, 110.00, 220.00, 440.00, 880.00, 1760.00, 3520.00],
    "A#": [29.14, 58.27, 116.54, 233.08, 466.16, 932.33, 1864.66, 3729.31],
    "B": [30.87, 61.74, 123.47, 246.94, 493.88, 987.77, 1975.53, 3951.07],
}

# Find Note


def findNote(inputFrequency):
    shorterDistance = float("inf")
    noteFound = None
    octave = None
    for note, frequencies in musicalNotes.items():
        for i, frequency in enumerate(frequencies):
            distance = numpy.abs(frequency - inputFrequency)
            if distance < shorterDistance:
                shorterDistance = distance
                noteFound = note
                octave = i
    return noteFound, octave


# Get K-medoids


def getMedoids(data, size, maxIterations=100):
    medoids = random.sample(range(len(data)), size)
    distances = [[numpy.abs(data[i]["Frequency"] - data[j]["Frequency"]) for j in range(len(data))] for i in range(len(data))]
    for _ in range(maxIterations):
        clusters = [[] for _ in range(size)]
        for i, item in enumerate(data):
            nearestMedoid = min(range(size), key=lambda x: distances[i][medoids[x]])
            clusters[nearestMedoid].append(i)
            newMedoids = copy.deepcopy(medoids)
            for i, cluster in enumerate(clusters):
                if cluster:
                    costs = [numpy.sum([distances[index][m] for m in cluster]) for index in cluster]
                    minCost = cluster[costs.index(min(costs))]
                    newMedoids[i] = minCost
                if newMedoids == medoids:
                    break
                medoids = copy.deepcopy(newMedoids)
        clusters = [[] for _ in range(size)]
        for i, item in enumerate(data):
            nearestMedoid = min(range(size), key=lambda x: distances[i][medoids[x]])
            clusters[nearestMedoid].append(i)
        result = []
        for i, cluster in enumerate(clusters):
            for index in cluster:
                d = copy.deepcopy(data[index])
                d["Medoid"] = str(data[medoids[i]]["Frequency"])
                result.append(d)
        return result


# Process Audio


def processAudio(path, clusterSize):
    audio, sampleRate = librosa.load(path)
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sampleRate)
    beats = librosa.frames_to_samples(beats)
    result = []
    for i, beat in enumerate(beats):
        start = beat if i != 0 else 0
        end = beats[i + 1] if i < len(beats) - 1 else len(audio)
        chunk = audio[start:end]
        fft = scipy.fft.fft(chunk)
        frequencies = scipy.fft.fftfreq(len(chunk), 1 / sampleRate)
        index = numpy.argmax(numpy.abs(fft[1:])) + 1
        result.append({"Frequency": round(frequencies[index], 2), "Amplitude": round(20 * numpy.log10(numpy.abs(fft[index])), 2)})
    for item in result[:]:
        if item["Frequency"] < 0 or item["Frequency"] > 4000:
            result.remove(item)
        item["Note"], item["Octave"] = findNote(item["Frequency"])
    result = getMedoids(result, clusterSize)
    dataFrame = pandas.DataFrame.from_dict(result)
    for item in ["Note", "Medoid"]:
        dataFrame = dataFrame.sort_values(by=item)
        dash = plotly.express.scatter(
            dataFrame,
            x="Frequency",
            y="Amplitude",
            color=item,
            color_discrete_sequence=colorPalette,
            hover_data=["Note", "Medoid"],
            size="Octave",
            template="plotly_white",
            title="Audio K-medoids (Tempo: {:.0f} beats per minute)".format(tempo),
        )
        dash.show()


# Interface

parser = argparse.ArgumentParser(prog="Audio K-medoids", description="Analyze any audio with K-medoids", add_help=False)
parser.add_argument("filepath", type=str, help="Set the WAV file path.")
parser.add_argument("-h", "--help", action="help", help="Show the help message.")
parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0.0", help="Show script version.")
parser.add_argument("-s", "--size", type=int, choices=range(2, 9), default=4, help="Set the cluster size.")
args = parser.parse_args()
processAudio(args.filepath, args.size)
