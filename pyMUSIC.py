import numpy as np
from pySoundToolbox.pySoundToolbox import genAnalyticSignal

class MUSICEst:
    """
    Simple implementation of the MUSIC algorithm for acoustic waves.
    """

    def __init__(self, antennaPositions=np.array([[0.113, -0.036, -0.076, -0.113], [0.0, 0.0, 0.0, 0.0]]), angleStepsNum=720, searchInterval=(0, 2 * np.pi)):
        """
        Initializes MUSIC structure.

        @param antennaPositions positions of the antennas to an abitrary origin as real numpy array or matrix
        @param angleStepsNum amount of different angles to try. The given search interval will be split
            in angleStepsNum sub intervals
        @param searchInterval tuple (or another indexable structure with two element), which span
            an interval for angle investigation. The first element is the starting angle and the second one
            the last angle in rad. The angle space is searched at
            (searchInterval[0] - searchInterval[1]) / angleStepsNum positions.
            You should set this parameter depending on your array geometry (linear, circual, etc.).
        """

        assert searchInterval[0] < searchInterval[1]

        self.antennaPositions = np.matrix(antennaPositions)
        self.angleStepsNum = angleStepsNum
        self.searchInterval = searchInterval

    def getSteeringVector(self, angle, frequency):
        """
        Generates a steering vector for given array geometry.

        @param angle angle of sound source in rad
        @param frequency frequency of the emitted sound in Hz
        @param mode determines whether the returned vector will be testet against a covariance matrix
                or a cross-spectral matrix (possible values: 'cov', 'spec')
        @return a numpy nx1 matrix, with n as amount of antennas
        """

        antennasNum = self.antennaPositions.shape[1]
        speedOfSound = 343.2 # m/s
        doa = np.matrix([[np.cos(angle)], [np.sin(angle)]], dtype=np.float128)
        steeringVec = np.matrix(np.empty((antennasNum, 1), dtype=np.complex256))
        for i in range(antennasNum):
            # Using orthogonal projection, to get the propagation delay of the wavefront
            # A simpler implementation, because doa is already a normalized vector
            delay = (self.antennaPositions[:, i].transpose() * doa) / speedOfSound # Apply dot product matrix style
            steeringVec[i] = np.exp(-1j * 2 * np.pi * frequency * delay)
        return steeringVec

    def getSpectrum(self, noiseMat, frequency, noiseMatMode='cov'):
        """
        Creates a spectrum with to determine the doa for a given noise matrix.

        @param noiseMat noise matrix, consisting of the eigenvectors f the noise subspace
        @param frequency frequency of the searched source
        @param noiseMatMode determines the type of the matrix noiseMat is generated from.
                I am using a negative imaginary unit for wave paramerization, so the "Zeiger"
                rotates clockwise in time (that is typical for signal processing and can be also
                intepreted as negative frequency). But the DFT returns complex numbers with
                positive imaginary unit (positive frequency), so if I use a cross-spectral
                matrix to generate noiseMat the determinations are biased by pi. This behavour
                is corrected by this parameter.
                Possible values:
                    cov: a classical covariance matrix is used
                    spec: a cross-spectral matrix is used
        @param tuple (spectrum, angleSteps):
            spectrum: 1-D numpy array with the spectrum values
            angleSteps: analysed angles
        """

        if noiseMatMode not in ('cov', 'spec'):
            raise ValueError('noiseMatMode has to be "cov" or "spec!"')

        if noiseMatMode == 'spec':
            frequency = -frequency

        # Number of rows (or columns) of noise space matrix must be the number of antennas
        assert noiseMat.shape[0] == self.antennaPositions.shape[1]

        startAngle = self.searchInterval[0]
        endAngle = self.searchInterval[1]
        angleStepSize = (endAngle - startAngle) / self.angleStepsNum

        angleSteps = np.arange(startAngle, endAngle, angleStepSize)
        spectrum = np.empty(self.angleStepsNum, dtype=np.float)
        for i in range(self.angleStepsNum):
            steeringVec = self.getSteeringVector(angleSteps[i], frequency)
            # Because of precision errors, the score is sometimes complex, but the imaginary part
            # is pretty small (1e-20 or less), so just use real part
            spectrum[i] = 1 / (steeringVec.conjugate().transpose() * noiseMat * noiseMat.conjugate().transpose() * steeringVec).real

        return spectrum, angleSteps

    def getNoiseMat(self, cov, amountOfSources):
        """
        Returns the matrix consisting of the noise eigenvectors.

        @param cov covariance matrix of the signal. Needs to be a quadratic matrix. Its number of rows
            and cols is the number of array within the array.
        @param amountOfSources number of sources, so noise space can be estaminated
        @return noise matrix spanning the noise space
        """

        antennasNum = cov.shape[0]

        eigenVals, eigenVecs = np.linalg.eig(cov)
        # Bundle eigenvalues and eigenvectors
        eigenBundles = []
        for i in range(eigenVals.shape[0]):
            # Second item within the tuple just to keep strict order for sorting
            eigenBundles.append((abs(eigenVals[i]), i, np.matrix(eigenVecs[:, i]).transpose()))
        eigenBundles = sorted(eigenBundles)

        # Generate noise matrix
        noiseMat = np.matrix(np.empty((antennasNum, antennasNum - amountOfSources), dtype=np.complex))
        for i in range(antennasNum - amountOfSources):
            noiseMat[:, i] = eigenBundles[i][2]

        return noiseMat

    def getSourceAngles(self, data, amountOfSources=1, frequency=300):
        """
        Estimates the angles of arrival of the sources.

        @param data a numpy array with 24 PCM data sampled an microphone array with given geometry
            Note that the data need to be samples with approximated zero mean and the result
            of narrow band signals (preprocessing may be necessary, to get narrow band)
        @param amountOfSources number of sources
        @param frequency frequency of the signal to find
        @return a numpy array with the angles with the highest MUSIC score
        """

        # TODO currently fails, because I don't substract the peaks after finding them (because they're within a
        # curve), therefore all angles are clustered around the first occurence

        # Some helper variables to keep source code more readable
        antennasNum = self.antennaPositions.shape[1]
        samplesNum = data.shape[1]
        assert(data.shape[0] == antennasNum)

        # Numpy doesn't handle 1D arrays like 2D arrays when it comes to matrix multiplication, but I need
        # real matrix multiplication (and I ran into several pitfalls I hope to avoid with matrices)!

        # Run ASG
        analyticSignal = np.matrix(np.empty(data.shape, dtype=np.complex128))
        for i in range(antennasNum):
            analyticSignal[i, :] = np.matrix(genAnalyticSignal(data[i, :]))

        # Compute covariance matrix
        cov = (analyticSignal * analyticSignal.transpose().conjugate()) / samplesNum

        # Compute noise matrix
        noiseMat = self.getNoiseMat(cov, amountOfSources)

        # Generate spectrum
        spectrum, angleSteps = self.getSpectrum(noiseMat, frequency)

        # Pick peaks of spectrum (a heap would be more effective)
        specPeaks = np.zeros(amountOfSources)
        peakAngles = np.zeros(amountOfSources)
        for i in range(spectrum.shape[0]):
            smallestVal = spectrum[i]
            smallest = -1
            for j in range(amountOfSources):
                if smallestVal > specPeaks[j]:
                    smallest = j
                    smallestVal = specPeaks[j]
            if smallest >= 0:
                specPeaks[j] = spectrum[i]
                peakAngles[j] = angleSteps[i]

        return peakAngles
